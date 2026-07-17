from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LSSODiagnostics, lsso
from .modules_v2 import apply_rank_rotary


class _GroupedLSSOBase(nn.Module):
    """Shared implementation for grouped-relation LSSO variants.

    ``num_heads`` keeps the surrounding encoder's channel partition, while
    ``num_relation_groups`` controls how many independent relation fields and
    small linear systems are built.  Content channels remain distinct: heads
    assigned to the same relation group are concatenated as multiple right-hand
    sides of one structured low-rank solve.
    """

    use_rank_rotary = False

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_relation_groups: int,
        rank: int = 16,
        dropout: float = 0.0,
        eps: float = 1e-5,
        gamma_max: float = 1.2,
        theta_gamma_init: float = 0.5,
        no_global: bool = False,
        normalize_u: bool = True,
        length_normalize: bool = True,
        length_reference: float = 1.0,
        rope_base: float = 10000.0,
        rope_scale: float = 1.0,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if not 1 <= num_relation_groups <= num_heads:
            raise ValueError(
                "num_relation_groups must be between 1 and num_heads, "
                f"got {num_relation_groups} for num_heads={num_heads}"
            )
        if num_heads % num_relation_groups != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by "
                f"num_relation_groups={num_relation_groups}"
            )
        if self.use_rank_rotary and rank % 2 != 0:
            raise ValueError(f"GroupedRRLSSO requires an even rank, got rank={rank}")

        self.dim = dim
        self.num_heads = num_heads
        self.num_relation_groups = num_relation_groups
        self.heads_per_relation_group = num_heads // num_relation_groups
        self.head_dim = dim // num_heads
        self.group_dim = dim // num_relation_groups
        self.rank = rank
        self.eps = eps
        self.gamma_max = gamma_max
        self.no_global = no_global
        self.normalize_u = normalize_u
        self.length_normalize = length_normalize
        if length_reference <= 0:
            raise ValueError(f"length_reference must be positive, got {length_reference}")
        self.length_reference = float(length_reference)
        self.rope_base = rope_base
        self.rope_scale = rope_scale

        self.uc_dim = num_relation_groups * rank + dim
        self.w_uc = nn.Linear(dim, self.uc_dim, bias=bias)
        self.w_o = nn.Linear(dim, dim, bias=bias)
        self.register_buffer(
            "_eye",
            torch.eye(rank).view(1, 1, rank, rank),
            persistent=False,
        )

        self.theta_mu = nn.Parameter(torch.zeros(num_relation_groups))
        self.theta_gamma = nn.Parameter(
            torch.full((num_relation_groups,), float(theta_gamma_init), dtype=torch.float32)
        )

        self.dropout_p = dropout
        self.record_diagnostics = False
        self.prune_rank_keep: int | None = None
        self.last_diagnostics: LSSODiagnostics | None = None

    @property
    def solve_reduction(self) -> float:
        """Factor by which relation systems are reduced versus per-head LSSO."""

        return self.num_heads / self.num_relation_groups

    def _prepare_relation_basis(
        self,
        U: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        del position_ids
        return U

    def _prune_relation_basis(self, U: torch.Tensor, keep: int) -> torch.Tensor:
        scores = U.float().square().mean(dim=-2)
        indices = scores.topk(k=keep, dim=-1, largest=True, sorted=False).indices
        return U.gather(
            -1,
            indices[:, :, None, :].expand(
                U.shape[0], self.num_relation_groups, U.shape[2], keep
            ),
        )

    def _forward_grouped(
        self,
        x: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(f"expected input dimension {self.dim}, got {D}")

        G = self.num_relation_groups
        gd = self.group_dim
        r = self.rank
        group_mask = None
        if valid_mask is not None:
            if valid_mask.shape != (B, N):
                raise ValueError(
                    f"valid_mask must have shape {(B, N)}, got {tuple(valid_mask.shape)}"
                )
            group_mask = valid_mask[:, None, :, None].to(
                device=x.device,
                dtype=x.dtype,
            )

        UC = self.w_uc(x)
        U, C = UC.split((G * r, D), dim=-1)
        U = U.view(B, N, G, r).transpose(1, 2).contiguous()
        C = C.view(B, N, G, gd).transpose(1, 2).contiguous()

        if self.normalize_u:
            U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
        U = self._prepare_relation_basis(U, position_ids)

        solve_eye = self._eye
        if self.prune_rank_keep is not None and 0 < self.prune_rank_keep < r:
            keep = int(self.prune_rank_keep)
            if group_mask is not None:
                U = U * group_mask
            U = self._prune_relation_basis(U, keep)
            solve_eye = None

        mu = F.softplus(self.theta_mu) + self.eps
        gamma = self.gamma_max * torch.sigmoid(self.theta_gamma)
        if self.no_global:
            gamma = torch.zeros_like(gamma)

        mu = mu.view(1, G, 1, 1)
        gamma = gamma.view(1, G, 1, 1)
        if self.record_diagnostics:
            Y, aux = lsso(
                U,
                C,
                mu,
                gamma,
                eye=solve_eye,
                no_global=self.no_global or self.gamma_max == 0.0,
                return_aux=True,
                length_normalize=self.length_normalize,
                length_reference=self.length_reference,
                valid_mask=valid_mask,
            )
        else:
            Y = lsso(
                U,
                C,
                mu,
                gamma,
                eye=solve_eye,
                no_global=self.no_global or self.gamma_max == 0.0,
                length_normalize=self.length_normalize,
                length_reference=self.length_reference,
                valid_mask=valid_mask,
            )
            aux = None

        if aux is not None and aux.UtU is not None:
            self.last_diagnostics = self._diagnostics(
                aux.UtU,
                aux.local,
                aux.correction,
                aux.mu,
                aux.gamma,
            )
        else:
            self.last_diagnostics = None

        Y = Y.transpose(1, 2).contiguous().view(B, N, D)
        Y = self.w_o(Y)
        if valid_mask is not None:
            Y = Y * valid_mask[:, :, None].to(device=Y.device, dtype=Y.dtype)
        return Y

    def _diagnostics(
        self,
        UtU: torch.Tensor,
        local: torch.Tensor,
        correction: torch.Tensor,
        mu: torch.Tensor,
        gamma: torch.Tensor,
    ) -> LSSODiagnostics:
        with torch.no_grad():
            eigvals = torch.linalg.eigvalsh(UtU.float()).clamp_min(0.0)
            eig_sum = eigvals.sum(dim=-1)
            eig_sq_sum = (eigvals * eigvals).sum(dim=-1).clamp_min(self.eps)
            effective_rank = (eig_sum * eig_sum) / eig_sq_sum

            correction_norm = correction.float().norm(dim=(-2, -1))
            local_norm = local.float().norm(dim=(-2, -1)).clamp_min(self.eps)
            correction_ratio = correction_norm / local_norm
            gamma_over_mu = (gamma / mu).view(-1).detach().float().cpu()

        return LSSODiagnostics(
            gamma_over_mu=gamma_over_mu,
            effective_rank=effective_rank.detach().float().cpu(),
            correction_ratio=correction_ratio.detach().float().cpu(),
        )


class GroupedLSSO(_GroupedLSSOBase):
    """Bidirectional LSSO with one relation field per channel-head group.

    Setting ``num_relation_groups == num_heads`` is exactly the original LSSO
    parameterization.  Smaller values share the low-rank relation field across
    adjacent channel heads and solve all group content channels as multiple
    right-hand sides of one SPD system.
    """

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_grouped(
            x,
            valid_mask=valid_mask,
            position_ids=None,
        )


class GroupedRRLSSO(_GroupedLSSOBase):
    """Rank-rotary grouped-relation LSSO for bidirectional encoders."""

    use_rank_rotary = True

    def _prepare_relation_basis(
        self,
        U: torch.Tensor,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        return apply_rank_rotary(
            U,
            position_ids,
            base=self.rope_base,
            scale=self.rope_scale,
        )

    def _prune_relation_basis(self, U: torch.Tensor, keep: int) -> torch.Tensor:
        if keep % 2 != 0:
            raise ValueError("GroupedRRLSSO rank pruning must keep an even number of channels")
        B, G, N, r = U.shape
        pair_scores = U.float().square().mean(dim=-2).view(B, G, r // 2, 2).sum(dim=-1)
        pair_indices = pair_scores.topk(
            k=keep // 2,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        indices = torch.stack((2 * pair_indices, 2 * pair_indices + 1), dim=-1).flatten(-2)
        return U.gather(-1, indices[:, :, None, :].expand(B, G, N, keep))

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._forward_grouped(
            x,
            valid_mask=valid_mask,
            position_ids=position_ids,
        )
