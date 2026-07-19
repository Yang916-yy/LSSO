from __future__ import annotations

import torch
import torch.nn as nn

from .modules import (
    DEFAULT_ALPHA_INIT,
    DEFAULT_ALPHA_MAX,
    DEFAULT_GAIN_INIT,
    LSSODiagnostics,
    _initialize_solve_parameters,
    _legacy_solve_state_dict_pre_hook,
    _fold_fixed_gain_into_output,
    _solve_parameters,
    _sync_alpha_max_after_load,
    lsso_gain_alpha,
)
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
        gain_init: float = DEFAULT_GAIN_INIT,
        alpha_init: float = DEFAULT_ALPHA_INIT,
        solve_parameterization: str = "gain_alpha",
        alpha_max: float = DEFAULT_ALPHA_MAX,
        no_global: bool = False,
        normalize_u: bool = True,
        basis_normalization: str = "trace",
        length_normalize: bool = True,
        length_reference: float = 1.0,
        rotary_base: float = 10000.0,
        rotary_scale: float = 1.0,
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
        self.no_global = no_global
        self.normalize_u = normalize_u
        if basis_normalization not in {"trace", "token_rms"}:
            raise ValueError(
                "basis_normalization must be 'trace' or 'token_rms', "
                f"got {basis_normalization!r}"
            )
        self.basis_normalization = basis_normalization
        self.length_normalize = length_normalize
        if length_reference <= 0:
            raise ValueError(f"length_reference must be positive, got {length_reference}")
        self.length_reference = float(length_reference)
        self.rotary_base = rotary_base
        self.rotary_scale = rotary_scale

        self.uc_dim = num_relation_groups * rank + dim
        self.w_uc = nn.Linear(dim, self.uc_dim, bias=bias)
        self.w_o = nn.Linear(dim, dim, bias=bias)
        self.register_buffer(
            "_eye",
            torch.eye(rank).view(1, 1, rank, rank),
            persistent=False,
        )

        _initialize_solve_parameters(
            self,
            num_relation_groups,
            solve_parameterization=solve_parameterization,
            gain_init=gain_init,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
        )
        self.register_load_state_dict_pre_hook(_legacy_solve_state_dict_pre_hook)
        self.register_load_state_dict_post_hook(_sync_alpha_max_after_load)
        _fold_fixed_gain_into_output(
            self,
            groups=num_relation_groups,
            group_width=self.group_dim,
        )

        self.dropout_p = dropout
        self.record_diagnostics = False
        self.prune_rank_keep: int | None = None
        self.last_diagnostics: LSSODiagnostics | None = None

    def effective_gain_alpha(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the positive output gain and relative solve strength."""

        return _solve_parameters(self)

    def fold_fixed_gain_into_output(self, *, force: bool = False) -> None:
        _fold_fixed_gain_into_output(
            self,
            groups=self.num_relation_groups,
            group_width=self.group_dim,
            force=force,
        )

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

        token_rms = self.normalize_u and self.basis_normalization == "token_rms"
        trace_basis = self.normalize_u and self.basis_normalization == "trace"
        if token_rms:
            U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
        U = self._prepare_relation_basis(U, position_ids)

        solve_eye = self._eye
        if self.prune_rank_keep is not None and 0 < self.prune_rank_keep < r:
            keep = int(self.prune_rank_keep)
            if group_mask is not None:
                U = U * group_mask
            U = self._prune_relation_basis(U, keep)
            solve_eye = None

        gain, alpha = _solve_parameters(self)

        gain = gain.view(1, G, 1, 1)
        alpha = alpha.view(1, G, 1, 1)
        if self.record_diagnostics:
            Y, aux = lsso_gain_alpha(
                U,
                C,
                gain,
                alpha,
                eye=solve_eye,
                no_global=self._global_disabled,
                return_aux=True,
                length_normalize=self.length_normalize,
                length_reference=self.length_reference,
                trace_normalize=trace_basis,
                normalization_eps=self.eps,
                valid_mask=valid_mask,
            )
        else:
            Y = lsso_gain_alpha(
                U,
                C,
                gain,
                alpha,
                eye=solve_eye,
                no_global=self._global_disabled,
                length_normalize=self.length_normalize,
                length_reference=self.length_reference,
                trace_normalize=trace_basis,
                normalization_eps=self.eps,
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
            alpha = (gamma / mu).view(-1).detach().float().cpu()

        return LSSODiagnostics(
            alpha=alpha,
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
            base=self.rotary_base,
            scale=self.rotary_scale,
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
