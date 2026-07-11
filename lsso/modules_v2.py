from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import LSSODiagnostics, lsso


def apply_rank_rotary(
    U: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    base: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Apply a fixed rank-space rotary transform to the LSSO solve basis.

    Args:
        U: relation features, [B, H, N, r].
        position_ids: optional sequence indices, [N] or [B, N]. If omitted,
            uses arange(N). These indices only parameterize the rank-space
            rotation applied to U; this is not an embedding table.
        base: rotary frequency base.
        scale: multiplier applied to positions before forming angles.

    Returns:
        Rotated relation features with the same shape as U.
    """
    if U.dim() != 4:
        raise ValueError(f"U must have shape [B, H, N, r], got {tuple(U.shape)}")
    B, _H, N, r = U.shape
    if r % 2 != 0:
        raise ValueError(f"Rank rotary requires an even rank, got rank={r}")

    half = r // 2
    calc_dtype = torch.float64 if U.dtype == torch.float64 else torch.float32
    inv_freq = base ** (-torch.arange(0, half, device=U.device, dtype=calc_dtype) / half)

    if position_ids is None:
        positions = torch.arange(N, device=U.device, dtype=calc_dtype)
        angles = scale * positions[:, None] * inv_freq[None, :]
        cos = angles.cos().to(dtype=U.dtype).view(1, 1, N, half)
        sin = angles.sin().to(dtype=U.dtype).view(1, 1, N, half)
    else:
        positions = position_ids.to(device=U.device, dtype=calc_dtype)
        if positions.dim() == 1:
            if positions.numel() != N:
                raise ValueError(f"position_ids length {positions.numel()} must match N={N}")
            angles = scale * positions[:, None] * inv_freq[None, :]
            cos = angles.cos().to(dtype=U.dtype).view(1, 1, N, half)
            sin = angles.sin().to(dtype=U.dtype).view(1, 1, N, half)
        elif positions.dim() == 2:
            if positions.shape != (B, N):
                raise ValueError(f"position_ids must have shape [B, N]={B, N}, got {tuple(positions.shape)}")
            angles = scale * positions[:, :, None] * inv_freq[None, None, :]
            cos = angles.cos().to(dtype=U.dtype).view(B, 1, N, half)
            sin = angles.sin().to(dtype=U.dtype).view(B, 1, N, half)
        else:
            raise ValueError("position_ids must be None, [N], or [B, N]")

    even = U[..., 0::2]
    odd = U[..., 1::2]
    out = torch.empty_like(U)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def apply_rank_rope(
    U: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    *,
    base: float = 10000.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Backward-compatible name for :func:`apply_rank_rotary`."""
    return apply_rank_rotary(U, position_ids, base=base, scale=scale)


class RRLSSO(nn.Module):
    """
    Rank-Rotary LSSO.

    The v1 solve is kept intact, but the low-rank solve basis U is transformed
    by a fixed rank-space rotary map before building the global statistics:

        U_tilde_i = R(p_i) U_i
        (mu I + gamma U_tilde U_tilde^T) Y = C

    This yields a relative-index kernel in the rank basis:

        K_ij = u_i^T R(p_j - p_i) u_j

    Despite the historical RoPE-LSSO name, this transform is not a learned or
    absolute position embedding. It only rotates U inside the bidirectional
    LSSO operator.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
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
        if rank % 2 != 0:
            raise ValueError(f"RRLSSO requires an even rank, got rank={rank}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
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

        self.uc_dim = num_heads * rank + dim
        self.w_uc = nn.Linear(dim, self.uc_dim, bias=bias)
        self.w_o = nn.Linear(dim, dim, bias=bias)
        self.register_buffer(
            "_eye",
            torch.eye(rank).view(1, 1, rank, rank),
            persistent=False,
        )

        self.theta_mu = nn.Parameter(torch.zeros(num_heads))
        self.theta_gamma = nn.Parameter(
            torch.full((num_heads,), float(theta_gamma_init), dtype=torch.float32)
        )

        self.dropout_p = dropout
        self.record_diagnostics = False
        self.prune_rank_keep: int | None = None
        self.last_diagnostics: LSSODiagnostics | None = None

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        dh = self.head_dim
        r = self.rank
        head_mask = None
        if valid_mask is not None:
            head_mask = valid_mask[:, None, :, None].to(dtype=x.dtype)

        UC = self.w_uc(x)
        U, C = UC.split((H * r, D), dim=-1)
        U = U.view(B, N, H, r).transpose(1, 2).contiguous()
        C = C.view(B, N, H, dh).transpose(1, 2).contiguous()

        if self.normalize_u:
            U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)

        U = apply_rank_rotary(
            U,
            position_ids,
            base=self.rope_base,
            scale=self.rope_scale,
        )

        if head_mask is not None:
            U = U * head_mask
            C = C * head_mask

        solve_eye = self._eye
        if self.prune_rank_keep is not None and 0 < self.prune_rank_keep < r:
            keep = int(self.prune_rank_keep)
            if keep % 2 != 0:
                raise ValueError("RRLSSO rank pruning must keep an even number of channels")
            pair_scores = U.float().square().mean(dim=-2).view(B, H, r // 2, 2).sum(dim=-1)
            pair_indices = pair_scores.topk(k=keep // 2, dim=-1, largest=True, sorted=False).indices
            indices = torch.stack((2 * pair_indices, 2 * pair_indices + 1), dim=-1).flatten(-2)
            U = U.gather(-1, indices[:, :, None, :].expand(B, H, N, keep))
            solve_eye = None

        mu = F.softplus(self.theta_mu) + self.eps
        gamma = self.gamma_max * torch.sigmoid(self.theta_gamma)
        if self.no_global:
            gamma = torch.zeros_like(gamma)

        mu = mu.view(1, H, 1, 1)
        gamma = gamma.view(1, H, 1, 1)
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
            Y = Y * valid_mask[:, :, None].to(dtype=Y.dtype)
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


RoPELSSO = RRLSSO
