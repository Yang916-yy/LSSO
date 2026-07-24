from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mathdx_backend import (
    try_rank_rotary,
)
from .modules import (
    DEFAULT_ALPHA_INIT,
    DEFAULT_GAIN_INIT,
    LSSODiagnostics,
    _initialize_solve_parameters,
    _fold_fixed_gain_into_output,
    _solve_parameters,
    lsso_gain_alpha,
)


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


class RRLSSO(nn.Module):
    """
    Rank-Rotary LSSO.

    The v1 solve is kept intact, but the low-rank solve basis U is transformed
    by a fixed rank-space rotary map before building the global statistics:

        U_tilde_i = R(p_i) U_i
        (mu I + gamma U_tilde U_tilde^T) Y = C

    This yields a relative-index kernel in the rank basis:

        K_ij = u_i^T R(p_j - p_i) u_j

    Rank Rotary acts only on the solve basis ``U`` inside the bidirectional
    operator; it is independent of token position embeddings.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        rank: int = 16,
        dropout: float = 0.0,
        eps: float = 1e-5,
        gain_init: float = DEFAULT_GAIN_INIT,
        alpha_init: float = DEFAULT_ALPHA_INIT,
        solve_parameterization: str = "gain_alpha",
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
        if rank % 2 != 0:
            raise ValueError(f"RRLSSO requires an even rank, got rank={rank}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
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

        self.uc_dim = num_heads * rank + dim
        self.w_uc = nn.Linear(dim, self.uc_dim, bias=bias)
        self.w_o = nn.Linear(dim, dim, bias=bias)
        self.register_buffer(
            "_eye",
            torch.eye(rank).view(1, 1, rank, rank),
            persistent=False,
        )
        self.register_buffer("_rotary_cos_cache", torch.empty(0), persistent=False)
        self.register_buffer("_rotary_sin_cache", torch.empty(0), persistent=False)

        _initialize_solve_parameters(
            self,
            num_heads,
            solve_parameterization=solve_parameterization,
            gain_init=gain_init,
            alpha_init=alpha_init,
        )
        _fold_fixed_gain_into_output(
            self,
            groups=num_heads,
            group_width=self.head_dim,
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
            groups=self.num_heads,
            group_width=self.head_dim,
            force=force,
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        dh = self.head_dim
        r = self.rank
        UC = self.w_uc(x)
        U, C = UC.split((H * r, D), dim=-1)
        # The native Rank Rotary kernel consumes this token-major strided view
        # and writes the rotated basis directly in head-major order. Avoid an
        # otherwise redundant full U layout copy before rotation.
        U = U.view(B, N, H, r).transpose(1, 2)
        C = C.view(B, N, H, dh).transpose(1, 2)

        pruning_active = self.prune_rank_keep is not None and 0 < self.prune_rank_keep < r
        token_rms = self.normalize_u and self.basis_normalization == "token_rms"
        trace_basis = self.normalize_u and self.basis_normalization == "trace"
        if position_ids is None:
            expected_shape = (1, 1, N, r // 2)
            cache_valid = (
                self._rotary_cos_cache.shape == expected_shape
                and self._rotary_cos_cache.device == U.device
                and self._rotary_cos_cache.dtype == U.dtype
                and not (
                    torch.is_grad_enabled()
                    and torch.is_inference(self._rotary_cos_cache)
                )
            )
            if not cache_valid:
                calc_dtype = torch.float64 if U.dtype == torch.float64 else torch.float32
                inv_freq = self.rotary_base ** (
                    -torch.arange(0, r // 2, device=U.device, dtype=calc_dtype)
                    / (r // 2)
                )
                positions = torch.arange(N, device=U.device, dtype=calc_dtype)
                angles = self.rotary_scale * positions[:, None] * inv_freq[None, :]
                self._rotary_cos_cache = angles.cos().to(U.dtype).view(expected_shape)
                self._rotary_sin_cache = angles.sin().to(U.dtype).view(expected_shape)
            cos = self._rotary_cos_cache
            sin = self._rotary_sin_cache
            # Token-RMS is a PyTorch-only ablation. The maintained native
            # preprocessing path contains only the orthogonal rank rotation.
            if token_rms:
                U = U * torch.rsqrt(
                    torch.mean(U * U, dim=-1, keepdim=True) + self.eps
                )
            rotated = try_rank_rotary(U, cos, sin)
            if rotated is None:
                even = U[..., 0::2]
                odd = U[..., 1::2]
                rotated = torch.empty_like(U)
                rotated[..., 0::2] = even * cos - odd * sin
                rotated[..., 1::2] = even * sin + odd * cos
            U = rotated
        else:
            if token_rms:
                U = U * torch.rsqrt(
                    torch.mean(U * U, dim=-1, keepdim=True) + self.eps
                )
            U = apply_rank_rotary(
                U,
                position_ids,
                base=self.rotary_base,
                scale=self.rotary_scale,
            )

        solve_eye = self._eye
        if pruning_active:
            keep = int(self.prune_rank_keep)
            if valid_mask is not None:
                head_mask = valid_mask[:, None, :, None].to(
                    device=x.device,
                    dtype=x.dtype,
                )
                U = U * head_mask
            if keep % 2 != 0:
                raise ValueError("RRLSSO rank pruning must keep an even number of channels")
            pair_scores = U.float().square().mean(dim=-2).view(B, H, r // 2, 2).sum(dim=-1)
            pair_indices = pair_scores.topk(k=keep // 2, dim=-1, largest=True, sorted=False).indices
            indices = torch.stack((2 * pair_indices, 2 * pair_indices + 1), dim=-1).flatten(-2)
            U = U.gather(-1, indices[:, :, None, :].expand(B, H, N, keep))
            solve_eye = None

        gain, alpha = _solve_parameters(self)

        gain = gain.view(1, H, 1, 1)
        alpha = alpha.view(1, H, 1, 1)
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
                padding_ratio_hint=padding_ratio_hint,
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
                padding_ratio_hint=padding_ratio_hint,
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
        if valid_mask is not None and self.w_o.bias is not None:
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
