from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mathdx_backend import (
    solve_spd_autograd,
    try_masked_stats_solve_spd,
    try_prepare_basis,
    try_stats_solve_spd,
)


@dataclass
class LSSODiagnostics:
    gamma_over_mu: torch.Tensor
    effective_rank: torch.Tensor
    correction_ratio: torch.Tensor


@dataclass
class LSSOAux:
    UtU: torch.Tensor | None
    local: torch.Tensor
    correction: torch.Tensor
    mu: torch.Tensor
    gamma: torch.Tensor


@dataclass
class SolveStateCache:
    """Compressed low-rank solve state.

    The cache stores only aggregate low-rank statistics:

        S = sum U_i^T U_i
        P = sum U_i^T C_i

    For RRLSSO, apply the rank rotary transform to ``U`` before updating the state.
    """

    S: torch.Tensor
    P: torch.Tensor
    length: int | torch.Tensor = 0


def _solve_dtype(*tensors: torch.Tensor) -> torch.dtype:
    return torch.float64 if any(t.dtype == torch.float64 for t in tensors) else torch.float32


def length_normalize_basis(
    U: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    reference_length: float = 1.0,
) -> torch.Tensor:
    """Scale a bidirectional relation basis by ``sqrt(L_ref / L_eff)``.

    With this scaling, ``U.T @ U`` and the complete Woodbury correction use
    mean rather than sum statistics, so duplicating/padding a sequence does not
    change the global correction strength. ``reference_length`` is a fixed
    compatibility scale: setting it to the training length preserves the old
    operator exactly at that length while still removing length drift.
    """

    if U.dim() != 4:
        raise ValueError(f"U must have shape [B, H, N, r], got {tuple(U.shape)}")
    if reference_length <= 0:
        raise ValueError(f"reference_length must be positive, got {reference_length}")

    B, _H, N, _r = U.shape
    calc_dtype = torch.float64 if U.dtype == torch.float64 else torch.float32
    if valid_mask is None:
        lengths = torch.full(
            (B,),
            float(N),
            device=U.device,
            dtype=calc_dtype,
        )
    else:
        if valid_mask.shape != (B, N):
            raise ValueError(
                f"valid_mask must have shape {(B, N)}, got {tuple(valid_mask.shape)}"
            )
        lengths = valid_mask.to(device=U.device, dtype=calc_dtype).sum(dim=-1)

    scale = torch.sqrt(
        torch.as_tensor(reference_length, device=U.device, dtype=calc_dtype)
        / lengths.clamp_min(1.0)
    )
    return U * scale.to(dtype=U.dtype).view(B, 1, 1, 1)


def _bmm_accumulate(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Batched matmul for compact LSSO statistics.

    CUDA BF16/FP16 statistics are accumulated directly into FP32 output.  This
    avoids first writing reduced-precision statistics and then launching
    conversion kernels before the SPD solve. Float64 is preserved for
    gradcheck and high-precision reference use.
    """
    if dtype == torch.float64:
        return torch.bmm(left.to(torch.float64), right.to(torch.float64))
    if (
        dtype == torch.float32
        and left.is_cuda
        and right.is_cuda
        and left.dtype == right.dtype
        and left.dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
    ):
        return torch.bmm(left, right, out_dtype=torch.float32)
    if left.dtype == right.dtype:
        return torch.bmm(left, right)
    return torch.bmm(left.to(torch.float32), right.to(torch.float32))


def _solve_no_check(G: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve batched small systems without synchronization-heavy error checks."""
    with torch.amp.autocast(device_type=G.device.type, enabled=False):
        return solve_spd_autograd(G, rhs)


def make_solve_state(
    U: torch.Tensor,
    C: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> SolveStateCache:
    """Build an S/P solve-state cache from relation features and targets.

    Args:
        U: low-rank relation features, [B, H, N, r]. If using RRLSSO,
            pass the already rank-rotated U.
        C: target states, [B, H, N, dh].
        valid_mask: optional valid-token mask, [B, N].
    """
    if U.dim() != 4 or C.dim() != 4:
        raise ValueError("U and C must have shapes [B, H, N, r] and [B, H, N, dh]")
    if U.shape[:3] != C.shape[:3]:
        raise ValueError(f"U and C leading dimensions must match, got {tuple(U.shape)} and {tuple(C.shape)}")
    if valid_mask is not None:
        mask = valid_mask[:, None, :, None].to(device=U.device, dtype=U.dtype)
        U = U * mask
        C = C * mask

    B, H, N, r = U.shape
    dh = C.shape[-1]
    calc_dtype = _solve_dtype(U, C)
    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    Ut = U_bh.transpose(1, 2)
    S = _bmm_accumulate(Ut, U_bh, dtype=calc_dtype).view(B, H, r, r)
    P = _bmm_accumulate(Ut, C_bh, dtype=calc_dtype).view(B, H, r, dh)
    cache_length: int | torch.Tensor
    if valid_mask is None:
        cache_length = N
    else:
        cache_length = valid_mask.to(device=U.device, dtype=torch.long).sum(dim=-1)
    return SolveStateCache(S=S, P=P, length=cache_length)


def update_solve_state(
    cache: SolveStateCache | None,
    U: torch.Tensor,
    C: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> SolveStateCache:
    """Append tokens to an S/P solve-state cache."""
    new_state = make_solve_state(U, C, valid_mask=valid_mask)
    if cache is None:
        return new_state
    if cache.S.shape != new_state.S.shape:
        raise ValueError(f"cache.S shape {tuple(cache.S.shape)} does not match new S shape {tuple(new_state.S.shape)}")
    if cache.P.shape != new_state.P.shape:
        raise ValueError(f"cache.P shape {tuple(cache.P.shape)} does not match new P shape {tuple(new_state.P.shape)}")
    return SolveStateCache(S=cache.S + new_state.S, P=cache.P + new_state.P, length=cache.length + new_state.length)


def read_solve_state(
    U: torch.Tensor,
    C: torch.Tensor,
    cache: SolveStateCache,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    length_normalize: bool = True,
    length_reference: float = 1.0,
) -> torch.Tensor:
    """Read solved token states from an S/P solve-state cache.

    This is a low-level aggregate-statistics utility retained for diagnostics
    and diagnostics.
    """
    B, H, N, r = U.shape
    dh = C.shape[-1]
    if cache.S.shape != (B, H, r, r):
        raise ValueError(f"cache.S shape {tuple(cache.S.shape)} does not match {(B, H, r, r)}")
    if cache.P.shape != (B, H, r, dh):
        raise ValueError(f"cache.P shape {tuple(cache.P.shape)} does not match {(B, H, r, dh)}")
    if mu.dim() == 1:
        mu = mu.view(1, H, 1, 1)
    if gamma.dim() == 1:
        gamma = gamma.view(1, H, 1, 1)

    solve_dtype = _solve_dtype(U, C, mu, gamma)
    output_dtype = C.dtype if U.dtype == C.dtype else torch.promote_types(U.dtype, C.dtype)
    S = cache.S.to(solve_dtype)
    P = cache.P.to(solve_dtype)
    if length_normalize:
        if length_reference <= 0:
            raise ValueError(f"length_reference must be positive, got {length_reference}")
        if isinstance(cache.length, torch.Tensor):
            if cache.length.shape != (B,):
                raise ValueError(
                    f"tensor cache.length must have shape {(B,)}, got {tuple(cache.length.shape)}"
                )
            length_scale = (
                float(length_reference)
                / cache.length.to(device=U.device, dtype=solve_dtype).clamp_min(1)
            ).view(B, 1, 1, 1)
        else:
            length_scale = float(length_reference) / max(1, cache.length)
        S = S * length_scale
        P = P * length_scale
    mu_calc = mu.to(solve_dtype)
    gamma_calc = gamma.to(solve_dtype)
    inv_mu = mu_calc.reciprocal()
    alpha = gamma_calc * inv_mu
    gamma_over_mu2 = alpha * inv_mu

    eye = torch.eye(r, device=U.device, dtype=solve_dtype).view(1, 1, r, r)
    G = eye + alpha * S
    K = _solve_no_check(
        G.reshape(B * H, r, r),
        P.reshape(B * H, r, dh),
    ).to(output_dtype).view(B, H, r, dh)
    correction = torch.bmm(U.flatten(0, 1).to(output_dtype), K.flatten(0, 1)).view(B, H, N, dh)
    Y = C.to(output_dtype).mul(inv_mu.to(output_dtype)) - gamma_over_mu2.to(output_dtype) * correction
    return Y.to(C.dtype)


def _lsso_woodbury_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    eye: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, N, r = U.shape
    dh = C.shape[-1]
    calc_dtype = torch.promote_types(U.dtype, C.dtype)
    solve_dtype = torch.float64 if calc_dtype == torch.float64 else torch.float32
    output_dtype = C.dtype if U.dtype == C.dtype else calc_dtype
    U_calc = U.to(calc_dtype)
    C_calc = C.to(calc_dtype)
    mu_calc = mu.to(calc_dtype)
    gamma_calc = gamma.to(calc_dtype)
    inv_mu = mu_calc.reciprocal()
    gamma_over_mu = gamma_calc * inv_mu
    gamma_over_mu2 = gamma_over_mu * inv_mu

    U_bh = U_calc.flatten(0, 1)
    C_bh = C_calc.flatten(0, 1)
    alpha_bh = gamma_over_mu.expand(B, H, 1, 1).reshape(B * H).float()
    K = try_stats_solve_spd(U_bh, C_bh, alpha_bh)
    if K is None:
        Ut_bh = U_bh.transpose(1, 2)
        UtU = _bmm_accumulate(Ut_bh, U_bh, dtype=solve_dtype).view(B, H, r, r)
        UtC = _bmm_accumulate(Ut_bh, C_bh, dtype=solve_dtype).view(B, H, r, dh)
        if eye is None:
            eye = torch.eye(r, device=U.device, dtype=calc_dtype).view(1, 1, r, r)
        G = eye.to(solve_dtype) + gamma_over_mu.to(solve_dtype) * UtU.to(solve_dtype)
        K = _solve_no_check(
            G.view(B * H, r, r),
            UtC.to(solve_dtype).view(B * H, r, dh),
        ).to(calc_dtype)
    # Scale the compact rank-space solution, then let the GEMM epilogue write
    # directly into the output-sized tensor.  This avoids materializing UK and
    # a separate local C / mu tensor at the same time.
    alpha_bh = gamma_over_mu.expand(B, H, 1, 1).reshape(B * H, 1, 1)
    if torch.is_grad_enabled():
        K_readout = K * alpha_bh.to(K.dtype)
    else:
        K.mul_(alpha_bh.to(K.dtype))
        K_readout = K
    Y = torch.baddbmm(C_bh, U_bh, K_readout, beta=1.0, alpha=-1.0).view(B, H, N, dh)
    Y.mul_(inv_mu)
    return Y.to(output_dtype)


def _exclusive_prefix(x: torch.Tensor) -> torch.Tensor:
    zeros = x.new_zeros(*x.shape[:2], 1, *x.shape[3:])
    return torch.cat((zeros, x[:, :, :-1]), dim=2)


def _lsso_prefix_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    eye: torch.Tensor | None = None,
    *,
    exclusive: bool = False,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LSSOAux]:
    B, H, N, r = U.shape
    dh = C.shape[-1]
    output_dtype = C.dtype if U.dtype == C.dtype else torch.promote_types(U.dtype, C.dtype)
    calc_dtype = torch.float64 if U.dtype == torch.float64 or C.dtype == torch.float64 else torch.float32

    U_calc = U.to(calc_dtype)
    C_calc = C.to(calc_dtype)
    mu_calc = mu.to(calc_dtype)
    gamma_calc = gamma.to(calc_dtype)
    inv_mu = mu_calc.reciprocal()
    gamma_over_mu = gamma_calc * inv_mu
    gamma_over_mu2 = gamma_over_mu * inv_mu

    S_token = U_calc.unsqueeze(-1) * U_calc.unsqueeze(-2)
    P_token = U_calc.unsqueeze(-1) * C_calc.unsqueeze(-2)
    S = torch.cumsum(S_token, dim=2)
    P = torch.cumsum(P_token, dim=2)
    if exclusive:
        S = _exclusive_prefix(S)
        P = _exclusive_prefix(P)

    if eye is None:
        eye = torch.eye(r, device=U.device, dtype=calc_dtype).view(1, 1, 1, r, r)
    elif eye.dim() == 4:
        eye = eye.unsqueeze(2)

    solve_dtype = torch.float64 if calc_dtype == torch.float64 else torch.float32
    G = eye.to(solve_dtype) + gamma_over_mu.unsqueeze(2).to(solve_dtype) * S.to(solve_dtype)
    K = _solve_no_check(
        G.reshape(B * H * N, r, r),
        P.to(solve_dtype).reshape(B * H * N, r, dh),
    ).to(calc_dtype).view(B, H, N, r, dh)

    UK = torch.einsum("bhnr,bhnrd->bhnd", U_calc, K)
    local = inv_mu * C_calc
    correction = gamma_over_mu2 * UK
    Y = local - correction

    if return_aux:
        return (
            Y.to(output_dtype),
            LSSOAux(
                UtU=S.detach() if not torch.is_grad_enabled() else S,
                local=local.to(output_dtype),
                correction=correction.to(output_dtype),
                mu=mu,
                gamma=gamma,
            ),
        )
    return Y.to(output_dtype)


def _lsso_prefix_chunked_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    eye: torch.Tensor | None = None,
    *,
    exclusive: bool = False,
    chunk_size: int = 128,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LSSOAux]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    B, H, N, r = U.shape
    dh = C.shape[-1]
    output_dtype = C.dtype if U.dtype == C.dtype else torch.promote_types(U.dtype, C.dtype)
    calc_dtype = torch.float64 if U.dtype == torch.float64 or C.dtype == torch.float64 else torch.float32

    U_calc = U.to(calc_dtype)
    C_calc = C.to(calc_dtype)
    mu_calc = mu.to(calc_dtype)
    gamma_calc = gamma.to(calc_dtype)
    inv_mu = mu_calc.reciprocal()
    gamma_over_mu = gamma_calc * inv_mu
    gamma_over_mu2 = gamma_over_mu * inv_mu

    if eye is None:
        eye = torch.eye(r, device=U.device, dtype=calc_dtype).view(1, 1, 1, r, r)
    elif eye.dim() == 4:
        eye = eye.unsqueeze(2)

    solve_dtype = torch.float64 if calc_dtype == torch.float64 else torch.float32
    S_state = torch.zeros(B, H, r, r, device=U.device, dtype=calc_dtype)
    P_state = torch.zeros(B, H, r, dh, device=U.device, dtype=calc_dtype)
    Y_chunks: list[torch.Tensor] = []
    UtU_chunks: list[torch.Tensor] = []
    local_chunks: list[torch.Tensor] = []
    correction_chunks: list[torch.Tensor] = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        U_block = U_calc[:, :, start:end]
        C_block = C_calc[:, :, start:end]

        S_token = U_block.unsqueeze(-1) * U_block.unsqueeze(-2)
        P_token = U_block.unsqueeze(-1) * C_block.unsqueeze(-2)
        S_prefix = torch.cumsum(S_token, dim=2)
        P_prefix = torch.cumsum(P_token, dim=2)
        if exclusive:
            S_block = S_state.unsqueeze(2) + _exclusive_prefix(S_prefix)
            P_block = P_state.unsqueeze(2) + _exclusive_prefix(P_prefix)
        else:
            S_block = S_state.unsqueeze(2) + S_prefix
            P_block = P_state.unsqueeze(2) + P_prefix

        G = eye.to(solve_dtype) + gamma_over_mu.unsqueeze(2).to(solve_dtype) * S_block.to(solve_dtype)
        block_len = end - start
        K = _solve_no_check(
            G.reshape(B * H * block_len, r, r),
            P_block.to(solve_dtype).reshape(B * H * block_len, r, dh),
        ).to(calc_dtype).view(B, H, block_len, r, dh)

        UK = torch.einsum("bhnr,bhnrd->bhnd", U_block, K)
        local = inv_mu * C_block
        correction = gamma_over_mu2 * UK
        Y_chunks.append(local - correction)

        if return_aux:
            UtU_chunks.append(S_block)
            local_chunks.append(local)
            correction_chunks.append(correction)

        S_state = S_state + S_token.sum(dim=2)
        P_state = P_state + P_token.sum(dim=2)

    Y = torch.cat(Y_chunks, dim=2)
    if return_aux:
        return (
            Y.to(output_dtype),
            LSSOAux(
                UtU=torch.cat(UtU_chunks, dim=2),
                local=torch.cat(local_chunks, dim=2).to(output_dtype),
                correction=torch.cat(correction_chunks, dim=2).to(output_dtype),
                mu=mu,
                gamma=gamma,
            ),
        )
    return Y.to(output_dtype)


class _LSSOAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U: torch.Tensor,
        C: torch.Tensor,
        mu: torch.Tensor,
        gamma: torch.Tensor,
        eye: torch.Tensor | None,
    ) -> torch.Tensor:
        Y = _lsso_woodbury_forward(U, C, mu, gamma, eye)
        ctx.save_for_backward(U, Y, mu, gamma)
        return Y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        U, Y, mu, gamma = ctx.saved_tensors
        B, H, N, r = U.shape
        dh = Y.shape[-1]

        grad = grad_output.contiguous()
        P = _lsso_woodbury_forward(U, grad, mu, gamma)

        calc_dtype = torch.float64 if U.dtype == torch.float64 or Y.dtype == torch.float64 else torch.float32
        matmul_dtype = U.dtype if U.dtype in (torch.float16, torch.bfloat16) else calc_dtype
        U_m = U.to(matmul_dtype).flatten(0, 1)
        Y_m = Y.to(matmul_dtype).flatten(0, 1)
        P_m = P.to(matmul_dtype).flatten(0, 1)

        YtU = torch.bmm(Y_m.transpose(1, 2), U_m)
        PtU = torch.bmm(P_m.transpose(1, 2), U_m)
        grad_U_m = torch.bmm(P_m, YtU)
        # Accumulate the symmetric second term in the GEMM epilogue.  This
        # avoids materializing another [B * H, N, r] tensor in backward.
        grad_U_m.baddbmm_(Y_m, PtU)
        grad_U_m.mul_(gamma.expand(B, H, 1, 1).to(matmul_dtype).reshape(B * H, 1, 1))
        grad_U = grad_U_m.neg_().view(B, H, N, r).to(U.dtype)

        grad_C = P.to(grad_output.dtype)

        if calc_dtype == torch.float64:
            grad_mu_bh = -(P.to(calc_dtype) * Y.to(calc_dtype)).sum(dim=(2, 3)).view(B, H, 1, 1)
        else:
            grad_mu_bh = -(P * Y).sum(dim=(2, 3), dtype=torch.float32).view(B, H, 1, 1)
        if YtU.dim() == 4:
            grad_gamma_bh = -(PtU.to(calc_dtype) * YtU.to(calc_dtype)).sum(dim=(2, 3)).view(B, H, 1, 1)
        else:
            grad_gamma_bh = -(PtU.to(calc_dtype) * YtU.to(calc_dtype)).sum(dim=(1, 2)).view(B, H, 1, 1)

        if mu.shape[0] == 1:
            grad_mu = grad_mu_bh.sum(dim=0, keepdim=True)
        else:
            grad_mu = grad_mu_bh
        if gamma.shape[0] == 1:
            grad_gamma = grad_gamma_bh.sum(dim=0, keepdim=True)
        else:
            grad_gamma = grad_gamma_bh

        return grad_U, grad_C, grad_mu.to(mu.dtype), grad_gamma.to(gamma.dtype), None


def _masked_length_scale(
    valid_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    length_normalize: bool,
    length_reference: float,
) -> torch.Tensor:
    if length_reference <= 0:
        raise ValueError(f"length_reference must be positive, got {length_reference}")
    if not length_normalize:
        return torch.ones(valid_mask.shape[0], device=valid_mask.device, dtype=dtype)
    lengths = valid_mask.sum(dim=-1, dtype=dtype).clamp_min_(1.0)
    return torch.sqrt(
        torch.as_tensor(length_reference, device=valid_mask.device, dtype=dtype) / lengths
    )


def _lsso_masked_woodbury_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    valid_mask: torch.Tensor,
    length_scale: torch.Tensor,
    eye: torch.Tensor | None = None,
) -> torch.Tensor:
    """Masked solve whose native path predicates padded U/C global loads."""
    B, H, N, r = U.shape
    dh = C.shape[-1]
    inv_mu = mu.reciprocal()
    alpha = gamma * inv_mu
    alpha_bh = alpha.expand(B, H, 1, 1).reshape(B * H).float()
    K = try_masked_stats_solve_spd(
        U, C, valid_mask, length_scale.float(), alpha_bh
    )
    if K is None:
        mask_u = valid_mask[:, None, :, None].to(dtype=U.dtype)
        mask_c = mask_u.to(dtype=C.dtype)
        U_scaled = U * mask_u * length_scale.to(U.dtype).view(B, 1, 1, 1)
        C_masked = C * mask_c
        return _lsso_woodbury_forward(U_scaled, C_masked, mu, gamma, eye)

    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    # The compact solve used s*U. The readout therefore needs alpha*s*U*K.
    readout_scale = (
        alpha.expand(B, H, 1, 1).reshape(B * H, 1, 1)
        * length_scale[:, None].expand(B, H).reshape(B * H, 1, 1)
    )
    K = K.to(U.dtype).mul_(readout_scale.to(U.dtype))
    Y = torch.baddbmm(C_bh, U_bh, K, beta=1.0, alpha=-1.0)
    Y = Y.view(B, H, N, dh).mul_(inv_mu)
    return Y.mul_(valid_mask[:, None, :, None].to(Y.dtype))


class _MaskedLSSOAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, C, mu, gamma, eye, valid_mask, length_scale):
        Y = _lsso_masked_woodbury_forward(
            U, C, mu, gamma, valid_mask, length_scale, eye
        )
        ctx.save_for_backward(U, Y, mu, gamma, valid_mask, length_scale)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        U, Y, mu, gamma, valid_mask, length_scale = ctx.saved_tensors
        B, H, N, r = U.shape
        mask = valid_mask[:, None, :, None]
        grad = grad_output.mul(mask.to(grad_output.dtype)).contiguous()
        P = _lsso_masked_woodbury_forward(
            U, grad, mu, gamma, valid_mask, length_scale
        )

        calc_dtype = torch.float64 if U.dtype == torch.float64 else torch.float32
        matmul_dtype = U.dtype if U.dtype in (torch.float16, torch.bfloat16) else calc_dtype
        U_m = U.to(matmul_dtype).flatten(0, 1)
        Y_m = Y.to(matmul_dtype).flatten(0, 1)
        P_m = P.to(matmul_dtype).flatten(0, 1)
        YtU = torch.bmm(Y_m.transpose(1, 2), U_m)
        PtU = torch.bmm(P_m.transpose(1, 2), U_m)
        grad_U_m = torch.bmm(P_m, YtU)
        grad_U_m.baddbmm_(Y_m, PtU)
        scale2 = length_scale.square()[:, None].expand(B, H).reshape(B * H, 1, 1)
        coeff = gamma.expand(B, H, 1, 1).reshape(B * H, 1, 1) * scale2
        grad_U = grad_U_m.mul_(coeff.to(matmul_dtype)).neg_().view(B, H, N, r)
        grad_U.mul_(mask.to(grad_U.dtype))
        grad_U = grad_U.to(U.dtype)
        grad_C = P.to(grad_output.dtype)

        grad_mu_bh = -(P * Y).sum(dim=(2, 3), dtype=calc_dtype).view(B, H, 1, 1)
        grad_gamma_bh = -(PtU.to(calc_dtype) * YtU.to(calc_dtype)).sum(dim=(1, 2))
        grad_gamma_bh = grad_gamma_bh.view(B, H)
        grad_gamma_bh.mul_(scale2.view(B, H).to(calc_dtype))
        grad_gamma_bh = grad_gamma_bh.view(B, H, 1, 1)
        grad_mu = grad_mu_bh.sum(dim=0, keepdim=True) if mu.shape[0] == 1 else grad_mu_bh
        grad_gamma = (
            grad_gamma_bh.sum(dim=0, keepdim=True)
            if gamma.shape[0] == 1
            else grad_gamma_bh
        )
        return (
            grad_U,
            grad_C,
            grad_mu.to(mu.dtype),
            grad_gamma.to(gamma.dtype),
            None,
            None,
            None,
        )


def lsso(
    U: torch.Tensor,
    C: torch.Tensor,
    mu: torch.Tensor,
    gamma: torch.Tensor,
    *,
    eye: torch.Tensor | None = None,
    no_global: bool = False,
    return_aux: bool = False,
    length_normalize: bool = True,
    length_reference: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, LSSOAux]:
    """
    Functional LSSO core, analogous to an attention kernel.

    Args:
        U: low-rank relation features, [B, H, N, r].
        C: target state, [B, H, N, dh].
        mu: positive shift, broadcastable to [B, H, 1, 1] or [H].
        gamma: global strength, broadcastable to [B, H, 1, 1] or [H].
        eye: optional identity buffer shaped [1, 1, r, r].
        no_global: if true, returns only mu^-1 C.
        return_aux: if true, also returns tensors used for diagnostics.
        length_normalize: use effective-length mean statistics instead of
            sequence sums.
        length_reference: fixed positive compatibility scale. A value matching
            an old checkpoint's training length preserves its old strength at
            that length.
        valid_mask: optional [B, N] mask used for effective lengths. Masked U/C
            entries are zeroed here as a safety measure.

    Returns:
        Y: solved token states, [B, H, N, dh].
    """
    B, H, N, r = U.shape
    dh = C.shape[-1]

    if mu.dim() == 1:
        mu = mu.view(1, H, 1, 1)
    if gamma.dim() == 1:
        gamma = gamma.view(1, H, 1, 1)

    if valid_mask is not None:
        if valid_mask.shape != (B, N):
            raise ValueError(
                f"valid_mask must have shape {(B, N)}, got {tuple(valid_mask.shape)}"
            )
        valid_mask = valid_mask.to(device=U.device, dtype=torch.bool).contiguous()
        # The normal masked path lets the CUDA statistics kernel inspect the
        # mask before loading U/C. Keep the materialized fallback for aux/no-op
        # diagnostic modes, whose outputs explicitly expose those tensors.
        if not no_global and not return_aux:
            scale_dtype = torch.float64 if U.dtype == torch.float64 else torch.float32
            length_scale = _masked_length_scale(
                valid_mask,
                dtype=scale_dtype,
                length_normalize=length_normalize,
                length_reference=length_reference,
            )
            if torch.is_grad_enabled() and (
                U.requires_grad or C.requires_grad or mu.requires_grad or gamma.requires_grad
            ):
                return _MaskedLSSOAutograd.apply(
                    U, C, mu, gamma, eye, valid_mask, length_scale
                )
            return _lsso_masked_woodbury_forward(
                U, C, mu, gamma, valid_mask, length_scale, eye
            )
        solve_mask = valid_mask[:, None, :, None].to(device=U.device, dtype=U.dtype)
        U = U * solve_mask
        C = C * solve_mask.to(dtype=C.dtype)

    if length_normalize:
        U = length_normalize_basis(
            U,
            valid_mask,
            reference_length=length_reference,
        )

    inv_mu = mu.reciprocal()
    gamma_over_mu = gamma * inv_mu
    gamma_over_mu2 = gamma_over_mu * inv_mu

    if (
        torch.is_grad_enabled()
        and not no_global
        and not return_aux
        and (U.requires_grad or C.requires_grad or mu.requires_grad or gamma.requires_grad)
    ):
        return _LSSOAutograd.apply(U, C, mu, gamma, eye)

    local = None
    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    Ut_bh = U_bh.transpose(1, 2)

    if no_global:
        Y = C.mul(inv_mu)
        if return_aux:
            local = Y
            correction = torch.zeros_like(Y)
            UtU = torch.bmm(Ut_bh, U_bh).view(B, H, r, r)
        else:
            UtU = None
    else:
        solve_dtype = _solve_dtype(U, C, mu, gamma)
        K = None
        if not return_aux:
            alpha_bh = gamma_over_mu.expand(B, H, 1, 1).reshape(B * H).float()
            K = try_stats_solve_spd(U_bh, C_bh, alpha_bh)
        if K is None:
            UtU = _bmm_accumulate(Ut_bh, U_bh, dtype=solve_dtype).view(B, H, r, r)
            UtC = _bmm_accumulate(Ut_bh, C_bh, dtype=solve_dtype).view(B, H, r, dh)

            if eye is None:
                eye = torch.eye(r, device=U.device, dtype=solve_dtype).view(1, 1, r, r)
            G = eye.to(solve_dtype) + gamma_over_mu.to(solve_dtype) * UtU
            K = _solve_no_check(
                G.view(B * H, r, r),
                UtC.to(solve_dtype).view(B * H, r, dh),
            ).to(U.dtype)
        else:
            UtU = None
            K = K.to(U.dtype)

        if return_aux:
            UK = torch.bmm(U_bh, K).view(B, H, N, dh)
            local = inv_mu * C
            correction = gamma_over_mu2 * UK
            Y = local - correction
        else:
            # Recompute the compact scaling rather than retaining the
            # output-sized UK tensor through two pointwise kernels.
            alpha_bh = gamma_over_mu.expand(B, H, 1, 1).reshape(B * H, 1, 1)
            if torch.is_grad_enabled():
                K_readout = K * alpha_bh.to(K.dtype)
            else:
                K.mul_(alpha_bh.to(K.dtype))
                K_readout = K
            Y = torch.baddbmm(C_bh, U_bh, K_readout, beta=1.0, alpha=-1.0)
            Y = Y.view(B, H, N, dh)
            Y.mul_(inv_mu)

    if return_aux:
        assert local is not None
        return Y, LSSOAux(UtU=UtU, local=local, correction=correction, mu=mu, gamma=gamma)
    return Y


class LSSO(nn.Module):
    """
    LSSO v1: Learnable Sylvester Solve Operator.

    Per head:
        (mu I + gamma U U^T) Y = C

    Woodbury:
        Y = mu^-1 C
            - gamma / mu^2 * U @ solve(I + gamma/mu * U^T U, U^T C)
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
        bias: bool = False,
    ) -> None:
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

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

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        dh = self.head_dim
        r = self.rank
        head_mask = None
        if valid_mask is not None:
            head_mask = valid_mask[:, None, :, None].to(
                device=x.device,
                dtype=x.dtype,
            )

        UC = self.w_uc(x)
        U, C = UC.split((H * r, D), dim=-1)
        U = U.view(B, N, H, r).transpose(1, 2).contiguous()
        C = C.view(B, N, H, dh).transpose(1, 2).contiguous()

        pruning_active = self.prune_rank_keep is not None and 0 < self.prune_rank_keep < r
        fused_basis = False
        if (
            self.normalize_u
            and self.length_normalize
            and valid_mask is None
            and not pruning_active
        ):
            prepared = try_prepare_basis(
                U,
                eps=self.eps,
                length_scale=(self.length_reference / N) ** 0.5,
            )
            if prepared is not None:
                U = prepared
                fused_basis = True
        if self.normalize_u and not fused_basis:
            U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
        solve_eye = self._eye
        if pruning_active:
            keep = int(self.prune_rank_keep)
            if head_mask is not None:
                # Rank selection must ignore padding.  The normal solve path
                # applies the mask once inside lsso(), so avoid masking twice.
                U = U * head_mask
            scores = U.float().square().mean(dim=-2)
            indices = scores.topk(k=keep, dim=-1, largest=True, sorted=False).indices
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
                length_normalize=self.length_normalize and not fused_basis,
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
                length_normalize=self.length_normalize and not fused_basis,
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
