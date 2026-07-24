from __future__ import annotations

import os
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mathdx_backend import (
    record_mathdx_path,
    solve_spd_autograd,
    try_effective_stats_solve_readout,
    try_dual_backward_statistics_tensorcore,
    try_dual_grad_u_tensorcore,
    try_masked_stats_solve_readout,
    try_masked_trace_stats_solve_readout,
    try_trace_stats_solve_readout,
)
from .types import LSSOAux, LSSODiagnostics, SolveStateCache


DEFAULT_GAIN_INIT = 1.0
_FROZEN_ALPHA_INIT = 1.0
_MASK_REQUIRES_PRIMAL_CACHE: dict[
    int, tuple[weakref.ReferenceType[torch.Tensor], int, int, bool]
] = {}


def _initialize_solve_parameters(
    module: nn.Module,
    count: int,
    *,
    gain_init: float,
) -> None:
    """Create the frozen public gain/log-strength parameterization."""

    if gain_init <= 0:
        raise ValueError(f"gain_init must be positive, got {gain_init}")
    module._global_disabled = bool(module.no_global)
    module.theta_gain = nn.Parameter(
        torch.full(
            (count,),
            float(torch.log(torch.tensor(gain_init, dtype=torch.float64))),
            dtype=torch.float32,
        )
    )
    module.theta_alpha = nn.Parameter(
        torch.full(
            (count,),
            float(torch.log(torch.tensor(_FROZEN_ALPHA_INIT, dtype=torch.float64))),
            dtype=torch.float32,
        )
    )


def _solve_parameters(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gain and an unbounded positive log-parameterized strength."""

    gain = torch.exp(module.theta_gain)
    alpha = torch.exp(module.theta_alpha)
    if module.no_global:
        alpha = torch.zeros_like(alpha)
    return gain, alpha


def _solve_log_parameters(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gain and the canonical log-strength without materializing alpha."""

    gain = torch.exp(module.theta_gain)
    return gain, module.theta_alpha


def _solve_coefficients(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy adapter returning ``mu, gamma`` for historical callers."""

    gain, alpha = _solve_parameters(module)
    mu = gain.reciprocal()
    return mu, alpha * mu


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


def _trace_normalization_factors(
    U: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    eps: float,
    length_normalize: bool,
    length_reference: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-system ``scale**2`` and its energy denominator.

    The target Gram trace is ``rank * length_reference`` when length
    normalization is enabled, and ``rank * effective_length`` otherwise.
    The epsilon term is scaled by the number of valid basis elements so its
    meaning matches the epsilon in an ordinary RMS denominator.
    """

    if U.dim() != 4:
        raise ValueError(f"U must have shape [B, H, N, r], got {tuple(U.shape)}")
    if eps < 0:
        raise ValueError(f"eps must be non-negative, got {eps}")
    if length_reference <= 0:
        raise ValueError(f"length_reference must be positive, got {length_reference}")
    B, _H, N, rank = U.shape
    calc_dtype = torch.float64 if U.dtype == torch.float64 else torch.float32
    if valid_mask is None:
        safe = U
        lengths = torch.full((B,), float(N), device=U.device, dtype=calc_dtype)
    else:
        if valid_mask.shape != (B, N):
            raise ValueError(
                f"valid_mask must have shape {(B, N)}, got {tuple(valid_mask.shape)}"
            )
        active = valid_mask[:, None, :, None].to(device=U.device, dtype=torch.bool)
        safe = torch.where(active, U, torch.zeros((), device=U.device, dtype=U.dtype))
        lengths = valid_mask.to(device=U.device, dtype=calc_dtype).sum(dim=-1).clamp_min(1.0)

    energy = safe.to(calc_dtype).square().sum(dim=(-2, -1), keepdim=True)
    element_count = lengths.view(B, 1, 1, 1) * rank
    denominator = energy + float(eps) * element_count
    if length_normalize:
        target = torch.full_like(denominator, rank * float(length_reference))
    else:
        target = element_count.expand_as(denominator)
    # eps=0 is useful for exact theory tests; keep the all-zero basis finite.
    tiny = torch.finfo(calc_dtype).tiny
    scale_squared = target / denominator.clamp_min(tiny)
    return scale_squared, denominator


def trace_normalize_basis(
    U: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-5,
    length_normalize: bool = True,
    length_reference: float = 1.0,
) -> torch.Tensor:
    """Normalize only the globally non-identifiable relation-basis scale.

    Unlike token-wise RMS normalization, this fixes the Gram trace per
    sample/head while preserving all relative token magnitudes.
    """

    scale_squared, _ = _trace_normalization_factors(
        U,
        valid_mask,
        eps=eps,
        length_normalize=length_normalize,
        length_reference=length_reference,
    )
    if valid_mask is None:
        safe = U
    else:
        active = valid_mask[:, None, :, None].to(device=U.device, dtype=torch.bool)
        safe = torch.where(active, U, torch.zeros((), device=U.device, dtype=U.dtype))
    return safe * scale_squared.sqrt().to(U.dtype)


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
        and left.dtype == right.dtype
        and left.dtype in (torch.float16, torch.bfloat16)
    ):
        if left.is_cuda and right.is_cuda and not torch.is_grad_enabled():
            return torch.bmm(left, right, out_dtype=torch.float32)
        # ``out_dtype`` is intentionally unavailable to differentiable bmm.
        # Cast explicitly so the portable/create_graph path still honors the
        # FP32 compact-statistics contract.
        return torch.bmm(left.float(), right.float())
    if left.dtype == right.dtype:
        return torch.bmm(left, right)
    return torch.bmm(left.to(torch.float32), right.to(torch.float32))


def _solve_no_check(G: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve batched small systems without synchronization-heavy error checks."""
    with torch.amp.autocast(device_type=G.device.type, enabled=False):
        return solve_spd_autograd(G, rhs)


def _balanced_woodbury_system(
    gram: torch.Tensor,
    cross: torch.Tensor,
    alpha: torch.Tensor,
    eye: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale the compact Woodbury system without extreme coefficients.

    For alpha >= 1 this is the reciprocal form
    ``Gram + alpha^-1 I``. For alpha < 1 it is the direct normalized form
    ``alpha Gram + I`` with ``alpha cross`` on the right. Both branches
    represent exactly the same compact correction while keeping every
    coefficient in [0, 1].
    """

    one = torch.ones((), device=alpha.device, dtype=alpha.dtype)
    strong = alpha >= one
    gram_scale = torch.where(strong, one, alpha)
    diagonal_scale = torch.where(strong, alpha.reciprocal(), one)
    return (
        gram_scale * gram + diagonal_scale * eye,
        gram_scale * cross,
    )


# On SM120, a single cuBLAS GEMM remains faster through ordinary ViT-B/4 and
# 8K contexts. Split-N starts paying for its compact-partial reduction around
# 16K tokens; keep the threshold conservative to avoid short-sequence regressions.
_SPLIT_STATS_MIN_SEQUENCE = 16384
_SPLIT_STATS_CHUNK_SIZE = 1024
_SPLIT_BACKWARD_MIN_SEQUENCE = 32768
_SPLIT_BACKWARD_MAX_SYSTEMS = 8
_SPLIT_BACKWARD_CHUNK_SIZE = 1024


def _masked_trace_forward_strategy(
    sequence: int,
    systems: int,
    padding_ratio_hint: float | None,
    *,
    compute_capability: tuple[int, int] | None = None,
) -> tuple[str, int]:
    """Choose the masked Trace statistics schedule without GPU sync.

    TMA exists from Hopper (CC 9.0), but it transfers regular tiles and cannot
    preserve the high-padding skip-read contract. Consequently the masked CTA
    remains preferable when a CPU collator reports mostly padding. Long,
    dense/unknown masks use split-N cuBLAS statistics. SM120 uses its locally
    measured 4K crossover (8K for high padding); other Ampere-or-newer
    families conservatively use 8K/16K until measured on their own hardware.
    """

    if compute_capability is None:
        compute_capability = (
            torch.cuda.get_device_capability()
            if torch.cuda.is_available() else (0, 0)
        )
    major, minor = compute_capability
    # Local SM120 measurements put the crossover near 3--4K tokens for
    # 1--96 systems. Other families retain a deliberately conservative 8K
    # boundary until measured on their own hardware.
    threshold = 4096 if (major, minor) == (12, 0) else 8192
    chunk_size = int(os.environ.get(
        "LSSO_MASKED_TRACE_SPLIT_CHUNK",
        "1024" if major >= 9 else "512",
    ))
    if chunk_size <= 0:
        raise ValueError("LSSO_MASKED_TRACE_SPLIT_CHUNK must be positive")
    mode = os.environ.get("LSSO_MASKED_TRACE_SPLIT_N", "auto").lower()
    if mode in {"0", "off", "false", "no", "cta"}:
        return "cta", chunk_size
    if mode in {"1", "on", "true", "yes", "force", "split_n"}:
        return "split_n", chunk_size
    if systems <= 0 or sequence < threshold:
        return "cta", chunk_size
    if (
        padding_ratio_hint is not None
        and padding_ratio_hint >= 0.75
        and sequence < 2 * threshold
    ):
        return "cta", chunk_size
    return "split_n", chunk_size


def _compact_statistics(
    U: torch.Tensor,
    C: torch.Tensor,
    *,
    solve_dtype: torch.dtype,
    split_n: bool = False,
    chunk_size: int = _SPLIT_STATS_CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute compact Woodbury statistics, optionally sequence-parallel.

    The split-N path maps sequence chunks into an additional batch dimension,
    lets cuBLAS evaluate all partial products in parallel, and only reduces the
    compact ``r x r`` / ``r x d`` partials.  It bounds the K dimension of each
    GEMM and avoids assigning an ultra-long sequence to one native statistics
    CTA.  No token-sized intermediate is materialized.
    """
    if not split_n or U.shape[-2] <= chunk_size:
        Ut = U.transpose(1, 2)
        return (
            _bmm_accumulate(Ut, U, dtype=solve_dtype),
            _bmm_accumulate(Ut, C, dtype=solve_dtype),
        )
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    systems, sequence, rank = U.shape
    rhs_width = C.shape[2]
    full_chunks = sequence // chunk_size
    full_length = full_chunks * chunk_size
    gram = cross = None
    if full_chunks:
        # [systems, chunks, K, width] -> [systems * chunks, K, width].
        # Do not pad the complete activation: a short tail is cheaper to
        # evaluate as one additional compact GEMM.
        U_chunks = U[:, :full_length].reshape(
            systems * full_chunks, chunk_size, rank
        )
        C_chunks = C[:, :full_length].reshape(
            systems * full_chunks, chunk_size, rhs_width
        )
        Ut_chunks = U_chunks.transpose(1, 2)
        gram = _bmm_accumulate(
            Ut_chunks, U_chunks, dtype=solve_dtype
        ).view(systems, full_chunks, rank, rank).sum(dim=1)
        cross = _bmm_accumulate(
            Ut_chunks, C_chunks, dtype=solve_dtype
        ).view(systems, full_chunks, rank, rhs_width).sum(dim=1)
    if full_length < sequence:
        U_tail = U[:, full_length:]
        C_tail = C[:, full_length:]
        Ut_tail = U_tail.transpose(1, 2)
        gram_tail = _bmm_accumulate(Ut_tail, U_tail, dtype=solve_dtype)
        cross_tail = _bmm_accumulate(Ut_tail, C_tail, dtype=solve_dtype)
        gram = gram_tail if gram is None else gram + gram_tail
        cross = cross_tail if cross is None else cross + cross_tail
    assert gram is not None and cross is not None
    return gram, cross


def _backward_compact_statistics(
    U: torch.Tensor,
    Y: torch.Tensor,
    P: torch.Tensor,
    *,
    split_n: bool = False,
    chunk_size: int = _SPLIT_BACKWARD_CHUNK_SIZE,
    valid_mask: torch.Tensor | None = None,
    heads: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``Y.T@U``, ``P.T@U`` and per-system ``-<P,Y>``.

    For very long, low-batch workloads, sequence chunks become an additional
    GEMM batch dimension. This supplies enough parallel work without creating
    token-sized statistics. The caller still uses cuBLAS for the output-sized
    grad-U readout, which is faster than a serial per-system CTA at long N.
    """
    compact_dtype = _solve_dtype(U, Y, P)
    if not split_n or U.shape[1] <= chunk_size:
        native = try_dual_backward_statistics_tensorcore(
            U, Y, P, valid_mask=valid_mask, heads=heads
        )
        if native is not None:
            return native
        if U.ndim == 4:
            U = U.flatten(0, 1).contiguous()
            Y = Y.flatten(0, 1).contiguous()
            P = P.flatten(0, 1).contiguous()
        if valid_mask is not None:
            active = valid_mask.repeat_interleave(heads, dim=0).unsqueeze(-1)
            U = torch.where(active, U, torch.zeros((), device=U.device, dtype=U.dtype))
            Y = torch.where(active, Y, torch.zeros((), device=Y.device, dtype=Y.dtype))
            P = torch.where(active, P, torch.zeros((), device=P.device, dtype=P.dtype))
        return (
            _bmm_accumulate(Y.transpose(1, 2), U, dtype=compact_dtype),
            _bmm_accumulate(P.transpose(1, 2), U, dtype=compact_dtype),
            -(P * Y).sum(dim=(1, 2), dtype=_solve_dtype(P, Y)),
        )
    if U.ndim == 4:
        U = U.flatten(0, 1).contiguous()
        Y = Y.flatten(0, 1).contiguous()
        P = P.flatten(0, 1).contiguous()
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    if valid_mask is not None:
        active = valid_mask.repeat_interleave(heads, dim=0).unsqueeze(-1)
        U = torch.where(active, U, torch.zeros((), device=U.device, dtype=U.dtype))
        Y = torch.where(active, Y, torch.zeros((), device=Y.device, dtype=Y.dtype))
        P = torch.where(active, P, torch.zeros((), device=P.device, dtype=P.dtype))
    systems, sequence, rank = U.shape
    width = Y.shape[2]
    full_chunks = sequence // chunk_size
    full_length = full_chunks * chunk_size
    YtU = PtU = grad_mu = None
    if full_chunks:
        U_chunks = U[:, :full_length].reshape(
            systems * full_chunks, chunk_size, rank
        )
        Y_chunks = Y[:, :full_length].reshape(
            systems * full_chunks, chunk_size, width
        )
        P_chunks = P[:, :full_length].reshape(
            systems * full_chunks, chunk_size, width
        )
        YtU = _bmm_accumulate(
            Y_chunks.transpose(1, 2), U_chunks, dtype=compact_dtype
        ).view(systems, full_chunks, width, rank).sum(
            dim=1, dtype=compact_dtype
        )
        PtU = _bmm_accumulate(
            P_chunks.transpose(1, 2), U_chunks, dtype=compact_dtype
        ).view(systems, full_chunks, width, rank).sum(
            dim=1, dtype=compact_dtype
        )
        grad_mu = -(P_chunks * Y_chunks).sum(
            dim=(1, 2), dtype=_solve_dtype(P, Y)
        ).view(systems, full_chunks).sum(dim=1)
    if full_length < sequence:
        U_tail = U[:, full_length:]
        Y_tail = Y[:, full_length:]
        P_tail = P[:, full_length:]
        YtU_tail = _bmm_accumulate(
            Y_tail.transpose(1, 2), U_tail, dtype=compact_dtype
        )
        PtU_tail = _bmm_accumulate(
            P_tail.transpose(1, 2), U_tail, dtype=compact_dtype
        )
        grad_mu_tail = -(P_tail * Y_tail).sum(
            dim=(1, 2), dtype=_solve_dtype(P, Y)
        )
        YtU = YtU_tail if YtU is None else YtU + YtU_tail
        PtU = PtU_tail if PtU is None else PtU + PtU_tail
        grad_mu = (
            grad_mu_tail if grad_mu is None else grad_mu + grad_mu_tail
        )
    assert YtU is not None and PtU is not None and grad_mu is not None
    return YtU, PtU, grad_mu


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
    UtU_bh, UtC_bh = _compact_statistics(
        U_bh,
        C_bh,
        solve_dtype=solve_dtype,
        split_n=N >= _SPLIT_STATS_MIN_SEQUENCE,
    )
    UtU = UtU_bh.view(B, H, r, r)
    UtC = UtC_bh.view(B, H, r, dh)
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
    if K_readout.dtype != U_bh.dtype:
        K_readout = K_readout.to(U_bh.dtype)
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

        split_n = (
            N >= _SPLIT_BACKWARD_MIN_SEQUENCE
            and B * H <= _SPLIT_BACKWARD_MAX_SYSTEMS
        )
        # A custom autograd backward inherits the caller's autocast state.
        # Disable it here so FP32 fallback statistics cannot produce a BF16
        # bmm output and then fail an in-place FP32 baddbmm epilogue.
        with torch.autocast(device_type=U.device.type, enabled=False):
            YtU, PtU, grad_mu_flat = _backward_compact_statistics(
                U_m,
                Y_m,
                P_m,
                split_n=split_n,
                chunk_size=_SPLIT_BACKWARD_CHUNK_SIZE,
            )
            YtU_readout = YtU.to(matmul_dtype).contiguous()
            PtU_readout = PtU.to(matmul_dtype).contiguous()
            coefficient = -gamma.expand(B, H, 1, 1).reshape(B * H).float()
            grad_U_m = try_dual_grad_u_tensorcore(
                P_m, Y_m, YtU_readout, PtU_readout, coefficient
            )
            if grad_U_m is None:
                grad_U_m = torch.bmm(P_m, YtU_readout)
                # Accumulate the symmetric second term in the GEMM epilogue.
                grad_U_m.baddbmm_(Y_m, PtU_readout)
                grad_U_m.mul_(
                    gamma.expand(B, H, 1, 1)
                    .to(matmul_dtype)
                    .reshape(B * H, 1, 1)
                ).neg_()
        grad_U = grad_U_m.view(B, H, N, r).to(U.dtype)

        grad_C = P.to(grad_output.dtype)

        grad_mu_bh = grad_mu_flat.view(B, H, 1, 1)
        grad_gamma_bh = -(
            PtU.to(calc_dtype) * YtU.to(calc_dtype)
        ).sum(dim=(1, 2)).view(B, H, 1, 1)

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
    padding_ratio_hint: float | None = None,
) -> torch.Tensor:
    """Compatibility masked solve implemented through the compact fallback."""
    B, H, N, r = U.shape
    active = valid_mask[:, None, :, None]
    U_safe = torch.where(active, U, torch.zeros((), device=U.device, dtype=U.dtype))
    C_masked = torch.where(active, C, torch.zeros((), device=C.device, dtype=C.dtype))
    U_scaled = U_safe * length_scale.to(U.dtype).view(B, 1, 1, 1)
    return _lsso_woodbury_forward(U_scaled, C_masked, mu, gamma, eye)


class _MaskedLSSOAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, U, C, mu, gamma, eye, valid_mask, length_scale, padding_ratio_hint
    ):
        Y = _lsso_masked_woodbury_forward(
            U,
            C,
            mu,
            gamma,
            valid_mask,
            length_scale,
            eye,
            padding_ratio_hint,
        )
        ctx.save_for_backward(U, Y, mu, gamma, valid_mask, length_scale)
        ctx.padding_ratio_hint = padding_ratio_hint
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        U, Y, mu, gamma, valid_mask, length_scale = ctx.saved_tensors
        B, H, N, r = U.shape
        mask = valid_mask[:, None, :, None]
        grad = torch.where(
            mask,
            grad_output,
            torch.zeros((), device=grad_output.device, dtype=grad_output.dtype),
        ).contiguous()
        P = _lsso_masked_woodbury_forward(
            U,
            grad,
            mu,
            gamma,
            valid_mask,
            length_scale,
            padding_ratio_hint=ctx.padding_ratio_hint,
        )

        calc_dtype = torch.float64 if U.dtype == torch.float64 else torch.float32
        matmul_dtype = U.dtype if U.dtype in (torch.float16, torch.bfloat16) else calc_dtype
        # torch.where, unlike multiplication by a zero mask, prevents NaN or
        # Inf padding values from entering cuBLAS on hybrid/split-N paths.
        active = mask.to(torch.bool)
        U_safe = torch.where(active, U, torch.zeros((), device=U.device, dtype=U.dtype))
        Y_safe = torch.where(active, Y, torch.zeros((), device=Y.device, dtype=Y.dtype))
        P_safe = torch.where(active, P, torch.zeros((), device=P.device, dtype=P.dtype))
        U_m = U_safe.to(matmul_dtype).flatten(0, 1)
        Y_m = Y_safe.to(matmul_dtype).flatten(0, 1)
        P_m = P_safe.to(matmul_dtype).flatten(0, 1)
        split_n = (
            N >= _SPLIT_BACKWARD_MIN_SEQUENCE
            and B * H <= _SPLIT_BACKWARD_MAX_SYSTEMS
        )
        with torch.autocast(device_type=U.device.type, enabled=False):
            YtU, PtU, grad_mu_flat = _backward_compact_statistics(
                U_m,
                Y_m,
                P_m,
                split_n=split_n,
                chunk_size=_SPLIT_BACKWARD_CHUNK_SIZE,
            )
            YtU_readout = YtU.to(matmul_dtype).contiguous()
            PtU_readout = PtU.to(matmul_dtype).contiguous()
            scale2 = length_scale.square()[:, None].expand(B, H).reshape(B * H)
            coefficient = -(
                gamma.expand(B, H, 1, 1).reshape(B * H).float()
                * scale2.float()
            ).contiguous()
            grad_U_m = try_dual_grad_u_tensorcore(
                P_m, Y_m, YtU_readout, PtU_readout, coefficient
            )
            if grad_U_m is None:
                grad_U_m = torch.bmm(P_m, YtU_readout)
                grad_U_m.baddbmm_(Y_m, PtU_readout)
                grad_U_m.mul_(coefficient.to(matmul_dtype).view(B * H, 1, 1))
        grad_U = grad_U_m.view(B, H, N, r)
        grad_U.mul_(mask.to(grad_U.dtype))
        grad_U = grad_U.to(U.dtype)
        grad_C = P.to(grad_output.dtype)

        grad_mu_bh = grad_mu_flat.view(B, H, 1, 1)
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
            None,
        )


def _trace_gain_alpha_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    gain: torch.Tensor,
    effective_alpha: torch.Tensor,
    eye: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    padding_ratio_hint: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a solve and retain its compact Woodbury correction."""

    B, H, N, rank = U.shape
    width = C.shape[-1]
    alpha_flat = effective_alpha.expand(B, H, 1, 1).reshape(B * H).float().contiguous()
    gain_flat = gain.expand(B, H, 1, 1).reshape(B * H).float().contiguous()
    split_trace = False
    split_chunk_size = _SPLIT_STATS_CHUNK_SIZE
    if valid_mask is None:
        native = try_effective_stats_solve_readout(U, C, alpha_flat, gain_flat)
    else:
        strategy, split_chunk_size = _masked_trace_forward_strategy(
            N,
            B * H,
            padding_ratio_hint,
            compute_capability=(
                torch.cuda.get_device_capability(U.device)
                if U.is_cuda else (0, 0)
            ),
        )
        split_trace = strategy == "split_n"
        native = None if split_trace else try_masked_stats_solve_readout(
                U,
                C,
                valid_mask,
                torch.ones(B, device=U.device, dtype=torch.float32),
                alpha_flat,
                gain_flat,
                padding_ratio_hint=padding_ratio_hint,
            )
    if native is not None:
        return native
    if valid_mask is None:
        U_safe, C_safe = U, C
    else:
        active = valid_mask[:, None, :, None]
        U_safe = torch.where(active, U, torch.zeros((), device=U.device, dtype=U.dtype))
        C_safe = torch.where(active, C, torch.zeros((), device=C.device, dtype=C.dtype))
    solve_dtype = _solve_dtype(U, C)
    U_bh = U_safe.flatten(0, 1)
    C_bh = C_safe.flatten(0, 1)
    gram_bh, cross_bh = _compact_statistics(
        U_bh,
        C_bh,
        solve_dtype=solve_dtype,
        split_n=split_trace or N >= _SPLIT_STATS_MIN_SEQUENCE,
        chunk_size=split_chunk_size,
    )
    record_mathdx_path(
        "backward.adjoint_split_n" if split_trace
        else "backward.adjoint_torch"
    )
    if eye is None:
        eye = torch.eye(rank, device=U.device, dtype=solve_dtype).view(1, 1, rank, rank)
    effective = effective_alpha.expand(B, H, 1, 1).to(solve_dtype)
    system, balanced_cross = _balanced_woodbury_system(
        gram_bh.view(B, H, rank, rank),
        cross_bh.view(B, H, rank, width),
        effective,
        eye.to(solve_dtype),
    )
    compact = _solve_no_check(
        system.reshape(B * H, rank, rank),
        balanced_cross.reshape(B * H, rank, width),
    ).to(U_bh.dtype)
    output = torch.baddbmm(
        C_safe.flatten(0, 1), U_bh, compact, beta=1.0, alpha=-1.0
    ).view(B, H, N, width)
    output.mul_(gain.expand(B, H, 1, 1).to(output.dtype))
    return output, compact.view(B, H, rank, width)


def _trace_statistics_forward(
    U: torch.Tensor,
    C: torch.Tensor,
    gain: torch.Tensor,
    alpha: torch.Tensor,
    eye: torch.Tensor | None,
    valid_mask: torch.Tensor | None,
    *,
    eps: float,
    length_normalize: bool,
    length_reference: float,
    padding_ratio_hint: float | None = None,
    theta_alpha: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Statistics-first trace-normalized forward.

    ``trace(U.T @ U)`` is the relation-basis energy, so normalization is
    derived from the Gram statistic already required by Woodbury. This avoids
    the historical standalone energy pass and never materializes normalized
    ``U``. The returned tensors are the effective strength, denominator, and
    scale-squared needed by the analytic backward.
    """

    B, H, N, rank = U.shape
    width = C.shape[-1]
    log_alpha = (
        theta_alpha if theta_alpha is not None else torch.log(alpha)
    )
    split_trace = False
    split_chunk_size = _SPLIT_STATS_CHUNK_SIZE
    if valid_mask is None:
        native = try_trace_stats_solve_readout(
            U,
            C,
            log_alpha.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
            gain.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
            normalization_eps=eps,
            length_reference=length_reference,
            length_normalize=length_normalize,
            input_is_log=True,
        )
        if native is not None:
            return native
    if valid_mask is not None:
        strategy, split_chunk_size = _masked_trace_forward_strategy(
            N,
            B * H,
            padding_ratio_hint,
            compute_capability=(
                torch.cuda.get_device_capability(U.device)
                if U.is_cuda else (0, 0)
            ),
        )
        split_trace = strategy == "split_n"
        native = None if split_trace else try_masked_trace_stats_solve_readout(
                U,
                C,
                valid_mask,
                log_alpha.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
                gain.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
                normalization_eps=eps,
                length_reference=length_reference,
                length_normalize=length_normalize,
                padding_ratio_hint=padding_ratio_hint,
                input_is_log=True,
            )
        if native is not None:
            return native
    if valid_mask is None:
        U_safe = U
        C_safe = C
        lengths = torch.full(
            (B,), float(N), device=U.device, dtype=torch.float32
        )
    else:
        active = valid_mask[:, None, :, None]
        U_safe = torch.where(
            active, U, torch.zeros((), device=U.device, dtype=U.dtype)
        )
        C_safe = torch.where(
            active, C, torch.zeros((), device=C.device, dtype=C.dtype)
        )
        lengths = valid_mask.sum(dim=-1, dtype=torch.float32).clamp_min(1.0)

    solve_dtype = _solve_dtype(U, C)
    U_bh = U_safe.flatten(0, 1)
    C_bh = C_safe.flatten(0, 1)
    gram_bh, cross_bh = _compact_statistics(
        U_bh,
        C_bh,
        solve_dtype=solve_dtype,
        split_n=split_trace or N >= _SPLIT_STATS_MIN_SEQUENCE,
        chunk_size=split_chunk_size,
    )
    record_mathdx_path(
        "forward.trace_masked_split_n" if split_trace
        else "forward.trace_torch_fallback"
    )
    gram = gram_bh.view(B, H, rank, rank)
    cross = cross_bh.view(B, H, rank, width)
    energy = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    element_count = lengths.view(B, 1, 1, 1).to(solve_dtype) * rank
    denominator = energy + float(eps) * element_count
    if length_normalize:
        target = torch.full_like(denominator, rank * float(length_reference))
    else:
        target = element_count.expand_as(denominator)
    scale_squared = target / denominator.clamp_min(torch.finfo(solve_dtype).tiny)
    effective_alpha = (
        alpha.expand(B, H, 1, 1).to(solve_dtype) * scale_squared
    )

    if eye is None:
        eye = torch.eye(rank, device=U.device, dtype=solve_dtype).view(1, 1, rank, rank)
    system, balanced_cross = _balanced_woodbury_system(
        gram, cross, effective_alpha, eye.to(solve_dtype)
    )
    compact = _solve_no_check(
        system.reshape(B * H, rank, rank),
        balanced_cross.reshape(B * H, rank, width),
    )
    output_dtype = C.dtype if U.dtype == C.dtype else torch.promote_types(U.dtype, C.dtype)
    compact = compact.to(U_bh.dtype)
    output = torch.baddbmm(
        C_safe.flatten(0, 1), U_bh, compact, beta=1.0, alpha=-1.0
    ).view(B, H, N, width)
    output.mul_(gain.expand(B, H, 1, 1).to(output.dtype))
    return (
        output.to(output_dtype),
        effective_alpha,
        denominator,
        scale_squared,
        compact.view(B, H, rank, width),
    )


def _adaptive_trace_reference(
    U: torch.Tensor,
    C: torch.Tensor,
    gain: torch.Tensor,
    alpha: torch.Tensor,
    valid_mask: torch.Tensor | None,
    *,
    eps: float,
    length_normalize: bool,
    length_reference: float,
    input_is_log: bool = False,
) -> torch.Tensor:
    """Exact smaller-side solve used when at least one system has ``Nvalid < r``.

    Each sample uses only its valid rows.  The token-side branch therefore
    forms ``U U.T`` without first forming ``U.T U`` or ``U.T C``; the rank-side
    branch retains Woodbury.  The two branches implement the same SPD
    resolvent and are fully differentiable, including second derivatives.
    This portable path is also the numerical oracle for native kernels.
    """

    B, H, N, rank = U.shape
    width = C.shape[-1]
    solve_dtype = _solve_dtype(U, C)
    gain_bh = gain.expand(B, H, 1, 1).to(solve_dtype)
    log_alpha_bh = (
        alpha if input_is_log else torch.log(alpha)
    ).expand(B, H, 1, 1).to(solve_dtype)

    if valid_mask is None and N < rank:
        # Common fixed-length short-sequence case: one batched primal solve,
        # with no gather/scatter and no rank-space projection/readout.
        u_bh = U.flatten(0, 1).to(solve_dtype)
        c_bh = C.flatten(0, 1).to(solve_dtype)
        kernel = torch.bmm(u_bh, u_bh.transpose(1, 2))
        energy = kernel.diagonal(dim1=-2, dim2=-1).sum(-1).view(B, H, 1, 1)
        element_count = float(N * rank)
        target = (
            float(rank) * float(length_reference)
            if length_normalize else element_count
        )
        denominator = energy + float(eps) * element_count
        log_effective = log_alpha_bh + torch.log(
            target / denominator.clamp_min(torch.finfo(solve_dtype).tiny)
        )
        reciprocal = log_effective >= 0.0
        delta = torch.where(reciprocal, torch.exp(-log_effective), 1.0)
        eta = torch.where(reciprocal, 1.0, torch.exp(log_effective))
        eye = torch.eye(N, device=U.device, dtype=solve_dtype).expand(
            B * H, N, N
        )
        system = (
            delta.reshape(B * H, 1, 1) * eye
            + eta.reshape(B * H, 1, 1) * kernel
        )
        solved = _solve_no_check(
            system, delta.reshape(B * H, 1, 1) * c_bh
        )
        record_mathdx_path("forward.trace_primal_batched")
        return (
            solved.view(B, H, N, width)
            * gain_bh
        ).to(C.dtype)

    sample_outputs: list[torch.Tensor] = []
    used_primal = False
    used_dual = False

    for batch in range(B):
        if valid_mask is None:
            indices = torch.arange(N, device=U.device)
        else:
            indices = torch.nonzero(valid_mask[batch], as_tuple=False).flatten()
        active_tokens = int(indices.numel())
        if active_tokens == 0:
            # Retain a zero-valued graph edge so all-padding batches produce
            # explicit zero gradients instead of ``None``.
            zero_dependency = (
                U[batch].sum() + gain_bh[batch].sum()
                + log_alpha_bh[batch].sum()
            ) * 0.0
            sample_outputs.append(C[batch] * 0.0 + zero_dependency)
            continue

        u_active = U[batch].index_select(1, indices).to(solve_dtype)
        c_active = C[batch].index_select(1, indices).to(solve_dtype)
        element_count = float(active_tokens * rank)
        target = (
            float(rank) * float(length_reference)
            if length_normalize else element_count
        )
        eye_size = active_tokens if active_tokens < rank else rank
        eye = torch.eye(
            eye_size, device=U.device, dtype=solve_dtype
        ).expand(H, eye_size, eye_size)

        if active_tokens < rank:
            # Primal/token-space system.  Derive trace from K's diagonal so
            # normalization adds no standalone U-energy read.
            kernel = torch.bmm(u_active, u_active.transpose(1, 2))
            energy = kernel.diagonal(dim1=-2, dim2=-1).sum(-1).view(H, 1, 1)
            denominator = energy + float(eps) * element_count
            scale_squared = target / denominator.clamp_min(
                torch.finfo(solve_dtype).tiny
            )
            log_effective = (
                log_alpha_bh[batch] + torch.log(scale_squared)
            )
            reciprocal = log_effective >= 0.0
            delta = torch.where(reciprocal, torch.exp(-log_effective), 1.0)
            eta = torch.where(reciprocal, 1.0, torch.exp(log_effective))
            system = delta * eye + eta * kernel
            solved = _solve_no_check(system, delta * c_active)
            used_primal = True
        else:
            gram = torch.bmm(u_active.transpose(1, 2), u_active)
            energy = gram.diagonal(dim1=-2, dim2=-1).sum(-1).view(H, 1, 1)
            denominator = energy + float(eps) * element_count
            scale_squared = target / denominator.clamp_min(
                torch.finfo(solve_dtype).tiny
            )
            log_effective = (
                log_alpha_bh[batch] + torch.log(scale_squared)
            )
            reciprocal = log_effective >= 0.0
            delta = torch.where(reciprocal, torch.exp(-log_effective), 1.0)
            eta = torch.where(reciprocal, 1.0, torch.exp(log_effective))
            cross = torch.bmm(u_active.transpose(1, 2), c_active)
            system = delta * eye + eta * gram
            balanced_cross = eta * cross
            compact = _solve_no_check(system, balanced_cross)
            solved = torch.baddbmm(c_active, u_active, compact, alpha=-1.0)
            used_dual = True

        solved = solved * gain_bh[batch]
        full = C[batch].new_zeros((H, N, width))
        sample_outputs.append(
            full.index_copy(1, indices, solved.to(C.dtype))
        )

    record_mathdx_path(
        "forward.trace_adaptive_mixed" if used_primal and used_dual
        else "forward.trace_primal_torch"
    )
    return torch.stack(sample_outputs, dim=0)


def _requires_primal_trace(
    U: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> bool:
    """Return whether any logical system is wider than its valid token side."""

    rank = U.shape[-1]
    if U.shape[-2] < rank:
        return True
    if valid_mask is None:
        return False
    # A shared padding mask is normally reused by every mixer layer. Cache the
    # one required device reduction by object identity and tensor version so a
    # deep masked model does not introduce one CPU synchronization per layer.
    key = id(valid_mask)
    version = valid_mask._version
    cached = _MASK_REQUIRES_PRIMAL_CACHE.get(key)
    if (
        cached is not None
        and cached[0]() is valid_mask
        and cached[1] == version
        and cached[2] == rank
    ):
        return cached[3]
    result = bool((valid_mask.sum(dim=-1) < rank).any().item())
    _MASK_REQUIRES_PRIMAL_CACHE[key] = (
        weakref.ref(valid_mask), version, rank, result
    )
    if len(_MASK_REQUIRES_PRIMAL_CACHE) > 64:
        dead = [
            cache_key for cache_key, entry in _MASK_REQUIRES_PRIMAL_CACHE.items()
            if entry[0]() is None
        ]
        for cache_key in dead:
            _MASK_REQUIRES_PRIMAL_CACHE.pop(cache_key, None)
    return result


def _trace_gain_alpha_backward(
    U: torch.Tensor,
    Y: torch.Tensor,
    grad_output: torch.Tensor,
    gain: torch.Tensor,
    alpha: torch.Tensor,
    theta_alpha: torch.Tensor,
    effective_alpha: torch.Tensor,
    scale_squared: torch.Tensor,
    denominator: torch.Tensor,
    forward_compact: torch.Tensor,
    valid_mask: torch.Tensor | None,
    padding_ratio_hint: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward for ``alpha_eff = alpha * tau / (||U||_F^2 + eps)``.

    The first term differentiates the solve while holding ``gamma_eff`` fixed.
    The final radial term is the derivative through its U-dependent scale.
    Returning per-system scalar gradients here makes broadcast reduction
    explicit in the autograd wrapper.
    """

    B, H, N, rank = U.shape
    if valid_mask is None:
        grad = grad_output
        mask = None
    else:
        mask = valid_mask[:, None, :, None].to(device=U.device, dtype=torch.bool)
        # The native adjoint solve predicates U/C loads with the mask. Keep the
        # poisoned-padding guard in its kernel instead of materializing three
        # output-sized torch.where tensors before dispatch.
        grad = grad_output

    P, adjoint_compact = _trace_gain_alpha_forward(
        U,
        grad,
        gain,
        effective_alpha,
        None,
        valid_mask,
        padding_ratio_hint,
    )
    calc_dtype = (
        torch.float64
        if U.dtype == torch.float64 or Y.dtype == torch.float64
        else torch.float32
    )
    matmul_dtype = U.dtype if U.dtype in (torch.float16, torch.bfloat16) else calc_dtype
    U_native = U.to(matmul_dtype)
    Y_native = Y.to(matmul_dtype)
    P_native = P.to(matmul_dtype)
    gain_flat = gain.expand(B, H, 1, 1).reshape(B * H).float().contiguous()
    effective_alpha_flat = effective_alpha.reshape(B * H).float().contiguous()
    compact_q = forward_compact.reshape(B * H, rank, -1).to(matmul_dtype)
    compact_r = adjoint_compact.reshape(B * H, rank, -1).to(matmul_dtype)
    with torch.autocast(device_type=U.device.type, enabled=False):
        # q=(G+beta I)^-1 U^T C and r=(G+beta I)^-1 U^T dY
        # expose a cancellation-free derivative:
        #   dU_direct = -(P q^T + Y r^T)
        #   d log(alpha_eff) = -g beta <r,q>.
        # Unlike the historical alpha*(Y^T U/P^T U) form, neither expression
        # subtracts nearly equal high-alpha quantities.
        log_effective_alpha = (
            theta_alpha.expand(B, H, 1, 1).reshape(B * H).to(calc_dtype)
            + torch.log(
                scale_squared.reshape(B * H).to(calc_dtype).clamp_min(
                    torch.finfo(calc_dtype).tiny
                )
            )
        )
        beta = torch.exp(-log_effective_alpha)
        compact_inner = (
            compact_q.to(calc_dtype) * compact_r.to(calc_dtype)
        ).sum(dim=(1, 2))
        grad_log_effective_alpha = (
            -gain_flat.to(calc_dtype) * beta * compact_inner
        )
        if mask is None:
            gain_inner = (grad_output.to(calc_dtype) * Y.to(calc_dtype)).sum(
                dim=(2, 3)
            )
        else:
            safe_product = torch.where(
                mask,
                grad_output.to(calc_dtype) * Y.to(calc_dtype),
                torch.zeros((), device=U.device, dtype=calc_dtype),
            )
            gain_inner = safe_product.sum(dim=(2, 3))
        grad_gain_flat = (
            gain_inner.reshape(B * H) / gain_flat.to(calc_dtype)
        )
        radial_coefficient = (
            -2.0
            * grad_log_effective_alpha.to(calc_dtype)
            / denominator.reshape(B * H).to(calc_dtype)
        )
        direct_grad = try_dual_grad_u_tensorcore(
            P_native,
            Y_native,
            compact_q.transpose(1, 2).contiguous(),
            compact_r.transpose(1, 2).contiguous(),
            -torch.ones_like(effective_alpha_flat),
            radial_u=U_native,
            radial_coefficient=radial_coefficient.float().contiguous(),
            valid_mask=valid_mask,
            heads=H,
        )
        radial_fused = direct_grad is not None
        if direct_grad is None:
            U_m = U_native.flatten(0, 1).contiguous()
            Y_m = Y_native.flatten(0, 1).contiguous()
            P_m = P_native.flatten(0, 1).contiguous()
            direct_grad = torch.bmm(
                P_m, compact_q.transpose(1, 2).contiguous()
            )
            direct_grad.baddbmm_(
                Y_m, compact_r.transpose(1, 2).contiguous()
            )
            direct_grad.neg_()
    # d log(alpha_eff) / dU = -2 U / denominator.  Returning the
    # log-strength derivative directly avoids an underflow-prone
    # dL/dalpha * exp(theta_alpha) chain at very large strengths.
    radial_coefficient = (
        -2.0
        * grad_log_effective_alpha.to(calc_dtype)
        / denominator.reshape(B * H).to(calc_dtype)
    )
    if radial_fused:
        grad_U = direct_grad.view(B, H, N, rank)
    else:
        grad_U_m = direct_grad.to(calc_dtype)
        radial_source = U_m.to(calc_dtype)
        if valid_mask is not None:
            active = valid_mask.repeat_interleave(H, dim=0).unsqueeze(-1)
            radial_source = torch.where(
                active,
                radial_source,
                torch.zeros((), device=U.device, dtype=calc_dtype),
            )
        grad_U_m.add_(
            radial_source * radial_coefficient.view(B * H, 1, 1)
        )
        grad_U = grad_U_m.view(B, H, N, rank)
        if mask is not None:
            grad_U = torch.where(
                mask, grad_U, torch.zeros((), device=grad_U.device, dtype=grad_U.dtype)
            )

    grad_gain_bh = grad_gain_flat.view(B, H, 1, 1).to(calc_dtype)
    grad_theta_bh = grad_log_effective_alpha.view(B, H, 1, 1).to(calc_dtype)
    return (
        grad_U.to(U.dtype),
        P.to(grad_output.dtype),
        grad_gain_bh,
        grad_theta_bh,
    )


class _TraceNormalizedLSSOAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U,
        C,
        gain,
        theta_alpha,
        eye,
        valid_mask,
        eps,
        length_normalize,
        length_reference,
        padding_ratio_hint,
    ):
        alpha = torch.exp(theta_alpha)
        (
            Y,
            effective_alpha,
            denominator,
            scale_squared,
            forward_compact,
        ) = _trace_statistics_forward(
            U,
            C,
            gain,
            alpha,
            eye,
            valid_mask,
            eps=float(eps),
            length_normalize=bool(length_normalize),
            length_reference=float(length_reference),
            padding_ratio_hint=padding_ratio_hint,
            theta_alpha=theta_alpha,
        )
        mask_tensor = (
            valid_mask
            if valid_mask is not None
            else torch.empty(0, device=U.device, dtype=torch.bool)
        )
        eye_tensor = (
            eye if eye is not None
            else torch.empty(0, device=U.device, dtype=torch.float32)
        )
        # Saving a token-major C view on CUDA would retain the complete joint
        # UC projection storage, including the otherwise-dead U segment. The
        # maintained low-precision CUDA path is first-order; keep C only for
        # portable/reference executions that support create_graph=True.
        save_higher_order_c = not U.is_cuda or os.environ.get(
            "LSSO_CUDA_HIGHER_ORDER", "0"
        ).lower() in {"1", "on", "true", "yes"}
        saved_c = C if save_higher_order_c else torch.empty(
            0, device=C.device, dtype=C.dtype
        )
        ctx.save_for_backward(
            U,
            saved_c,
            Y,
            gain,
            theta_alpha,
            effective_alpha,
            scale_squared,
            denominator,
            forward_compact,
            mask_tensor,
            eye_tensor,
        )
        ctx.has_mask = valid_mask is not None
        ctx.has_eye = eye is not None
        ctx.has_higher_order_c = save_higher_order_c
        ctx.eps = float(eps)
        ctx.length_normalize = bool(length_normalize)
        ctx.length_reference = float(length_reference)
        ctx.padding_ratio_hint = padding_ratio_hint
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        (
            U,
            C,
            Y,
            gain,
            theta_alpha,
            effective_alpha,
            scale_squared,
            denominator,
            forward_compact,
            mask_tensor,
            eye_tensor,
        ) = ctx.saved_tensors
        valid_mask = mask_tensor if ctx.has_mask else None
        if torch.is_grad_enabled():
            # The first-order training path uses the analytic/native backward.
            # For create_graph=True, recompute the differentiable reference so
            # saved forward outputs are not incorrectly treated as constants
            # by a second derivative.
            if not ctx.has_higher_order_c:
                raise RuntimeError(
                    "higher-order gradients are available on the portable "
                    "reference path; CUDA low-precision Trace training uses "
                    "the memory-optimized first-order backward"
                )
            eye = eye_tensor if ctx.has_eye else None
            with torch.enable_grad():
                reference, _, _, _, _ = _trace_statistics_forward(
                    U,
                    C,
                    gain,
                    torch.exp(theta_alpha),
                    eye,
                    valid_mask,
                    eps=ctx.eps,
                    length_normalize=ctx.length_normalize,
                    length_reference=ctx.length_reference,
                    padding_ratio_hint=ctx.padding_ratio_hint,
                )
                inputs = (U, C, gain, theta_alpha)
                required = [value for value in inputs if value.requires_grad]
                computed = torch.autograd.grad(
                    reference,
                    required,
                    grad_output,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
            iterator = iter(computed)
            gradients = tuple(
                next(iterator) if value.requires_grad else None
                for value in inputs
            )
            record_mathdx_path("backward.trace_higher_order_torch")
            return (
                gradients[0], gradients[1], gradients[2], gradients[3],
                None, None, None, None, None, None,
            )
        B, H = U.shape[:2]
        alpha = torch.exp(theta_alpha)
        grad_U, grad_C, grad_gain_bh, grad_theta_bh = _trace_gain_alpha_backward(
            U,
            Y,
            grad_output,
            gain,
            alpha,
            theta_alpha,
            effective_alpha,
            scale_squared,
            denominator,
            forward_compact,
            valid_mask,
            ctx.padding_ratio_hint,
        )
        grad_gain = (
            grad_gain_bh.sum(dim=0, keepdim=True)
            if gain.shape[0] == 1
            else grad_gain_bh
        )
        grad_theta = (
            grad_theta_bh.sum(dim=0, keepdim=True)
            if theta_alpha.shape[0] == 1
            else grad_theta_bh
        )
        return (
            grad_U,
            grad_C,
            grad_gain.to(gain.dtype),
            grad_theta.to(theta_alpha.dtype),
            None,
            None,
            None,
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
    trace_normalize: bool = False,
    normalization_eps: float = 1e-5,
    valid_mask: torch.Tensor | None = None,
    padding_ratio_hint: float | None = None,
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
        trace_normalize: absorb a per-sample/head Gram-trace normalization
            into the Woodbury coefficient while preserving token magnitudes.
        normalization_eps: RMS-style epsilon used by trace normalization.
        valid_mask: optional [B, N] mask used for effective lengths. Masked U/C
            entries are zeroed here as a safety measure.
        padding_ratio_hint: optional CPU-side padding fraction used to select
            the native masked kernel without synchronizing the CUDA mask.

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

    if trace_normalize and not no_global:
        gain = mu.reciprocal()
        alpha = gamma * gain
        return lsso_gain_alpha(
            U,
            C,
            gain,
            alpha,
            eye=eye,
            no_global=False,
            return_aux=return_aux,
            length_normalize=length_normalize,
            length_reference=length_reference,
            trace_normalize=True,
            normalization_eps=normalization_eps,
            valid_mask=valid_mask,
            padding_ratio_hint=padding_ratio_hint,
        )

    if valid_mask is not None:
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
                    U,
                    C,
                    mu,
                    gamma,
                    eye,
                    valid_mask,
                    length_scale,
                    padding_ratio_hint,
                )
            return _lsso_masked_woodbury_forward(
                U,
                C,
                mu,
                gamma,
                valid_mask,
                length_scale,
                eye,
                padding_ratio_hint,
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
        UtU = _bmm_accumulate(Ut_bh, U_bh, dtype=solve_dtype).view(B, H, r, r)
        UtC = _bmm_accumulate(Ut_bh, C_bh, dtype=solve_dtype).view(B, H, r, dh)

        if eye is None:
            eye = torch.eye(r, device=U.device, dtype=solve_dtype).view(1, 1, r, r)
        G = eye.to(solve_dtype) + gamma_over_mu.to(solve_dtype) * UtU
        K = _solve_no_check(
            G.view(B * H, r, r),
            UtC.to(solve_dtype).view(B * H, r, dh),
        ).to(U.dtype)

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


def lsso_gain_alpha(
    U: torch.Tensor,
    C: torch.Tensor,
    gain: torch.Tensor,
    alpha: torch.Tensor,
    *,
    eye: torch.Tensor | None = None,
    no_global: bool = False,
    return_aux: bool = False,
    length_normalize: bool = True,
    length_reference: float = 1.0,
    trace_normalize: bool = False,
    normalization_eps: float = 1e-5,
    valid_mask: torch.Tensor | None = None,
    padding_ratio_hint: float | None = None,
    _log_alpha: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LSSOAux]:
    """Canonical structured solve parameterized by output gain and strength.

    The historical :func:`lsso` functional remains a compatibility surface for
    explicit ``mu, gamma`` callers. Maintained modules use this function so the
    trainable scalars and their gradients retain direct ``g, alpha`` meaning.
    """

    B, H = U.shape[:2]
    if gain.dim() == 1:
        gain = gain.view(1, H, 1, 1)
    if alpha.dim() == 1:
        alpha = alpha.view(1, H, 1, 1)
    theta_alpha = alpha if _log_alpha else torch.log(alpha)
    positive_alpha = None if _log_alpha else alpha
    if valid_mask is not None:
        if valid_mask.shape != (B, U.shape[2]):
            raise ValueError(
                f"valid_mask must have shape {(B, U.shape[2])}, "
                f"got {tuple(valid_mask.shape)}"
            )
        valid_mask = valid_mask.to(device=U.device, dtype=torch.bool).contiguous()

    # Auxiliary diagnostics intentionally use the differentiable reference
    # construction. This uncommon path exposes normalized Gram statistics and
    # does not participate in the optimized training dispatcher.
    if trace_normalize and not no_global and return_aux:
        normalized_U = trace_normalize_basis(
            U,
            valid_mask,
            eps=normalization_eps,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        mu = gain.reciprocal()
        gamma = (
            torch.exp(theta_alpha) if positive_alpha is None else positive_alpha
        ) * mu
        return lsso(
            normalized_U,
            C,
            mu,
            gamma,
            eye=eye,
            no_global=False,
            return_aux=True,
            length_normalize=False,
            length_reference=length_reference,
            trace_normalize=False,
            valid_mask=valid_mask,
            padding_ratio_hint=padding_ratio_hint,
        )

    if trace_normalize and not no_global:
        if _requires_primal_trace(U, valid_mask):
            return _adaptive_trace_reference(
                U,
                C,
                gain,
                theta_alpha if _log_alpha else positive_alpha,
                valid_mask,
                eps=normalization_eps,
                length_normalize=length_normalize,
                length_reference=length_reference,
                input_is_log=_log_alpha,
            )
        if torch.is_grad_enabled() and (
            U.requires_grad
            or C.requires_grad
            or gain.requires_grad
            or alpha.requires_grad
        ):
            return _TraceNormalizedLSSOAutograd.apply(
                U,
                C,
                gain,
                theta_alpha,
                eye,
                valid_mask,
                normalization_eps,
                length_normalize,
                length_reference,
                padding_ratio_hint,
            )
        (
            output,
            _effective_alpha,
            _denominator,
            _scale_squared,
            _compact,
        ) = _trace_statistics_forward(
            U,
            C,
            gain,
            (
                torch.exp(theta_alpha)
                if positive_alpha is None else positive_alpha
            ),
            eye,
            valid_mask,
            eps=normalization_eps,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        return output

    mu = gain.reciprocal()
    gamma = (
        torch.exp(theta_alpha) if positive_alpha is None else positive_alpha
    ) * mu
    return lsso(
        U,
        C,
        mu,
        gamma,
        eye=eye,
        no_global=no_global,
        return_aux=return_aux,
        length_normalize=length_normalize,
        length_reference=length_reference,
        trace_normalize=False,
        normalization_eps=normalization_eps,
        valid_mask=valid_mask,
        padding_ratio_hint=padding_ratio_hint,
    )


class LSSO(nn.Module):
    """
    LSSO v1: Learnable Structured Solve Operator.

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
        gain_init: float = DEFAULT_GAIN_INIT,
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

        _initialize_solve_parameters(
            self,
            num_heads,
            gain_init=gain_init,
        )

        self.dropout_p = dropout
        self.record_diagnostics = False
        self.prune_rank_keep: int | None = None
        self.last_diagnostics: LSSODiagnostics | None = None

    def effective_gain_alpha(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the positive output gain and relative solve strength."""

        return _solve_parameters(self)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        *,
        padding_ratio_hint: float | None = None,
    ) -> torch.Tensor:
        B, N, D = x.shape
        H = self.num_heads
        dh = self.head_dim
        r = self.rank
        UC = self.w_uc(x)
        U, C = UC.split((H * r, D), dim=-1)
        U = U.view(B, N, H, r).transpose(1, 2).contiguous()
        C = C.view(B, N, H, dh).transpose(1, 2)

        pruning_active = self.prune_rank_keep is not None and 0 < self.prune_rank_keep < r
        trace_basis = self.normalize_u
        solve_eye = self._eye
        if pruning_active:
            keep = int(self.prune_rank_keep)
            if valid_mask is not None:
                head_mask = valid_mask[:, None, :, None].to(
                    device=x.device,
                    dtype=x.dtype,
                )
                # Rank selection must ignore padding.  The normal solve path
                # applies the mask once inside lsso(), so avoid masking twice.
                U = U * head_mask
            scores = U.float().square().mean(dim=-2)
            indices = scores.topk(k=keep, dim=-1, largest=True, sorted=False).indices
            U = U.gather(-1, indices[:, :, None, :].expand(B, H, N, keep))
            solve_eye = None

        gain, theta_alpha = _solve_log_parameters(self)

        gain = gain.view(1, H, 1, 1)
        theta_alpha = theta_alpha.view(1, H, 1, 1)
        if self.record_diagnostics:
            Y, aux = lsso_gain_alpha(
                U,
                C,
                gain,
                torch.exp(theta_alpha),
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
                theta_alpha,
                eye=solve_eye,
                no_global=self._global_disabled,
                length_normalize=self.length_normalize,
                length_reference=self.length_reference,
                trace_normalize=trace_basis,
                normalization_eps=self.eps,
                valid_mask=valid_mask,
                padding_ratio_hint=padding_ratio_hint,
                _log_alpha=True,
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
