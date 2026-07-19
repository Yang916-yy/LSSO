from __future__ import annotations

import os

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
DEFAULT_ALPHA_INIT = 1.2
DEFAULT_ALPHA_MAX = 3.0
_LEGACY_GAIN_ALPHA_MAX = 2.0


def _initialize_solve_parameters(
    module: nn.Module,
    count: int,
    *,
    solve_parameterization: str,
    gain_init: float,
    alpha_init: float,
    alpha_max: float,
) -> None:
    """Create the canonical gain/strength parameter pair."""

    if solve_parameterization not in {"gain_alpha", "fixed_gain_alpha"}:
        raise ValueError(
            "solve_parameterization must be 'gain_alpha' or 'fixed_gain_alpha', "
            f"got {solve_parameterization!r}"
        )
    if gain_init <= 0:
        raise ValueError(f"gain_init must be positive, got {gain_init}")
    if alpha_max < 0:
        raise ValueError(f"alpha_max must be non-negative, got {alpha_max}")
    if not 0 <= alpha_init <= alpha_max:
        raise ValueError(
            f"alpha_init must lie in [0, alpha_max], got {alpha_init} "
            f"for alpha_max={alpha_max}"
        )

    module.solve_parameterization = solve_parameterization
    module.alpha_max = float(alpha_max)
    module.register_buffer(
        "_alpha_max_state",
        torch.tensor(float(alpha_max), dtype=torch.float32),
        persistent=True,
    )
    module._global_disabled = bool(module.no_global or alpha_max == 0.0)
    if alpha_max == 0:
        theta_alpha0 = torch.tensor(0.0, dtype=torch.float64)
    else:
        fraction = torch.tensor(alpha_init / float(alpha_max), dtype=torch.float64)
        if not 0.0 < float(fraction) < 1.0:
            raise ValueError(
                "finite sigmoid initialization requires "
                f"0 < alpha_init < alpha_max, got {alpha_init}, {alpha_max}"
            )
        theta_alpha0 = torch.logit(fraction)
    if solve_parameterization == "gain_alpha":
        module.theta_gain = nn.Parameter(
            torch.full(
                (count,),
                float(torch.log(torch.tensor(gain_init, dtype=torch.float64))),
                dtype=torch.float32,
            )
        )
    else:
        module.register_buffer(
            "_matched_initial_gain",
            torch.full((count,), float(gain_init), dtype=torch.float32),
            persistent=False,
        )
        module._fixed_gain_folded = False
    module.theta_alpha = nn.Parameter(
        torch.full((count,), float(theta_alpha0), dtype=torch.float32)
    )


def _solve_parameters(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the canonical per-head output gain and solve strength."""

    alpha_max = module._alpha_max_state.to(dtype=module.theta_alpha.dtype)
    if module.solve_parameterization == "gain_alpha":
        gain = torch.exp(module.theta_gain)
        alpha = alpha_max * torch.sigmoid(module.theta_alpha)
    else:
        gain = torch.ones_like(module.theta_alpha)
        alpha = alpha_max * torch.sigmoid(module.theta_alpha)
    if module.no_global:
        alpha = torch.zeros_like(alpha)
    return gain, alpha


def _solve_coefficients(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy adapter returning ``mu, gamma`` for historical callers."""

    gain, alpha = _solve_parameters(module)
    mu = gain.reciprocal()
    return mu, alpha * mu


def _legacy_solve_state_dict_pre_hook(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Migrate historical theta_mu/theta_gamma checkpoints on load.

    Historical released configurations used gamma_max=1.2. The conversion is
    exact whenever the resulting alpha lies inside the new configured bound.
    """

    del local_metadata, strict, missing_keys, unexpected_keys
    max_key = prefix + "_alpha_max_state"
    mu_key = prefix + "theta_mu"
    gamma_key = prefix + "theta_gamma"
    theta_alpha_key = prefix + "theta_alpha"
    legacy_mu_gamma = mu_key in state_dict or gamma_key in state_dict
    if max_key not in state_dict:
        # Gain/alpha checkpoints created before the ceiling became persistent
        # used alpha_max=2 by default. Preserve their represented function on
        # raw state-dict load. An explicitly non-default constructor ceiling is
        # respected for uncommon historical experiments.
        fallback = (
            module.alpha_max
            if module.alpha_max != DEFAULT_ALPHA_MAX
            else _LEGACY_GAIN_ALPHA_MAX
        )
        state_dict[max_key] = torch.tensor(
            module.alpha_max if legacy_mu_gamma else fallback,
            dtype=module._alpha_max_state.dtype,
            device=module._alpha_max_state.device,
        )
    if not legacy_mu_gamma:
        return
    if module.solve_parameterization != "gain_alpha":
        error_msgs.append(
            f"{prefix[:-1]}: legacy solve scalars can only migrate into gain_alpha"
        )
        return
    if mu_key not in state_dict or gamma_key not in state_dict:
        error_msgs.append(f"{prefix[:-1]}: incomplete legacy solve scalar pair")
        return
    theta_mu = state_dict.pop(mu_key).float()
    theta_gamma = state_dict.pop(gamma_key).float()
    mu = F.softplus(theta_mu) + module.eps
    gain = mu.reciprocal()
    alpha = 1.2 * torch.sigmoid(theta_gamma) / mu
    alpha_max = float(state_dict[max_key].item())
    if torch.any(alpha <= 0) or torch.any(alpha >= alpha_max):
        error_msgs.append(
            f"{prefix[:-1]}: legacy alpha falls outside (0, alpha_max={alpha_max})"
        )
        return
    state_dict[prefix + "theta_gain"] = gain.log()
    state_dict[theta_alpha_key] = torch.logit(alpha / alpha_max)


def _sync_alpha_max_after_load(module: nn.Module, incompatible_keys) -> None:
    """Keep the public scalar mirror aligned with checkpoint metadata."""

    del incompatible_keys
    module.alpha_max = float(module._alpha_max_state.item())


def _fold_fixed_gain_into_output(
    module: nn.Module,
    *,
    groups: int,
    group_width: int,
    force: bool = False,
) -> None:
    """Absorb the matched initial head gain into ``W_O`` exactly once."""

    if module.solve_parameterization != "fixed_gain_alpha":
        return
    if module._fixed_gain_folded and not force:
        return
    with torch.no_grad():
        scale = module._matched_initial_gain.to(
            device=module.w_o.weight.device,
            dtype=module.w_o.weight.dtype,
        ).repeat_interleave(group_width)
        if scale.numel() != groups * group_width:
            raise RuntimeError("fixed-gain output-fold shape mismatch")
        module.w_o.weight.mul_(scale.unsqueeze(0))
    module._fixed_gain_folded = True


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
    chunks = (sequence + chunk_size - 1) // chunk_size
    padded_sequence = chunks * chunk_size
    if padded_sequence != sequence:
        U = F.pad(U, (0, 0, 0, padded_sequence - sequence))
        C = F.pad(C, (0, 0, 0, padded_sequence - sequence))
    # [systems, chunks, K, width] -> [systems * chunks, K, width]
    U_chunks = U.view(systems, chunks, chunk_size, rank).flatten(0, 1)
    C_chunks = C.view(systems, chunks, chunk_size, rhs_width).flatten(0, 1)
    Ut_chunks = U_chunks.transpose(1, 2)
    gram = _bmm_accumulate(Ut_chunks, U_chunks, dtype=solve_dtype)
    cross = _bmm_accumulate(Ut_chunks, C_chunks, dtype=solve_dtype)
    return (
        gram.view(systems, chunks, rank, rank).sum(dim=1),
        cross.view(systems, chunks, rank, rhs_width).sum(dim=1),
    )


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
    chunks = (sequence + chunk_size - 1) // chunk_size
    padded_sequence = chunks * chunk_size
    if padded_sequence != sequence:
        amount = padded_sequence - sequence
        U = F.pad(U, (0, 0, 0, amount))
        Y = F.pad(Y, (0, 0, 0, amount))
        P = F.pad(P, (0, 0, 0, amount))
    U_chunks = U.view(systems, chunks, chunk_size, rank).flatten(0, 1)
    Y_chunks = Y.view(systems, chunks, chunk_size, width).flatten(0, 1)
    P_chunks = P.view(systems, chunks, chunk_size, width).flatten(0, 1)
    YtU = _bmm_accumulate(
        Y_chunks.transpose(1, 2), U_chunks, dtype=compact_dtype
    )
    PtU = _bmm_accumulate(
        P_chunks.transpose(1, 2), U_chunks, dtype=compact_dtype
    )
    grad_mu = -(P_chunks * Y_chunks).sum(
        dim=(1, 2), dtype=_solve_dtype(P, Y)
    )
    return (
        YtU.view(systems, chunks, width, rank).sum(dim=1, dtype=compact_dtype),
        PtU.view(systems, chunks, width, rank).sum(dim=1, dtype=compact_dtype),
        grad_mu.view(systems, chunks).sum(dim=1),
    )


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
) -> torch.Tensor:
    """Evaluate a solve with an already normalized effective strength."""

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
    beta = effective_alpha.expand(B, H, 1, 1).to(solve_dtype)
    system = eye.to(solve_dtype) + beta * gram_bh.view(B, H, rank, rank)
    compact = _solve_no_check(
        system.reshape(B * H, rank, rank),
        cross_bh.reshape(B * H, rank, width),
    ).to(U_bh.dtype)
    compact.mul_(beta.reshape(B * H, 1, 1).to(compact.dtype))
    output = torch.baddbmm(
        C_safe.flatten(0, 1), U_bh, compact, beta=1.0, alpha=-1.0
    ).view(B, H, N, width)
    output.mul_(gain.expand(B, H, 1, 1).to(output.dtype))
    return output


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Statistics-first trace-normalized forward.

    ``trace(U.T @ U)`` is the relation-basis energy, so normalization is
    derived from the Gram statistic already required by Woodbury. This avoids
    the historical standalone energy pass and never materializes normalized
    ``U``. The returned tensors are the effective strength, denominator, and
    scale-squared needed by the analytic backward.
    """

    B, H, N, rank = U.shape
    width = C.shape[-1]
    split_trace = False
    split_chunk_size = _SPLIT_STATS_CHUNK_SIZE
    if valid_mask is None:
        native = try_trace_stats_solve_readout(
            U,
            C,
            alpha.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
            gain.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
            normalization_eps=eps,
            length_reference=length_reference,
            length_normalize=length_normalize,
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
                alpha.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
                gain.expand(B, H, 1, 1).reshape(B * H).float().contiguous(),
                normalization_eps=eps,
                length_reference=length_reference,
                length_normalize=length_normalize,
                padding_ratio_hint=padding_ratio_hint,
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
    system = eye.to(solve_dtype) + effective_alpha * gram
    compact = _solve_no_check(
        system.reshape(B * H, rank, rank),
        cross.reshape(B * H, rank, width),
    )
    output_dtype = C.dtype if U.dtype == C.dtype else torch.promote_types(U.dtype, C.dtype)
    compact = compact.to(U_bh.dtype)
    scaled_compact = compact * effective_alpha.reshape(B * H, 1, 1).to(compact.dtype)
    output = torch.baddbmm(
        C_safe.flatten(0, 1), U_bh, scaled_compact, beta=1.0, alpha=-1.0
    ).view(B, H, N, width)
    output.mul_(gain.expand(B, H, 1, 1).to(output.dtype))
    return (
        output.to(output_dtype),
        effective_alpha,
        denominator,
        scale_squared,
    )


def _trace_gain_alpha_backward(
    U: torch.Tensor,
    Y: torch.Tensor,
    grad_output: torch.Tensor,
    gain: torch.Tensor,
    alpha: torch.Tensor,
    effective_alpha: torch.Tensor,
    scale_squared: torch.Tensor,
    denominator: torch.Tensor,
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

    P = _trace_gain_alpha_forward(
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
    effective_gamma = effective_alpha / gain
    gain_flat = gain.expand(B, H, 1, 1).reshape(B * H).float().contiguous()
    effective_alpha_flat = effective_alpha.reshape(B * H).float().contiguous()
    effective_flat = effective_gamma.reshape(B * H).float().contiguous()
    # Trace normalization always uses the Tensor-Core/cuBLAS dispatcher. The
    # legacy scalar all-in-one kernel cannot fuse the radial derivative and
    # duplicates mask traffic, so it remains available only to the legacy
    # non-trace parameterization.
    split_n = (
        N >= _SPLIT_BACKWARD_MIN_SEQUENCE
        and B * H <= _SPLIT_BACKWARD_MAX_SYSTEMS
    )
    with torch.autocast(device_type=U.device.type, enabled=False):
        YtU, PtU, grad_mu_flat = _backward_compact_statistics(
            U_native,
            Y_native,
            P_native,
            split_n=split_n,
            chunk_size=_SPLIT_BACKWARD_CHUNK_SIZE,
            valid_mask=valid_mask,
            heads=H,
        )
        YtU_readout = YtU.to(matmul_dtype).contiguous()
        PtU_readout = PtU.to(matmul_dtype).contiguous()
        grad_effective_gamma = -(
            PtU.to(calc_dtype) * YtU.to(calc_dtype)
        ).sum(dim=(1, 2))
        grad_effective_alpha = grad_effective_gamma / gain_flat.to(calc_dtype)
        grad_gain_flat = -(
            grad_mu_flat
            + effective_alpha_flat.to(calc_dtype) * grad_effective_gamma
        ) / gain_flat.to(calc_dtype).square()
        radial_coefficient = (
            -2.0
            * effective_alpha_flat.to(calc_dtype)
            * grad_effective_alpha.to(calc_dtype)
            / denominator.reshape(B * H).to(calc_dtype)
        )
        direct_grad = try_dual_grad_u_tensorcore(
            P_native,
            Y_native,
            YtU_readout,
            PtU_readout,
            -effective_flat,
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
            direct_grad = torch.bmm(P_m, YtU_readout)
            direct_grad.baddbmm_(Y_m, PtU_readout)
            direct_grad.mul_(
                -effective_gamma.to(matmul_dtype).reshape(B * H, 1, 1)
            )
    # d alpha_eff / dU = -2 * alpha_eff * U / denominator. The
    # low-level compact gradient is with respect to gamma_eff=alpha_eff/g.
    radial_coefficient = (
        -2.0
        * effective_alpha.reshape(B * H).to(calc_dtype)
        * grad_effective_alpha.to(calc_dtype)
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
    grad_alpha_bh = (
        grad_effective_alpha.view(B, H, 1, 1).to(calc_dtype)
        * scale_squared.to(calc_dtype)
    )
    return (
        grad_U.to(U.dtype),
        P.to(grad_output.dtype),
        grad_gain_bh,
        grad_alpha_bh,
    )


class _TraceNormalizedLSSOAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U,
        C,
        gain,
        alpha,
        eye,
        valid_mask,
        eps,
        length_normalize,
        length_reference,
        padding_ratio_hint,
    ):
        Y, effective_alpha, denominator, scale_squared = _trace_statistics_forward(
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
            alpha,
            effective_alpha,
            scale_squared,
            denominator,
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
            alpha,
            effective_alpha,
            scale_squared,
            denominator,
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
                reference, _, _, _ = _trace_statistics_forward(
                    U,
                    C,
                    gain,
                    alpha,
                    eye,
                    valid_mask,
                    eps=ctx.eps,
                    length_normalize=ctx.length_normalize,
                    length_reference=ctx.length_reference,
                    padding_ratio_hint=ctx.padding_ratio_hint,
                )
                inputs = (U, C, gain, alpha)
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
        grad_U, grad_C, grad_gain_bh, grad_alpha_bh = _trace_gain_alpha_backward(
            U,
            Y,
            grad_output,
            gain,
            alpha,
            effective_alpha,
            scale_squared,
            denominator,
            valid_mask,
            ctx.padding_ratio_hint,
        )
        grad_gain = (
            grad_gain_bh.sum(dim=0, keepdim=True)
            if gain.shape[0] == 1
            else grad_gain_bh
        )
        grad_alpha = (
            grad_alpha_bh.sum(dim=0, keepdim=True)
            if alpha.shape[0] == 1
            else grad_alpha_bh
        )
        return (
            grad_U,
            grad_C,
            grad_gain.to(gain.dtype),
            grad_alpha.to(alpha.dtype),
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
        gamma = alpha * mu
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
                alpha,
                eye,
                valid_mask,
                normalization_eps,
                length_normalize,
                length_reference,
                padding_ratio_hint,
            )
        output, _effective_alpha, _denominator, _scale_squared = _trace_statistics_forward(
            U,
            C,
            gain,
            alpha,
            eye,
            valid_mask,
            eps=normalization_eps,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        return output

    mu = gain.reciprocal()
    gamma = alpha * mu
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
        alpha_init: float = DEFAULT_ALPHA_INIT,
        solve_parameterization: str = "gain_alpha",
        alpha_max: float = DEFAULT_ALPHA_MAX,
        no_global: bool = False,
        normalize_u: bool = True,
        basis_normalization: str = "trace",
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
            solve_parameterization=solve_parameterization,
            gain_init=gain_init,
            alpha_init=alpha_init,
            alpha_max=alpha_max,
        )
        self.register_load_state_dict_pre_hook(_legacy_solve_state_dict_pre_hook)
        self.register_load_state_dict_post_hook(_sync_alpha_max_after_load)
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
        token_rms = self.normalize_u and self.basis_normalization == "token_rms"
        trace_basis = self.normalize_u and self.basis_normalization == "trace"
        # Historical token-wise RMS is intentionally PyTorch-only. The
        # maintained CUDA path is reserved for trace-normalized statistics.
        if token_rms:
            U = U * torch.rsqrt(torch.mean(U * U, dim=-1, keepdim=True) + self.eps)
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
