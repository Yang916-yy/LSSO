from __future__ import annotations

from collections import Counter
import os

import torch

from .backends import _loader


_FUSED_BASE_RHS_WIDTH = 64
_FUSED_WIDE_RHS_WIDTH = 192
_FUSED_WIDE_MAX_SYSTEMS = 128
_FUSED_WIDE_MAX_SEQUENCE = 256
_MATHDX_SOLVE_RANK_BUCKETS = (16, 32, 48, 64)
_MATHDX_FUSED_TRACE_RANKS = (16, 32, 48, 64)
_PATH_COUNTERS: Counter[str] = Counter()


def _path_counters_enabled() -> bool:
    return os.environ.get("LSSO_PATH_COUNTERS", "0").lower() in {
        "1", "on", "true", "yes"
    }


def record_mathdx_path(path: str) -> None:
    """Record an opt-in dispatcher event without synchronizing CUDA.

    Counters are process-local diagnostics. They are disabled by default so
    the training hot path does not pay Python bookkeeping overhead.
    """

    if _path_counters_enabled():
        _PATH_COUNTERS[path] += 1


def reset_mathdx_path_counters() -> None:
    _PATH_COUNTERS.clear()


def get_mathdx_path_counters() -> dict[str, int]:
    return dict(_PATH_COUNTERS)


def _check_mathdx_info(info: torch.Tensor, path: str) -> None:
    """Optionally synchronize and fail on a cuSolverDx POTRF error.

    Normal training intentionally avoids reading ``info`` back to the host.
    ``LSSO_MATHDX_DEBUG_INFO=1`` turns the check on for smoke/debug runs.
    """

    if os.environ.get("LSSO_MATHDX_DEBUG_INFO", "0").lower() not in {
        "1", "on", "true", "yes"
    }:
        return
    failed = torch.nonzero(info, as_tuple=False).flatten().cpu().tolist()
    record_mathdx_path("debug.info_check")
    if failed:
        values = info.flatten()[failed].cpu().tolist()
        raise RuntimeError(
            f"{path}: cuSolverDx POTRF failed for systems {failed} "
            f"with info values {values}"
        )


def _fused_rhs_shape_eligible(
    systems: int,
    sequence: int,
    rhs_width: int,
) -> bool:
    """Shape policy for the CTA-local stats/solve/readout kernels.

    The CUDA implementation can iterate over arbitrary 32-column RHS tiles,
    but wide systems become cuBLAS-favorable once either the system grid or K
    dimension is large. Keep the common <=64 path unchanged and only use the
    96--192 fast path in its measured short/moderate-batch regime.
    """
    if rhs_width <= _FUSED_BASE_RHS_WIDTH:
        return True
    return (
        rhs_width <= _FUSED_WIDE_RHS_WIDTH
        and systems <= _FUSED_WIDE_MAX_SYSTEMS
        and sequence <= _FUSED_WIDE_MAX_SEQUENCE
    )


def _mathdx_solve_rank_bucket(rank: int) -> int | None:
    """Return the smallest native solve rank that contains ``rank``."""
    return next((bucket for bucket in _MATHDX_SOLVE_RANK_BUCKETS if rank <= bucket), None)


def load_mathdx_backend(path=None) -> bool:
    """Load the optional WSL/Linux MathDx operator library.

    The backend is deliberately not compiled during package import. Build it
    with ``tools/build_mathdx_backend.sh`` and optionally override the shared
    library path with ``LSSO_MATHDX_LIBRARY``.
    """

    return _loader.load(path)


def mathdx_load_error() -> Exception | None:
    return _loader.load_error()


def is_mathdx_available() -> bool:
    """Return whether the optional native backend can be loaded."""

    return _loader.is_available()


def solve_spd(gram: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve FP32 batched SPD systems with device-side Cholesky/POTRS."""

    if not load_mathdx_backend():
        raise RuntimeError(f"failed to load LSSO MathDx backend: {_loader.load_error()}")
    return torch.ops.lsso_mathdx.solve_spd(gram, rhs)


def solve_spd_or_torch(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Use an exact/native rank bucket, otherwise preserve the Torch path.

    Padding ``G`` to ``diag(G, I)`` and ``rhs`` to ``[rhs; 0]`` leaves the
    leading solution exactly equal to ``G^{-1} rhs``. This lets arbitrary
    ranks up to 64 reuse four compiled MathDx solvers without approximation.
    """

    rank = gram.shape[-1] if gram.ndim == 3 else -1
    bucket = _mathdx_solve_rank_bucket(rank)
    eligible = (
        gram.is_cuda
        and rhs.is_cuda
        and gram.dtype == torch.float32
        and rhs.dtype == torch.float32
        and gram.ndim == 3
        and rhs.ndim == 3
        and gram.shape[0] == rhs.shape[0]
        and gram.shape[1] == rank
        and rhs.shape[1] == rank
        and bucket is not None
    )
    if eligible and load_mathdx_backend():
        solution, _info = torch.ops.lsso_mathdx.solve_spd(
            gram.contiguous(), rhs.contiguous()
        )
        _check_mathdx_info(_info, "solve_spd.bucket")
        record_mathdx_path(f"solve.bucket_r{bucket}")
        return solution
    record_mathdx_path("solve.torch_fallback")
    return torch.linalg.solve_ex(gram, rhs, check_errors=False)[0]


class _SolveSPDFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        solution = solve_spd_or_torch(gram, rhs)
        ctx.save_for_backward(gram, solution)
        return solution

    @staticmethod
    def backward(ctx, grad_solution: torch.Tensor):
        gram, solution = ctx.saved_tensors
        grad_solution = grad_solution.contiguous()
        # The native operator intentionally has no dispatcher-level autograd
        # kernel. Use PyTorch for higher-order differentiation and MathDx for
        # the normal first-order training path.
        if torch.is_grad_enabled():
            grad_rhs = torch.linalg.solve_ex(
                gram.transpose(-1, -2), grad_solution, check_errors=False
            )[0]
        else:
            grad_rhs = solve_spd_or_torch(gram.transpose(-1, -2), grad_solution)
        grad_gram = -torch.bmm(grad_rhs, solution.transpose(-1, -2))
        return grad_gram, grad_rhs


def solve_spd_autograd(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Autograd-aware exact SPD solve with a transparent Torch fallback."""

    return _SolveSPDFunction.apply(gram, rhs)


def try_masked_stats_solve_readout(
    u: torch.Tensor,
    c: torch.Tensor,
    valid_mask: torch.Tensor,
    length_scale: torch.Tensor,
    alpha: torch.Tensor,
    gain: torch.Tensor,
    *,
    max_sequence: int | None = None,
    padding_ratio_hint: float | None = None,
    native_padding_threshold: float = 0.75,
) -> torch.Tensor | None:
    """Fused masked statistics, solve, and padding-safe Woodbury readout."""
    if padding_ratio_hint is not None:
        if not 0.0 <= padding_ratio_hint <= 1.0:
            raise ValueError(
                f"padding_ratio_hint must be in [0, 1], got {padding_ratio_hint}"
            )
        if not 0.0 <= native_padding_threshold <= 1.0:
            raise ValueError(
                "native_padding_threshold must be in [0, 1], got "
                f"{native_padding_threshold}"
            )
        if padding_ratio_hint < native_padding_threshold:
            return None
    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    eligible = (
        u.is_cuda
        and c.is_cuda
        and valid_mask.is_cuda
        and length_scale.is_cuda
        and alpha.is_cuda
        and gain.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16)
        and valid_mask.dtype == torch.bool
        and length_scale.dtype == torch.float32
        and alpha.dtype == torch.float32
        and gain.dtype == torch.float32
        and u.ndim == 4
        and c.ndim == 4
        and valid_mask.ndim == 2
        and length_scale.ndim == 1
        and alpha.ndim == gain.ndim == 1
        and u.shape[:3] == c.shape[:3]
        and valid_mask.shape == (u.shape[0], u.shape[2])
        and length_scale.shape[0] == u.shape[0]
        and alpha.shape[0] == gain.shape[0] == u.shape[0] * u.shape[1]
        and u.shape[3] in _MATHDX_FUSED_TRACE_RANKS
        and (max_sequence is None or u.shape[2] <= max_sequence)
        and _fused_rhs_shape_eligible(
            u.shape[0] * u.shape[1], u.shape[2], c.shape[3]
        )
    )
    if not eligible or not load_mathdx_backend():
        return None
    output, _info = torch.ops.lsso_mathdx.masked_stats_solve_readout(
        u.contiguous(),
        c,
        valid_mask.contiguous(),
        length_scale.contiguous(),
        alpha.contiguous(),
        gain.contiguous(),
    )
    _check_mathdx_info(_info, "masked_stats_solve_readout")
    record_mathdx_path("forward.masked_stats_solve_readout")
    return output


def try_effective_stats_solve_readout(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    gain: torch.Tensor,
) -> torch.Tensor | None:
    """Fuse an unmasked solve/readout with an already effective strength.

    This is primarily the Trace backward adjoint path. Reusing the mask-aware
    CTA kernel with null mask/scale pointers gives rank-48/64 coverage without
    allocating or reading an all-ones mask.
    """

    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    eligible = (
        u.is_cuda
        and c.is_cuda
        and alpha.is_cuda
        and gain.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16)
        and alpha.dtype == gain.dtype == torch.float32
        and u.ndim == c.ndim == 4
        and u.shape[:3] == c.shape[:3]
        and u.shape[3] in _MATHDX_FUSED_TRACE_RANKS
        and alpha.shape == gain.shape == (u.shape[0] * u.shape[1],)
        and _fused_rhs_shape_eligible(
            u.shape[0] * u.shape[1], u.shape[2], c.shape[3]
        )
    )
    if not eligible or not load_mathdx_backend():
        return None
    empty_mask = torch.empty(0, device=u.device, dtype=torch.bool)
    empty_scale = torch.empty(0, device=u.device, dtype=torch.float32)
    output, _info = torch.ops.lsso_mathdx.masked_stats_solve_readout(
        u.contiguous(), c, empty_mask, empty_scale,
        alpha.contiguous(), gain.contiguous()
    )
    _check_mathdx_info(_info, "effective_stats_solve_readout")
    record_mathdx_path("backward.adjoint_native")
    return output


def try_masked_trace_stats_solve_readout(
    u: torch.Tensor,
    c: torch.Tensor,
    valid_mask: torch.Tensor,
    alpha: torch.Tensor,
    gain: torch.Tensor,
    *,
    normalization_eps: float,
    length_reference: float,
    length_normalize: bool,
    padding_ratio_hint: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Skip padding while deriving trace strength, solving, and reading out."""

    mode = os.environ.get("LSSO_MATHDX_MASKED_TRACE", "auto").lower()
    if mode in {"0", "off", "false", "no"}:
        return None
    if padding_ratio_hint is not None and not 0.0 <= padding_ratio_hint <= 1.0:
        raise ValueError(
            f"padding_ratio_hint must be in [0, 1], got {padding_ratio_hint}"
        )
    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    eligible = (
        u.is_cuda
        and c.is_cuda
        and valid_mask.is_cuda
        and alpha.is_cuda
        and gain.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16)
        and valid_mask.dtype == torch.bool
        and alpha.dtype == gain.dtype == torch.float32
        and u.ndim == c.ndim == 4
        and valid_mask.ndim == 2
        and alpha.ndim == gain.ndim == 1
        and u.shape[:3] == c.shape[:3]
        and valid_mask.shape == (u.shape[0], u.shape[2])
        and alpha.shape[0] == gain.shape[0] == u.shape[0] * u.shape[1]
        and u.shape[3] in _MATHDX_FUSED_TRACE_RANKS
        and _fused_rhs_shape_eligible(
            u.shape[0] * u.shape[1], u.shape[2], c.shape[3]
        )
    )
    if not eligible or not load_mathdx_backend():
        return None
    output, _info, effective, denominator, scale_squared = (
        torch.ops.lsso_mathdx.masked_trace_stats_solve_readout(
            u.contiguous(),
            c,
            valid_mask.contiguous(),
            alpha.contiguous(),
            gain.contiguous(),
            float(normalization_eps),
            float(length_reference),
            bool(length_normalize),
        )
    )
    _check_mathdx_info(_info, "masked_trace_stats_solve_readout")
    record_mathdx_path("forward.trace_masked_cta")
    B, H = u.shape[:2]
    compact_shape = (B, H, 1, 1)
    return (
        output,
        effective.view(compact_shape),
        denominator.view(compact_shape),
        scale_squared.view(compact_shape),
    )


def try_trace_stats_solve_readout(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    gain: torch.Tensor,
    *,
    normalization_eps: float,
    length_reference: float,
    length_normalize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fuse unmasked trace statistics, solve, and Woodbury readout.

    The CUDA entry point shares the masked kernel implementation but receives
    null mask/length pointers. It therefore avoids both mask traffic and an
    all-ones mask allocation while retaining the same single-kernel contract.
    """

    mode = os.environ.get("LSSO_MATHDX_TRACE", "auto").lower()
    if mode in {"0", "off", "false", "no"}:
        return None
    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    eligible = (
        u.is_cuda
        and c.is_cuda
        and alpha.is_cuda
        and gain.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16)
        and alpha.dtype == gain.dtype == torch.float32
        and u.ndim == c.ndim == 4
        and alpha.ndim == gain.ndim == 1
        and u.shape[:3] == c.shape[:3]
        and alpha.shape[0] == gain.shape[0] == u.shape[0] * u.shape[1]
        and u.shape[3] in _MATHDX_FUSED_TRACE_RANKS
        and _fused_rhs_shape_eligible(
            u.shape[0] * u.shape[1], u.shape[2], c.shape[3]
        )
    )
    if not eligible or not load_mathdx_backend():
        return None
    empty_mask = torch.empty(0, device=u.device, dtype=torch.bool)
    empty_scale = torch.empty(0, device=u.device, dtype=torch.float32)
    output, _info, effective, denominator, scale_squared = (
        torch.ops.lsso_mathdx.masked_trace_stats_solve_readout(
            u.contiguous(),
            c,
            empty_mask,
            alpha.contiguous(),
            gain.contiguous(),
            float(normalization_eps),
            float(length_reference),
            bool(length_normalize),
        )
    )
    _check_mathdx_info(_info, "trace_stats_solve_readout")
    record_mathdx_path("forward.trace_unmasked_cta")
    B, H = u.shape[:2]
    compact_shape = (B, H, 1, 1)
    return (
        output,
        effective.view(compact_shape),
        denominator.view(compact_shape),
        scale_squared.view(compact_shape),
    )


def try_dual_backward_statistics_tensorcore(
    u: torch.Tensor,
    y: torch.Tensor,
    p: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    heads: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Form Y^T U and P^T U while loading each U tile only once."""

    mode = os.environ.get("LSSO_MATHDX_DUAL_STATS", "auto").lower()
    if mode in {"0", "off", "false", "no"}:
        return None
    if u.ndim not in (3, 4) or y.ndim != u.ndim or p.ndim != u.ndim:
        return None
    forced = mode in {"1", "on", "true", "yes", "force"}
    logical_eligible = (
        u.ndim == y.ndim == p.ndim == 3
        and y.shape == p.shape
        and u.shape[:2] == y.shape[:2]
        and heads > 0
        and u.shape[0] % heads == 0
    ) or (
        u.ndim == y.ndim == p.ndim == 4
        and y.shape == p.shape
        and u.shape[:3] == y.shape[:3]
        and heads == u.shape[1]
    )
    batches = u.shape[0] // heads if u.ndim == 3 else u.shape[0]
    sequence = u.shape[1] if u.ndim == 3 else u.shape[2]
    systems = u.shape[0] if u.ndim == 3 else u.shape[0] * u.shape[1]
    rank = u.shape[2] if u.ndim == 3 else u.shape[3]
    mask_eligible = valid_mask is None or (
        valid_mask.is_cuda
        and valid_mask.dtype == torch.bool
        and valid_mask.ndim == 2
        and heads > 0
        and valid_mask.shape == (batches, sequence)
    )
    eligible = (
        not torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and u.is_cuda
        and y.is_cuda
        and p.is_cuda
        and u.dtype == y.dtype == p.dtype
        and u.dtype in (torch.float16, torch.bfloat16)
        and logical_eligible
        and rank in (16, 32, 48, 64)
        and mask_eligible
        and (forced or (systems <= 128 and sequence <= 512))
    )
    if not eligible or not load_mathdx_backend():
        return None
    mask_tensor = valid_mask if valid_mask is not None else torch.empty(
        0, device=u.device, dtype=torch.bool
    )
    result = torch.ops.lsso_mathdx.dual_backward_statistics_tensorcore(
        u, y, p, mask_tensor.contiguous(), int(heads)
    )
    record_mathdx_path("backward.dual_statistics")
    return result


def try_dual_grad_u_tensorcore(
    p: torch.Tensor,
    y: torch.Tensor,
    ytu: torch.Tensor,
    ptu: torch.Tensor,
    coefficient: torch.Tensor,
    *,
    radial_u: torch.Tensor | None = None,
    radial_coefficient: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    heads: int = 1,
) -> torch.Tensor | None:
    """Compute ``coefficient * (P@YtU + Y@PtU)`` with one MMA epilogue.

    Compact statistics remain FP32 until the caller explicitly casts them to
    the BF16/FP16 Tensor-Core input dtype. The native kernel accepts two source
    pairs without materializing ``cat(P, Y)`` or a second output-sized tensor.
    """

    mode = os.environ.get("LSSO_MATHDX_DUAL_GRAD_U", "auto").lower()
    if mode in {"0", "off", "false", "no"}:
        return None
    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    if p.ndim not in (3, 4) or y.ndim != p.ndim or ytu.ndim != 3 or ptu.ndim != 3:
        return None
    logical_eligible = (
        p.ndim == y.ndim == 3
        and p.shape == y.shape
        and heads > 0
        and p.shape[0] % heads == 0
    ) or (
        p.ndim == y.ndim == 4
        and p.shape == y.shape
        and heads == p.shape[1]
    )
    batches = p.shape[0] // heads if p.ndim == 3 else p.shape[0]
    systems = p.shape[0] if p.ndim == 3 else p.shape[0] * p.shape[1]
    sequence = p.shape[1] if p.ndim == 3 else p.shape[2]
    width = p.shape[2] if p.ndim == 3 else p.shape[3]
    add_radial = radial_u is not None or radial_coefficient is not None
    expected_radial_shape = (
        (systems, sequence, ytu.shape[2])
        if p.ndim == 3
        else (p.shape[0], p.shape[1], sequence, ytu.shape[2])
    )
    radial_eligible = not add_radial or (
        radial_u is not None
        and radial_coefficient is not None
        and radial_u.is_cuda
        and radial_coefficient.is_cuda
        and radial_u.dtype == p.dtype
        and radial_coefficient.dtype == torch.float32
        and radial_u.shape == expected_radial_shape
        and radial_coefficient.shape == (systems,)
        and radial_coefficient.is_contiguous()
    )
    mask_eligible = valid_mask is None or (
        valid_mask.is_cuda
        and valid_mask.dtype == torch.bool
        and valid_mask.ndim == 2
        and heads > 0
        and valid_mask.shape == (batches, sequence)
    )
    eligible = (
        p.is_cuda
        and y.is_cuda
        and ytu.is_cuda
        and ptu.is_cuda
        and coefficient.is_cuda
        and p.dtype == y.dtype == ytu.dtype == ptu.dtype
        and p.dtype in (torch.float16, torch.bfloat16)
        and coefficient.dtype == torch.float32
        and logical_eligible
        and ytu.ndim == ptu.ndim == 3
        and ytu.shape == ptu.shape
        and systems == ytu.shape[0] == coefficient.shape[0]
        and width == ytu.shape[1]
        and ytu.shape[2] in (16, 32, 48, 64)
        and width % 16 == 0
        and coefficient.ndim == 1
        and ytu.is_contiguous()
        and ptu.is_contiguous()
        and coefficient.is_contiguous()
        and radial_eligible
        and mask_eligible
    )
    if not eligible or not load_mathdx_backend():
        return None
    if radial_u is None:
        radial_u = torch.empty(0, device=p.device, dtype=p.dtype)
        radial_coefficient = torch.empty(0, device=p.device, dtype=torch.float32)
    mask_tensor = valid_mask if valid_mask is not None else torch.empty(
        0, device=p.device, dtype=torch.bool
    )
    result = torch.ops.lsso_mathdx.dual_grad_u_tensorcore(
        p, y, ytu, ptu, coefficient, radial_u, radial_coefficient,
        mask_tensor.contiguous(), int(heads)
    )
    record_mathdx_path(
        "backward.dual_grad_u_radial" if add_radial
        else "backward.dual_grad_u"
    )
    return result


class _RankRotaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(cos, sin)
        return torch.ops.lsso_mathdx.rank_rotary(input, cos, sin, False)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cos, sin = ctx.saved_tensors
        grad_input = torch.ops.lsso_mathdx.rank_rotary(
            grad_output.contiguous(), cos, sin, True
        )
        return grad_input, None, None


def try_rank_rotary(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor | None:
    """Apply the fused CUDA rank rotation when the native backend is usable."""

    if torch.compiler.is_compiling():
        return None
    eligible = (
        input.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and cos.is_contiguous()
        and sin.is_contiguous()
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and input.dtype == cos.dtype == sin.dtype
        and input.ndim == 4
        and input.shape[-1] % 2 == 0
    )
    if not eligible or not load_mathdx_backend():
        return None
    return _RankRotaryFunction.apply(input, cos, sin)


def try_rank_rotary_inverse(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor | None:
    """Apply the inverse orthogonal rank rotation without autograd wrapping."""

    eligible = (
        not torch.compiler.is_compiling()
        and input.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and input.dtype == cos.dtype == sin.dtype
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and input.ndim == 4
        and input.shape[-1] % 2 == 0
        and cos.is_contiguous()
        and sin.is_contiguous()
    )
    if not eligible or not load_mathdx_backend():
        return None
    return torch.ops.lsso_mathdx.rank_rotary(input, cos, sin, True)
