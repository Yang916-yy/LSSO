from __future__ import annotations

import os
from pathlib import Path

import torch


_LOADED = False
_LOAD_ATTEMPTED = False
_LOAD_ERROR: Exception | None = None


def _default_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "build" / "mathdx" / "lib" / "lsso_mathdx.so"


def load_mathdx_backend(path: str | os.PathLike[str] | None = None) -> bool:
    """Load the optional WSL/Linux MathDx operator library.

    The backend is deliberately not compiled during package import. Build it
    with ``scripts/build_mathdx_backend.sh`` and optionally override the shared
    library path with ``LSSO_MATHDX_LIBRARY``.
    """

    global _LOADED, _LOAD_ATTEMPTED, _LOAD_ERROR
    if _LOADED:
        return True
    if os.environ.get("LSSO_DISABLE_MATHDX", "0").lower() in {"1", "true", "yes"}:
        _LOAD_ATTEMPTED = True
        _LOAD_ERROR = RuntimeError("MathDx backend disabled by LSSO_DISABLE_MATHDX")
        return False
    if _LOAD_ATTEMPTED and path is None:
        return False

    library = Path(
        path
        or os.environ.get("LSSO_MATHDX_LIBRARY", "")
        or _default_library_path()
    )
    _LOAD_ATTEMPTED = True
    try:
        torch.ops.load_library(str(library))
    except Exception as exc:  # optional backend; callers choose fallback policy
        _LOAD_ERROR = exc
        return False
    _LOADED = True
    _LOAD_ERROR = None
    return True


def mathdx_load_error() -> Exception | None:
    return _LOAD_ERROR


def is_mathdx_available() -> bool:
    """Return whether the optional native backend can be loaded."""

    return load_mathdx_backend()


def solve_spd(gram: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve FP32 batched SPD systems with device-side Cholesky/POTRS."""

    if not load_mathdx_backend():
        raise RuntimeError(f"failed to load LSSO MathDx backend: {_LOAD_ERROR}")
    return torch.ops.lsso_mathdx.solve_spd(gram, rhs)


def solve_spd_bpb(
    gram: torch.Tensor,
    rhs: torch.Tensor,
    batches_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Benchmark an explicit 1/2/4-system cuSolverDx CTA schedule."""

    if batches_per_block not in (1, 2, 4):
        raise ValueError("batches_per_block must be 1, 2, or 4")
    if not load_mathdx_backend():
        raise RuntimeError(f"failed to load LSSO MathDx backend: {_LOAD_ERROR}")
    return torch.ops.lsso_mathdx.solve_spd_bpb(
        gram, rhs, batches_per_block
    )


def solve_spd_or_torch(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Use cuSolverDx when supported, otherwise preserve the PyTorch path."""

    eligible = (
        gram.is_cuda
        and rhs.is_cuda
        and gram.dtype == torch.float32
        and rhs.dtype == torch.float32
        and gram.ndim == 3
        and gram.shape[-1] in (16, 32)
    )
    if eligible and load_mathdx_backend():
        solution, _info = torch.ops.lsso_mathdx.solve_spd(
            gram.contiguous(), rhs.contiguous()
        )
        return solution
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


def stats_solve_spd(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused mixed-input ``U.T@U``, ``U.T@C``, and FP32 SPD solve."""

    if not load_mathdx_backend():
        raise RuntimeError(f"failed to load LSSO MathDx backend: {_LOAD_ERROR}")
    return torch.ops.lsso_mathdx.stats_solve_spd(
        u.contiguous(), c.contiguous(), alpha.contiguous()
    )


def stats_solve_readout(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    inv_mu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse compact statistics, FP32 solve, and the Woodbury readout."""

    if not load_mathdx_backend():
        raise RuntimeError(f"failed to load LSSO MathDx backend: {_LOAD_ERROR}")
    return torch.ops.lsso_mathdx.stats_solve_readout(
        u.contiguous(), c.contiguous(), alpha.contiguous(), inv_mu.contiguous()
    )


def try_stats_solve_spd(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    *,
    max_sequence: int = 512,
) -> torch.Tensor | None:
    """Return fused stats/solve output when its measured fast-path applies."""

    # This low-level operator intentionally has no dispatcher autograd kernel.
    # The custom LSSO Function calls it under no-grad and supplies the analytic
    # backward; direct differentiable reference calls must remain in PyTorch.
    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None

    eligible = (
        u.is_cuda
        and c.is_cuda
        and alpha.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and alpha.dtype == torch.float32
        and u.ndim == 3
        and c.ndim == 3
        and alpha.ndim == 1
        and u.shape[2] in (16, 32)
        and u.shape[1] <= max_sequence
        and c.shape[2] <= 64
    )
    if not eligible or not load_mathdx_backend():
        return None
    solution, _info = torch.ops.lsso_mathdx.stats_solve_spd(
        u.contiguous(), c.contiguous(), alpha.contiguous()
    )
    return solution


def try_stats_solve_readout(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    inv_mu: torch.Tensor,
    *,
    max_sequence: int = 512,
) -> torch.Tensor | None:
    """Use the one-kernel Woodbury forward path when its tile shapes apply.

    U/C stay in BF16 or FP16 in global memory. Statistics, Cholesky and K are
    FP32, while the final ``(C - alpha * U@K) * inv_mu`` write uses C's dtype.
    """

    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    eligible = (
        u.is_cuda
        and c.is_cuda
        and alpha.is_cuda
        and inv_mu.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and alpha.dtype == torch.float32
        and inv_mu.dtype == torch.float32
        and u.ndim == 3
        and c.ndim == 3
        and alpha.ndim == 1
        and inv_mu.ndim == 1
        and u.shape[:2] == c.shape[:2]
        and alpha.shape[0] == inv_mu.shape[0] == u.shape[0]
        and u.shape[2] in (16, 32)
        and u.shape[1] <= max_sequence
        and c.shape[2] <= 64
    )
    if not eligible or not load_mathdx_backend():
        return None
    output, _info = torch.ops.lsso_mathdx.stats_solve_readout(
        u.contiguous(), c.contiguous(), alpha.contiguous(), inv_mu.contiguous()
    )
    return output


def try_masked_stats_solve_spd(
    u: torch.Tensor,
    c: torch.Tensor,
    valid_mask: torch.Tensor,
    length_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    max_sequence: int | None = None,
    padding_ratio_hint: float | None = None,
    native_padding_threshold: float = 0.75,
) -> torch.Tensor | None:
    """Fuse masked/scaled statistics and solve without reading padded U/C.

    ``u`` and ``c`` retain their native training dtype. The CUDA kernel checks
    ``valid_mask`` before loading either tensor, converts valid values to FP32
    in shared memory, and accumulates/solves the compact system in FP32. Long
    sequences are reduced in fixed-size token tiles inside each CTA, while the
    launch grid assigns one independent CTA to every ``(batch, head)`` system.

    ``max_sequence`` is an optional caller-side policy cap retained for
    compatibility. The native kernel itself has no 512-token or 64-system
    limit. ``padding_ratio_hint`` enables a zero-synchronization hybrid policy:
    when supplied by a CPU-side collator, low-padding inputs return ``None``
    and use the cuBLAS fallback. The backend never reads a CUDA mask back to
    the host to make this decision.
    """

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

    if torch.compiler.is_compiling():
        return None
    eligible = (
        u.is_cuda
        and c.is_cuda
        and valid_mask.is_cuda
        and length_scale.is_cuda
        and alpha.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and valid_mask.dtype == torch.bool
        and length_scale.dtype == torch.float32
        and alpha.dtype == torch.float32
        and u.ndim == 4
        and c.ndim == 4
        and valid_mask.ndim == 2
        and length_scale.ndim == 1
        and alpha.ndim == 1
        and u.shape[:3] == c.shape[:3]
        and valid_mask.shape == (u.shape[0], u.shape[2])
        and length_scale.shape[0] == u.shape[0]
        and alpha.shape[0] == u.shape[0] * u.shape[1]
        and u.shape[3] in (16, 32)
        and (max_sequence is None or u.shape[2] <= max_sequence)
        and c.shape[3] <= 64
    )
    if not eligible or not load_mathdx_backend():
        return None
    solution, _info = torch.ops.lsso_mathdx.masked_stats_solve_spd(
        u.contiguous(),
        c.contiguous(),
        valid_mask.contiguous(),
        length_scale.contiguous(),
        alpha.contiguous(),
    )
    return solution


def try_masked_stats_solve_readout(
    u: torch.Tensor,
    c: torch.Tensor,
    valid_mask: torch.Tensor,
    length_scale: torch.Tensor,
    alpha: torch.Tensor,
    inv_mu: torch.Tensor,
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
        and inv_mu.is_cuda
        and u.dtype == c.dtype
        and u.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and valid_mask.dtype == torch.bool
        and length_scale.dtype == torch.float32
        and alpha.dtype == torch.float32
        and inv_mu.dtype == torch.float32
        and u.ndim == 4
        and c.ndim == 4
        and valid_mask.ndim == 2
        and length_scale.ndim == 1
        and alpha.ndim == inv_mu.ndim == 1
        and u.shape[:3] == c.shape[:3]
        and valid_mask.shape == (u.shape[0], u.shape[2])
        and length_scale.shape[0] == u.shape[0]
        and alpha.shape[0] == inv_mu.shape[0] == u.shape[0] * u.shape[1]
        and u.shape[3] in (16, 32)
        and (max_sequence is None or u.shape[2] <= max_sequence)
        and c.shape[3] <= 64
    )
    if not eligible or not load_mathdx_backend():
        return None
    output, _info = torch.ops.lsso_mathdx.masked_stats_solve_readout(
        u.contiguous(),
        c.contiguous(),
        valid_mask.contiguous(),
        length_scale.contiguous(),
        alpha.contiguous(),
        inv_mu.contiguous(),
    )
    return output


def try_lsso_backward_fused(
    u: torch.Tensor,
    y: torch.Tensor,
    p: torch.Tensor,
    gamma_bh: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    length_scale: torch.Tensor | None = None,
    max_sequence: int | None = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fuse compact backward statistics, parameter reductions, and grad-U."""
    masked = valid_mask is not None
    eligible = (
        not torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and u.is_cuda
        and y.is_cuda
        and p.is_cuda
        and gamma_bh.is_cuda
        and u.dtype == y.dtype == p.dtype
        and u.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and u.ndim == y.ndim == p.ndim == 4
        and u.shape[:3] == y.shape[:3]
        and y.shape == p.shape
        and u.shape[3] in (16, 32)
        and y.shape[3] <= 64
        and (max_sequence is None or u.shape[2] <= max_sequence)
        and gamma_bh.dtype == torch.float32
        and gamma_bh.shape == (u.shape[0] * u.shape[1],)
    )
    if masked:
        eligible = eligible and (
            valid_mask is not None
            and length_scale is not None
            and valid_mask.is_cuda
            and length_scale.is_cuda
            and valid_mask.dtype == torch.bool
            and length_scale.dtype == torch.float32
            and valid_mask.shape == (u.shape[0], u.shape[2])
            and length_scale.shape == (u.shape[0],)
        )
    if not eligible or not load_mathdx_backend():
        return None
    if not masked:
        valid_mask = torch.empty(0, device=u.device, dtype=torch.bool)
        length_scale = torch.empty(0, device=u.device, dtype=torch.float32)
    return torch.ops.lsso_mathdx.lsso_backward_fused(
        u.contiguous(),
        y.contiguous(),
        p.contiguous(),
        gamma_bh.contiguous(),
        valid_mask.contiguous(),
        length_scale.contiguous(),
    )


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
        and input.is_contiguous()
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


class _NormalizeBasisFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, eps: float, length_scale: float):
        output, inv_rms = torch.ops.lsso_mathdx.normalize_basis(
            input, eps, length_scale
        )
        ctx.save_for_backward(input, inv_rms)
        ctx.length_scale = length_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, inv_rms = ctx.saved_tensors
        grad_input = torch.ops.lsso_mathdx.normalize_basis_backward(
            grad_output.contiguous(), input, inv_rms, ctx.length_scale
        )
        return grad_input, None, None


class _NormalizeRankRotaryFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        eps: float,
        length_scale: float,
    ):
        output, inv_rms = torch.ops.lsso_mathdx.normalize_rank_rotary(
            input, cos, sin, eps, length_scale
        )
        ctx.save_for_backward(input, inv_rms, cos, sin)
        ctx.length_scale = length_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, inv_rms, cos, sin = ctx.saved_tensors
        grad_input = torch.ops.lsso_mathdx.normalize_rank_rotary_backward(
            grad_output.contiguous(), input, inv_rms, cos, sin, ctx.length_scale
        )
        return grad_input, None, None, None, None


def try_prepare_basis(
    input: torch.Tensor,
    *,
    eps: float,
    length_scale: float,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Fuse RMS normalization, length scaling, and optional rank rotation."""

    if torch.compiler.is_compiling():
        return None
    eligible = (
        input.is_cuda
        and input.is_contiguous()
        and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and input.ndim == 4
        and 0 < input.shape[-1] <= 1024
    )
    rotary = cos is not None or sin is not None
    if rotary:
        eligible = (
            eligible
            and cos is not None
            and sin is not None
            and cos.is_cuda
            and sin.is_cuda
            and cos.is_contiguous()
            and sin.is_contiguous()
            and input.dtype == cos.dtype == sin.dtype
            and input.shape[-1] % 2 == 0
        )
    if not eligible or not load_mathdx_backend():
        return None
    if rotary:
        return _NormalizeRankRotaryFunction.apply(
            input, cos, sin, float(eps), float(length_scale)
        )
    return _NormalizeBasisFunction.apply(input, float(eps), float(length_scale))
