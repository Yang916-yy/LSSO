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
    """Fused FP32 ``U.T@U``, ``U.T@C``, Cholesky, and multi-RHS solve."""

    if not load_mathdx_backend():
        raise RuntimeError(f"failed to load LSSO MathDx backend: {_LOAD_ERROR}")
    return torch.ops.lsso_mathdx.stats_solve_spd(
        u.contiguous(), c.contiguous(), alpha.contiguous()
    )


def try_stats_solve_spd(
    u: torch.Tensor,
    c: torch.Tensor,
    alpha: torch.Tensor,
    *,
    max_sequence: int = 512,
) -> torch.Tensor | None:
    """Return fused stats/solve output when its measured fast-path applies."""

    eligible = (
        u.is_cuda
        and c.is_cuda
        and alpha.is_cuda
        and u.dtype == torch.float32
        and c.dtype == torch.float32
        and alpha.dtype == torch.float32
        and u.ndim == 3
        and c.ndim == 3
        and alpha.ndim == 1
        and u.shape[2] in (16, 32)
        and u.shape[1] <= max_sequence
        and u.shape[0] <= 64
        and c.shape[2] <= 64
    )
    if not eligible or not load_mathdx_backend():
        return None
    solution, _info = torch.ops.lsso_mathdx.stats_solve_spd(
        u.contiguous(), c.contiguous(), alpha.contiguous()
    )
    return solution
