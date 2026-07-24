"""Shared-library lifecycle for the optional MathDx backend."""

from __future__ import annotations

import os
import importlib
from pathlib import Path

import torch


_loaded = False
_attempted = False
_error: Exception | None = None

# Increment only when an operator schema, tensor contract, or numerical
# contract changes incompatibly. Kernel-internal scheduling is not ABI.
MATHDX_BACKEND_ABI = 2


def default_library_path() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "mathdx" / "lib" / "lsso_mathdx.so"


def packaged_library_path() -> Path | None:
    """Return a compatible optional runtime-wheel library, if installed."""

    try:
        runtime = importlib.import_module("lsso_mathdx_runtime")
    except ImportError:
        return None

    runtime_abi = int(getattr(runtime, "BACKEND_ABI", -1))
    runtime_torch = str(getattr(runtime, "TORCH_VERSION", ""))
    runtime_cuda = str(getattr(runtime, "CUDA_VERSION", ""))
    torch_version = torch.__version__.split("+")[0]
    torch_cuda = torch.version.cuda or ""
    if runtime_abi != MATHDX_BACKEND_ABI:
        raise RuntimeError(
            f"LSSO MathDx ABI mismatch: package expects {MATHDX_BACKEND_ABI}, "
            f"runtime provides {runtime_abi}"
        )
    if runtime_torch != torch_version:
        raise RuntimeError(
            f"LSSO MathDx runtime was built for torch {runtime_torch}, "
            f"but torch {torch_version} is installed"
        )
    if runtime_cuda != torch_cuda:
        raise RuntimeError(
            f"LSSO MathDx runtime was built for CUDA {runtime_cuda}, "
            f"but this PyTorch uses CUDA {torch_cuda or 'none'}"
        )
    return Path(runtime.library_path())


def _candidate_library(path: str | os.PathLike[str] | None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("LSSO_MATHDX_LIBRARY", "")
    if override:
        return Path(override)
    development = default_library_path()
    if development.is_file():
        return development
    packaged = packaged_library_path()
    return packaged if packaged is not None else development


def load(path: str | os.PathLike[str] | None = None) -> bool:
    """Load the backend once without compiling or failing package import."""

    global _loaded, _attempted, _error
    if _loaded:
        return True
    if os.environ.get("LSSO_DISABLE_MATHDX", "0").lower() in {"1", "true", "yes"}:
        _attempted = True
        _error = RuntimeError("MathDx backend disabled by LSSO_DISABLE_MATHDX")
        return False
    if _attempted and path is None:
        return False

    _attempted = True
    try:
        library = _candidate_library(path)
        torch.ops.load_library(str(library))
        binary_abi = int(torch.ops.lsso_mathdx.backend_abi())
        if binary_abi != MATHDX_BACKEND_ABI:
            raise RuntimeError(
                f"LSSO MathDx binary ABI {binary_abi} does not match "
                f"Python ABI {MATHDX_BACKEND_ABI}"
            )
    except Exception as exc:  # optional backend; callers choose fallback policy
        _error = exc
        return False
    _loaded = True
    _error = None
    return True


def load_error() -> Exception | None:
    return _error


def is_available() -> bool:
    return load()
