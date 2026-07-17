"""Shared-library lifecycle for the optional MathDx backend."""

from __future__ import annotations

import os
from pathlib import Path

import torch


_loaded = False
_attempted = False
_error: Exception | None = None


def default_library_path() -> Path:
    return Path(__file__).resolve().parents[2] / "build" / "mathdx" / "lib" / "lsso_mathdx.so"


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

    library = Path(
        path
        or os.environ.get("LSSO_MATHDX_LIBRARY", "")
        or default_library_path()
    )
    _attempted = True
    try:
        torch.ops.load_library(str(library))
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
