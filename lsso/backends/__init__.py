"""Optional execution backends.

Backends must not change the mathematical API.  They may decline unsupported
shapes and let the operator dispatch to the portable PyTorch implementation.
"""

from os import PathLike

from ._loader import MATHDX_BACKEND_ABI


def load(path: str | PathLike[str] | None = None) -> bool:
    """Load the optional MathDx backend without importing it eagerly."""

    from .mathdx import load as load_mathdx

    return load_mathdx(path)


def is_available() -> bool:
    """Return whether the optional MathDx backend is available."""

    from .mathdx import is_available as is_mathdx_available

    return is_mathdx_available()


def load_error() -> Exception | None:
    """Return a MathDx load error without making package import backend-hard."""

    from .mathdx import load_error as mathdx_load_error

    return mathdx_load_error()

__all__ = ["MATHDX_BACKEND_ABI", "is_available", "load", "load_error"]
