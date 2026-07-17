"""Stable facade for the optional MathDx CUDA backend.

The legacy :mod:`lsso.mathdx_backend` path remains supported.  Internal kernel
dispatch functions deliberately stay private until their contracts stabilize.
"""

from os import PathLike

from ..mathdx_backend import (
    is_mathdx_available,
    load_mathdx_backend,
    mathdx_load_error,
)


def load(path: str | PathLike[str] | None = None) -> bool:
    """Load the MathDx extension, returning whether it is available."""

    return load_mathdx_backend(path)


def is_available() -> bool:
    """Return whether the MathDx extension has been loaded successfully."""

    return is_mathdx_available()


def load_error() -> Exception | None:
    """Return the backend load error, if loading was attempted and failed."""

    return mathdx_load_error()


__all__ = ["is_available", "load", "load_error"]
