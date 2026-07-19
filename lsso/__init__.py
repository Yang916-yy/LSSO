"""Public package interface for LSSO.

New code should import public symbols from :mod:`lsso`, :mod:`lsso.ops`, or
:mod:`lsso.nn`.  Legacy implementation modules remain importable so existing
research scripts and checkpoints keep working.
"""

from .api import (
    GroupedLSSO,
    GroupedRRLSSO,
    LSSO,
    MixerAdapter,
    RRLSSO,
    RotaryMHA,
    SolveStateCache,
    apply_rank_rotary,
    length_normalize_basis,
    lsso,
    lsso_gain_alpha,
    make_solve_state,
    read_solve_state,
    trace_normalize_basis,
    update_solve_state,
)
from .mathdx_backend import (
    get_mathdx_path_counters,
    reset_mathdx_path_counters,
)
from .backends import MATHDX_BACKEND_ABI

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "MATHDX_BACKEND_ABI",
    "LSSO",
    "MixerAdapter",
    "RotaryMHA",
    "GroupedLSSO",
    "GroupedRRLSSO",
    "RRLSSO",
    "SolveStateCache",
    "apply_rank_rotary",
    "length_normalize_basis",
    "make_solve_state",
    "update_solve_state",
    "read_solve_state",
    "trace_normalize_basis",
    "lsso",
    "lsso_gain_alpha",
    "get_mathdx_path_counters",
    "reset_mathdx_path_counters",
]
