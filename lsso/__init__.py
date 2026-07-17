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
    RoPELSSO,
    RotaryMHA,
    SolveStateCache,
    apply_2d_rank_rotary,
    apply_2d_rotary,
    apply_rank_rope,
    apply_rank_rotary,
    length_normalize_basis,
    lsso,
    make_solve_state,
    make_2d_position_coords,
    read_solve_state,
    update_solve_state,
)

__version__ = "0.2.0.dev0"

__all__ = [
    "__version__",
    "LSSO",
    "MixerAdapter",
    "RotaryMHA",
    "GroupedLSSO",
    "GroupedRRLSSO",
    "RRLSSO",
    "RoPELSSO",
    "SolveStateCache",
    "apply_rank_rope",
    "apply_rank_rotary",
    "apply_2d_rotary",
    "apply_2d_rank_rotary",
    "make_2d_position_coords",
    "length_normalize_basis",
    "make_solve_state",
    "update_solve_state",
    "read_solve_state",
    "lsso",
]
