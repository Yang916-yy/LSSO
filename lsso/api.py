"""Stable public API.

Implementation files may be reorganized without changing this module.  Keep
this surface intentionally small; experimental helpers belong in their owning
module until they are stable enough to support.
"""

from .mixer_adapter import MixerAdapter, RotaryMHA
from .modules import (
    LSSO,
    length_normalize_basis,
    lsso,
    make_solve_state,
    read_solve_state,
    update_solve_state,
)
from .modules_grouped import GroupedLSSO, GroupedRRLSSO
from .modules_v2 import RRLSSO, RoPELSSO, apply_rank_rope, apply_rank_rotary
from .rotary_2d import apply_2d_rank_rotary, apply_2d_rotary, make_2d_position_coords
from .types import SolveStateCache

__all__ = [
    "GroupedLSSO",
    "GroupedRRLSSO",
    "LSSO",
    "MixerAdapter",
    "RRLSSO",
    "RoPELSSO",
    "RotaryMHA",
    "SolveStateCache",
    "apply_2d_rank_rotary",
    "apply_2d_rotary",
    "apply_rank_rope",
    "apply_rank_rotary",
    "length_normalize_basis",
    "lsso",
    "make_2d_position_coords",
    "make_solve_state",
    "read_solve_state",
    "update_solve_state",
]
