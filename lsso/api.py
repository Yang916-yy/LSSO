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
    lsso_gain_alpha,
    make_solve_state,
    read_solve_state,
    trace_normalize_basis,
    update_solve_state,
)
from .modules_grouped import GroupedLSSO, GroupedRRLSSO
from .modules_v2 import RRLSSO, apply_rank_rotary
from .types import SolveStateCache

__all__ = [
    "GroupedLSSO",
    "GroupedRRLSSO",
    "LSSO",
    "MixerAdapter",
    "RRLSSO",
    "RotaryMHA",
    "SolveStateCache",
    "apply_rank_rotary",
    "length_normalize_basis",
    "lsso",
    "lsso_gain_alpha",
    "make_solve_state",
    "read_solve_state",
    "trace_normalize_basis",
    "update_solve_state",
]
