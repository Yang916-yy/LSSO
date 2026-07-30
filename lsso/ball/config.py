from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoreMode(str, Enum):
    """Compact-core modes supported by the one LSSO operator."""

    DYNAMIC = "dynamic"
    STATIC = "static"
    ZERO = "zero"


@dataclass(frozen=True, slots=True)
class LSSOConfig:
    """Configuration for the one LSSO operator and its two ablations."""

    dim: int
    num_heads: int
    rank: int = 16
    core_mode: CoreMode = CoreMode.DYNAMIC
    rank_rotary: bool = True
    bias: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("dim", self.dim),
            ("num_heads", self.num_heads),
            ("rank", self.rank),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )

        try:
            core_mode = CoreMode(self.core_mode)
        except (TypeError, ValueError) as error:
            choices = ", ".join(mode.value for mode in CoreMode)
            raise ValueError(
                f"core_mode must be one of {{{choices}}}, got {self.core_mode!r}"
            ) from error
        object.__setattr__(self, "core_mode", core_mode)

        if not isinstance(self.rank_rotary, bool):
            raise TypeError(
                f"rank_rotary must be a bool, got {type(self.rank_rotary).__name__}"
            )
        if not isinstance(self.bias, bool):
            raise TypeError(f"bias must be a bool, got {type(self.bias).__name__}")

        if self.dim <= 0 or self.num_heads <= 0 or self.rank <= 0:
            raise ValueError("dim, num_heads, and rank must be positive")
        if self.dim % self.num_heads:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}"
            )
        if self.rank_rotary and self.rank % 2:
            raise ValueError(
                f"Rank-Rotary requires an even rank, got rank={self.rank}"
            )

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


__all__ = ["CoreMode", "LSSOConfig"]
