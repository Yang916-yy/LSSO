"""Shared result and state types for LSSO operators."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LSSODiagnostics:
    gamma_over_mu: torch.Tensor
    effective_rank: torch.Tensor
    correction_ratio: torch.Tensor


@dataclass
class LSSOAux:
    UtU: torch.Tensor | None
    local: torch.Tensor
    correction: torch.Tensor
    mu: torch.Tensor
    gamma: torch.Tensor


@dataclass
class SolveStateCache:
    """Compressed aggregate statistics for incremental solve/readout.

    ``S`` stores ``sum(U_i.T @ U_i)`` and ``P`` stores
    ``sum(U_i.T @ C_i)``.  RRLSSO callers rotate ``U`` before updating the
    state.
    """

    S: torch.Tensor
    P: torch.Tensor
    length: int | torch.Tensor = 0


__all__ = ["LSSOAux", "LSSODiagnostics", "SolveStateCache"]
