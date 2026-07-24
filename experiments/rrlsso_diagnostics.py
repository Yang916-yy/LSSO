"""Read-only diagnostics for unbounded log-parameterized RRLSSO scales."""

from __future__ import annotations

import torch

from lsso import RRLSSO


def rrlsso_parameter_diagnostics(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Aggregate gain, alpha, and reciprocal spectral-knee statistics."""

    theta_gain = []
    alpha = []
    for module in model.modules():
        if not isinstance(module, RRLSSO):
            continue
        if hasattr(module, "theta_gain"):
            theta_gain.append(module.theta_gain.detach().float().flatten())
        alpha.append(module.theta_alpha.detach().float().exp().flatten())
    reference = next(model.parameters()).detach().new_zeros((), dtype=torch.float32)
    gains = torch.cat(theta_gain) if theta_gain else reference.reshape(1)
    strengths = torch.cat(alpha) if alpha else reference.new_ones(1)
    beta = strengths.reciprocal()
    return {
        "gain_log_mean": gains.mean(),
        "gain_log_std": gains.std(unbiased=False),
        "alpha_mean": strengths.mean(),
        "alpha_std": strengths.std(unbiased=False),
        "alpha_observed_min": strengths.min(),
        "alpha_observed_max": strengths.max(),
        "beta_mean": beta.mean(),
    }


__all__ = ["rrlsso_parameter_diagnostics"]
