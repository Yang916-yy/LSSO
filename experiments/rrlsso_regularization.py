"""Model-size-aware regularization and diagnostics for RRLSSO training."""

from __future__ import annotations

import math

import torch

from lsso import RRLSSO


# CLI weights retain their DeiT-III Base meaning. Scaling a global mean by
# M / 144 keeps the raw restoring force on each head invariant as depth and
# head count change.
RRLSSO_REGULARIZATION_REFERENCE_SCALARS = 12 * 12


def rrlsso_regularization(
    model: torch.nn.Module,
    *,
    gain_reference: dict[str, torch.Tensor],
    gain_anchor_weight: float,
    alpha_saturation_weight: float,
    alpha_saturation_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gauge fixing and a logit-space solve-strength saturation barrier."""

    if not 0 < alpha_saturation_fraction < 1:
        raise ValueError("alpha_saturation_fraction must lie in (0, 1)")
    reference = next(model.parameters())
    gain_square_sum = reference.new_zeros((), dtype=torch.float32)
    alpha_square_sum = reference.new_zeros((), dtype=torch.float32)
    gain_count = alpha_count = 0
    threshold_logit = math.log(
        alpha_saturation_fraction / (1.0 - alpha_saturation_fraction)
    )
    seen_gain_keys = set()
    for name, module in model.named_modules():
        if not isinstance(module, RRLSSO):
            continue
        key = name or "<root>"
        if hasattr(module, "theta_gain"):
            if key not in gain_reference:
                raise KeyError(f"missing gain reference for RRLSSO module {key!r}")
            target = gain_reference[key].to(
                device=module.theta_gain.device, dtype=torch.float32
            )
            if target.shape != module.theta_gain.shape:
                raise ValueError(
                    f"gain reference shape mismatch for {key}: "
                    f"{tuple(target.shape)} != {tuple(module.theta_gain.shape)}"
                )
            delta = module.theta_gain.float() - target
            gain_square_sum = gain_square_sum + delta.square().sum()
            gain_count += delta.numel()
            seen_gain_keys.add(key)
        excess_logit = torch.relu(module.theta_alpha.float() - threshold_logit)
        alpha_square_sum = alpha_square_sum + excess_logit.square().sum()
        alpha_count += excess_logit.numel()
    unused = set(gain_reference) - seen_gain_keys
    if unused:
        raise KeyError(f"gain reference contains unknown RRLSSO modules: {sorted(unused)}")
    gain_penalty = gain_square_sum / max(gain_count, 1)
    alpha_penalty = alpha_square_sum / max(alpha_count, 1)
    gain_scale = gain_count / RRLSSO_REGULARIZATION_REFERENCE_SCALARS
    alpha_scale = alpha_count / RRLSSO_REGULARIZATION_REFERENCE_SCALARS
    total = (
        gain_anchor_weight * gain_scale * gain_penalty
        + alpha_saturation_weight * alpha_scale * alpha_penalty
    )
    return total, {"gain_anchor": gain_penalty, "alpha_saturation": alpha_penalty}


def make_rrlsso_gain_reference(
    model: torch.nn.Module,
    *,
    anchor_to_current: bool,
    state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build the persistent log-gain gauge reference for one training stage."""

    modules = {
        name or "<root>": module
        for name, module in model.named_modules()
        if isinstance(module, RRLSSO) and hasattr(module, "theta_gain")
    }
    if state is not None and set(state) != set(modules):
        missing = sorted(set(modules) - set(state))
        extra = sorted(set(state) - set(modules))
        raise KeyError(f"invalid gain-reference state; missing={missing}, extra={extra}")
    result = {}
    for key, module in modules.items():
        if state is not None:
            value = state[key]
            if value.shape != module.theta_gain.shape:
                raise ValueError(
                    f"gain reference shape mismatch for {key}: "
                    f"{tuple(value.shape)} != {tuple(module.theta_gain.shape)}"
                )
            value = value.to(module.theta_gain)
        elif anchor_to_current:
            value = module.theta_gain.detach()
        else:
            value = torch.zeros_like(module.theta_gain)
        result[key] = value.detach().clone()
    return result


def rrlsso_parameter_diagnostics(
    model: torch.nn.Module,
    gain_reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Aggregate scale and solve-strength diagnostics without changing gradients."""

    theta_gain = []
    gain_delta = []
    alpha_ratio = []
    for name, module in model.named_modules():
        if not isinstance(module, RRLSSO):
            continue
        if hasattr(module, "theta_gain"):
            key = name or "<root>"
            current = module.theta_gain.detach().float()
            target = gain_reference[key].to(current)
            theta_gain.append(current.flatten())
            gain_delta.append((current - target).flatten())
        alpha_ratio.append(module.theta_alpha.detach().float().sigmoid().flatten())
    reference = next(model.parameters()).detach().new_zeros((), dtype=torch.float32)
    gains = torch.cat(theta_gain) if theta_gain else reference.reshape(1)
    deltas = torch.cat(gain_delta) if gain_delta else reference.reshape(1)
    ratios = torch.cat(alpha_ratio) if alpha_ratio else reference.reshape(1)
    return {
        "gain_log_mean": gains.mean(),
        "gain_log_std": gains.std(unbiased=False),
        "gain_anchor_rms": deltas.square().mean().sqrt(),
        "alpha_ratio_mean": ratios.mean(),
        "alpha_ratio_std": ratios.std(unbiased=False),
        "alpha_fraction_gt_080": (ratios > 0.8).float().mean(),
        "alpha_fraction_gt_095": (ratios > 0.95).float().mean(),
    }


__all__ = [
    "RRLSSO_REGULARIZATION_REFERENCE_SCALARS",
    "make_rrlsso_gain_reference",
    "rrlsso_parameter_diagnostics",
    "rrlsso_regularization",
]
