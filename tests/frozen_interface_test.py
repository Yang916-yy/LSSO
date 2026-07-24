from __future__ import annotations

import inspect

import pytest
import torch

from lsso import GroupedLSSO, GroupedRRLSSO, LSSO, RRLSSO


@pytest.mark.parametrize(
    "module_type", [LSSO, RRLSSO, GroupedLSSO, GroupedRRLSSO]
)
def test_public_module_signature_hides_retired_parameterizations(
    module_type: type[torch.nn.Module],
) -> None:
    parameters = inspect.signature(module_type.__init__).parameters
    assert "alpha_init" not in parameters
    assert "solve_parameterization" not in parameters
    assert "basis_normalization" not in parameters


def test_alpha_starts_at_frozen_internal_reference_but_remains_learnable() -> None:
    layer = RRLSSO(32, 4, rank=8)
    gain, alpha = layer.effective_gain_alpha()
    torch.testing.assert_close(gain, torch.ones_like(gain))
    torch.testing.assert_close(alpha, torch.ones_like(alpha))
    assert layer.theta_alpha.requires_grad
    with torch.no_grad():
        layer.theta_alpha.copy_(torch.tensor([-8.0, -1.0, 2.0, 8.0]))
    _, changed = layer.effective_gain_alpha()
    torch.testing.assert_close(
        changed, layer.theta_alpha.exp(), rtol=2e-7, atol=0.0
    )


def test_checkpoint_theta_alpha_overrides_internal_initialization() -> None:
    source = RRLSSO(32, 4, rank=8)
    with torch.no_grad():
        source.theta_alpha.copy_(torch.tensor([-2.0, -0.5, 0.7, 3.0]))
    restored = RRLSSO(32, 4, rank=8)
    restored.load_state_dict(source.state_dict(), strict=True)
    torch.testing.assert_close(restored.theta_alpha, source.theta_alpha)


def test_removed_alpha_initialization_is_rejected_by_python_signature() -> None:
    with pytest.raises(TypeError, match="alpha_init"):
        RRLSSO(32, 4, rank=8, alpha_init=100.0)


def test_log_alpha_spectral_response_derivative_bound() -> None:
    theta = torch.linspace(-12.0, 12.0, 4097, dtype=torch.float64)
    eigenvalues = torch.logspace(-6, 6, 25, dtype=torch.float64)
    scaled = theta.exp().unsqueeze(1) * eigenvalues.unsqueeze(0)
    derivative_magnitude = scaled / (1.0 + scaled).square()
    assert derivative_magnitude.max() <= 0.25 + 1e-15
