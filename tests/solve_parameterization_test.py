from __future__ import annotations

import pytest
import torch

from lsso import GroupedRRLSSO, LSSO, RRLSSO


def test_token_rms_is_a_pytorch_only_ablation() -> None:
    import lsso.mathdx_backend as backend

    assert not hasattr(backend, "try_prepare_basis")
    layer = RRLSSO(32, 4, rank=8, basis_normalization="token_rms")
    x = torch.randn(2, 9, 32, requires_grad=True)
    layer(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def _copy_shared_state(source: torch.nn.Module, target: torch.nn.Module) -> None:
    target_state = target.state_dict()
    for name, value in source.state_dict().items():
        if name in target_state and target_state[name].shape == value.shape:
            target_state[name] = value.detach().clone()
    target.load_state_dict(target_state)


@pytest.mark.parametrize(
    "factory",
    [
        lambda mode: LSSO(32, 4, rank=8, solve_parameterization=mode),
        lambda mode: RRLSSO(32, 4, rank=8, solve_parameterization=mode),
        lambda mode: GroupedRRLSSO(
            32, 4, 2, rank=8, solve_parameterization=mode
        ),
    ],
)
def test_defaults_use_frozen_trace_initialization(factory) -> None:
    layer = factory("gain_alpha")
    gain, alpha = layer.effective_gain_alpha()
    assert layer.alpha_max == 3.0
    torch.testing.assert_close(layer._alpha_max_state, torch.tensor(3.0))
    torch.testing.assert_close(
        gain, torch.full_like(gain, 1.0), rtol=2e-7, atol=2e-7
    )
    torch.testing.assert_close(
        alpha, torch.full_like(alpha, 1.2), rtol=2e-7, atol=2e-7
    )


def test_pre_metadata_gain_alpha_checkpoint_preserves_legacy_ceiling() -> None:
    legacy = RRLSSO(
        32,
        4,
        rank=8,
        gain_init=1.4426742274994273,
        alpha_init=1.0776072417497349,
        alpha_max=2.0,
    )
    state = legacy.state_dict()
    state.pop("_alpha_max_state")

    restored = RRLSSO(32, 4, rank=8)
    restored.load_state_dict(state, strict=True)
    gain, alpha = restored.effective_gain_alpha()
    assert restored.alpha_max == 2.0
    torch.testing.assert_close(
        gain, torch.full_like(gain, 1.4426742274994273), rtol=2e-7, atol=2e-7
    )
    torch.testing.assert_close(
        alpha, torch.full_like(alpha, 1.0776072417497349), rtol=2e-7, atol=2e-7
    )


def test_legacy_state_dict_is_migrated_exactly() -> None:
    layer = RRLSSO(32, 4, rank=8)
    state = layer.state_dict()
    state.pop("theta_gain")
    state.pop("theta_alpha")
    state["theta_mu"] = torch.zeros(4)
    state["theta_gamma"] = torch.full((4,), 0.5)
    layer.load_state_dict(state, strict=True)
    gain, alpha = layer.effective_gain_alpha()
    torch.testing.assert_close(
        gain, torch.full_like(gain, 1.4426742274994273), rtol=2e-7, atol=2e-7
    )
    torch.testing.assert_close(
        alpha, torch.full_like(alpha, 1.0776072417497349), rtol=2e-7, atol=2e-7
    )


def test_gain_alpha_coordinates_are_decoupled() -> None:
    layer = RRLSSO(32, 4, rank=8, solve_parameterization="gain_alpha")
    gain0, alpha0 = (value.detach().clone() for value in layer.effective_gain_alpha())

    with torch.no_grad():
        layer.theta_gain.add_(0.25)
    gain1, alpha1 = layer.effective_gain_alpha()
    assert not torch.equal(gain1, gain0)
    torch.testing.assert_close(alpha1, alpha0)

    with torch.no_grad():
        layer.theta_alpha.sub_(0.4)
    gain2, alpha2 = layer.effective_gain_alpha()
    torch.testing.assert_close(gain2, gain1)
    assert not torch.equal(alpha2, alpha1)


@pytest.mark.parametrize(
    "factory,groups,width",
    [
        (lambda mode: LSSO(32, 4, rank=8, solve_parameterization=mode), 4, 8),
        (lambda mode: RRLSSO(32, 4, rank=8, solve_parameterization=mode), 4, 8),
        (
            lambda mode: GroupedRRLSSO(
                32, 4, 2, rank=8, solve_parameterization=mode
            ),
            2,
            16,
        ),
    ],
)
def test_fixed_gain_initialization_is_function_matched(factory, groups, width) -> None:
    torch.manual_seed(11)
    learned = factory("gain_alpha").double()
    fixed = factory("fixed_gain_alpha").double()
    _copy_shared_state(learned, fixed)
    fixed.fold_fixed_gain_into_output(force=True)

    gain, alpha = fixed.effective_gain_alpha()
    torch.testing.assert_close(gain, torch.ones_like(gain))
    learned_gain, learned_alpha = learned.effective_gain_alpha()
    torch.testing.assert_close(alpha, learned_alpha, rtol=2e-7, atol=2e-7)
    expected_scale = learned_gain.repeat_interleave(width)
    torch.testing.assert_close(
        fixed.w_o.weight,
        learned.w_o.weight * expected_scale.unsqueeze(0),
        rtol=2e-7,
        atol=2e-7,
    )

    x = torch.randn(2, 11, 32, dtype=torch.float64)
    torch.testing.assert_close(fixed(x), learned(x), rtol=2e-6, atol=2e-7)
    assert groups * width == 32
    assert not hasattr(fixed, "theta_gain")


def test_gain_alpha_rejects_initial_strength_above_bound() -> None:
    with pytest.raises(ValueError, match="alpha_init"):
        RRLSSO(
            32,
            4,
            rank=8,
            solve_parameterization="gain_alpha",
            alpha_max=1.0,
        )
