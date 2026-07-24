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
    assert not hasattr(layer, "alpha_max")
    assert not hasattr(layer, "_alpha_max_state")
    torch.testing.assert_close(
        gain, torch.full_like(gain, 1.0), rtol=2e-7, atol=2e-7
    )
    torch.testing.assert_close(
        alpha, torch.full_like(alpha, 1.0), rtol=2e-7, atol=2e-7
    )


def test_removed_bounded_checkpoint_is_rejected_instead_of_silently_migrated() -> None:
    layer = RRLSSO(32, 4, rank=8)
    state = layer.state_dict()
    state["_alpha_max_state"] = torch.tensor(3.0)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        layer.load_state_dict(state, strict=True)


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


def test_gain_alpha_requires_positive_but_has_no_upper_bound() -> None:
    with pytest.raises(ValueError, match="alpha_init"):
        RRLSSO(
            32,
            4,
            rank=8,
            solve_parameterization="gain_alpha",
            alpha_init=0.0,
        )
    layer = RRLSSO(32, 4, rank=8, alpha_init=100.0)
    _, alpha = layer.effective_gain_alpha()
    torch.testing.assert_close(alpha, torch.full_like(alpha, 100.0))


def test_log_alpha_parameterization_is_exact_and_unbounded() -> None:
    layer = RRLSSO(32, 4, rank=8, alpha_init=1.0).double()
    values = torch.tensor([-8.0, -1.0, 2.0, 8.0], dtype=torch.float64)
    with torch.no_grad():
        layer.theta_alpha.copy_(values)
    _, alpha = layer.effective_gain_alpha()
    torch.testing.assert_close(alpha, values.exp(), rtol=1e-12, atol=0.0)
    assert alpha[-1] > 1_000.0


def test_reciprocal_woodbury_matches_direct_solve_at_large_alpha() -> None:
    torch.manual_seed(41)
    tokens, rank, rhs = 19, 5, 7
    basis = torch.randn(tokens, rank, dtype=torch.float64)
    values = torch.randn(tokens, rhs, dtype=torch.float64)
    alpha = torch.tensor(10_000.0, dtype=torch.float64)
    beta = alpha.reciprocal()

    direct = torch.linalg.solve(
        torch.eye(tokens, dtype=torch.float64) + alpha * (basis @ basis.mT),
        values,
    )
    reciprocal = values - basis @ torch.linalg.solve(
        beta * torch.eye(rank, dtype=torch.float64) + basis.mT @ basis,
        basis.mT @ values,
    )
    torch.testing.assert_close(reciprocal, direct, rtol=2e-10, atol=2e-10)


def test_log_alpha_spectral_response_derivative_is_bounded_by_one_quarter() -> None:
    theta = torch.linspace(-12.0, 12.0, 4097, dtype=torch.float64)
    eigenvalues = torch.logspace(-6, 6, 25, dtype=torch.float64)
    scaled = theta.exp().unsqueeze(1) * eigenvalues.unsqueeze(0)
    derivative_magnitude = scaled / (1.0 + scaled).square()
    assert derivative_magnitude.max() <= 0.25 + 1e-15
