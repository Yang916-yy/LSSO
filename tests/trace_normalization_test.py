from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck, gradgradcheck

from lsso import apply_rank_rotary, lsso, lsso_gain_alpha, trace_normalize_basis
from lsso.modules_v2 import RRLSSO


@pytest.mark.parametrize(
    ("capability", "sequence", "padding", "expected", "chunk"),
    [
        ((8, 0), 9000, 0.1, "split_n", 512),
        ((8, 9), 9000, 0.8, "cta", 512),
        ((8, 9), 17000, 0.8, "split_n", 512),
        ((9, 0), 9000, None, "split_n", 1024),
        ((10, 0), 4096, 0.1, "cta", 1024),
        ((12, 0), 3000, 0.1, "cta", 1024),
        ((12, 0), 5000, 0.1, "split_n", 1024),
        ((12, 0), 5000, 0.8, "cta", 1024),
        ((12, 0), 9000, 0.8, "split_n", 1024),
    ],
)
def test_masked_trace_architecture_schedule(
    capability, sequence, padding, expected, chunk
) -> None:
    import lsso.modules as modules

    assert modules._masked_trace_forward_strategy(
        sequence, 4, padding, compute_capability=capability
    ) == (expected, chunk)


@pytest.mark.parametrize("masked", [False, True])
def test_trace_custom_function_has_second_derivatives(masked: bool) -> None:
    torch.manual_seed(3400 + int(masked))
    B, H, N, rank, width = 1, 1, 3, 2, 2
    u = (0.1 * torch.randn(B, H, N, rank, dtype=torch.float64)).requires_grad_()
    c = torch.randn(B, H, N, width, dtype=torch.float64, requires_grad=True)
    gain = torch.tensor([[[[1.1]]]], dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor([[[[0.4]]]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, True, False]]) if masked else None

    def function(u_, c_, gain_, alpha_):
        return lsso_gain_alpha(
            u_, c_, gain_, alpha_, trace_normalize=True,
            length_normalize=False, valid_mask=mask,
        )

    assert gradgradcheck(
        function,
        (u, c, gain, alpha),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=True,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_masked_trace_split_n_dispatch_matches_reference_and_blocks_nan(
    monkeypatch,
) -> None:
    import lsso.mathdx_backend as backend
    import lsso.modules as modules

    monkeypatch.setenv("LSSO_PATH_COUNTERS", "1")
    monkeypatch.setattr(
        modules, "_masked_trace_forward_strategy",
        lambda *args, **kwargs: ("split_n", 8),
    )
    backend.reset_mathdx_path_counters()
    torch.manual_seed(3450)
    B, H, N, rank, width = 2, 1, 19, 16, 16
    mask = torch.arange(N, device="cuda")[None, :] < torch.tensor(
        [N, 6], device="cuda"
    )[:, None]
    active = mask[:, None, :, None]
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    u = torch.where(active, u, torch.full_like(u, float("nan")))
    c = torch.where(active, c, torch.full_like(c, float("nan")))
    gain = torch.ones(1, H, 1, 1, device="cuda")
    alpha = torch.full((1, H, 1, 1), 0.4, device="cuda")
    with torch.no_grad():
        actual = lsso_gain_alpha(
            u, c, gain, alpha, trace_normalize=True,
            length_normalize=False, valid_mask=mask,
            padding_ratio_hint=0.1,
        )
        normalized = trace_normalize_basis(
            u, mask, eps=1e-5, length_normalize=False
        )
        expected = lsso_gain_alpha(
            normalized, c, gain, alpha, trace_normalize=False,
            length_normalize=False, valid_mask=mask,
        )
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(
        actual.masked_select((~active).expand_as(actual))
    ) == 0
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=5e-2, atol=5e-2
    )
    assert backend.get_mathdx_path_counters()[
        "forward.trace_masked_split_n"
    ] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rank", [24, 40, 56])
@pytest.mark.parametrize("masked", [False, True])
def test_trace_arbitrary_rank_keeps_bucketed_fallback(
    rank: int, masked: bool
) -> None:
    """Non-specialized ranks use the compact bucket solver, not a hard error."""

    torch.manual_seed(3300 + rank)
    B, H, N, width = 2, 2, 29, 32
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    gain = torch.full((1, H, 1, 1), 1.1, device="cuda")
    alpha = torch.full((1, H, 1, 1), 0.4, device="cuda")
    mask = None
    if masked:
        mask = torch.arange(N, device="cuda")[None, :] < torch.tensor(
            [N, 9], device="cuda"
        )[:, None]
    with torch.no_grad():
        actual = lsso_gain_alpha(
            u,
            c,
            gain,
            alpha,
            trace_normalize=True,
            length_normalize=False,
            valid_mask=mask,
        )
        normalized = trace_normalize_basis(
            u, mask, eps=1e-5, length_normalize=False
        )
        expected = lsso_gain_alpha(
            normalized,
            c,
            gain,
            alpha,
            trace_normalize=False,
            length_normalize=False,
            valid_mask=mask,
        )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.float(), expected.float(), rtol=5e-2, atol=5e-2
    )


def test_training_trace_path_derives_energy_from_gram_statistics(monkeypatch) -> None:
    import lsso.modules as modules

    def forbidden_prepass(*args, **kwargs):
        raise AssertionError("standalone trace-energy pass must not run")

    monkeypatch.setattr(modules, "_trace_normalization_factors", forbidden_prepass)
    z, c, _mu, _gamma, mask = _inputs()
    z.requires_grad_()
    c.requires_grad_()
    gain = torch.full((1, z.shape[1], 1, 1), 1.3, dtype=z.dtype, requires_grad=True)
    alpha = torch.full((1, z.shape[1], 1, 1), 0.8, dtype=z.dtype, requires_grad=True)
    output = lsso_gain_alpha(
        z,
        c,
        gain,
        alpha,
        trace_normalize=True,
        valid_mask=mask,
        length_reference=2.0,
    )
    torch.autograd.grad(output.square().sum(), (z, c, gain, alpha))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_trace_training_dispatches_the_radial_backward_epilogue(
    monkeypatch,
) -> None:
    import lsso.mathdx_backend as backend

    monkeypatch.setenv("LSSO_PATH_COUNTERS", "1")
    backend.reset_mathdx_path_counters()
    torch.manual_seed(2399)
    u = torch.randn(
        2, 2, 17, 32, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    c = torch.randn(
        2, 2, 17, 64, device="cuda", dtype=torch.bfloat16,
        requires_grad=True,
    )
    gain = torch.ones(1, 2, 1, 1, device="cuda", requires_grad=True)
    alpha = torch.full(
        (1, 2, 1, 1), 0.4, device="cuda", requires_grad=True
    )
    output = lsso_gain_alpha(
        u, c, gain, alpha, trace_normalize=True,
        length_normalize=False,
    )
    output.float().square().mean().backward()
    counters = backend.get_mathdx_path_counters()
    assert counters["forward.trace_unmasked_cta"] == 1
    assert counters["backward.adjoint_native"] == 1
    assert counters["backward.dual_grad_u_radial"] == 1


def _inputs(dtype=torch.float64, device="cpu"):
    torch.manual_seed(2401)
    B, H, N, rank, width = 2, 3, 7, 4, 5
    z = torch.randn(B, H, N, rank, dtype=dtype, device=device)
    c = torch.randn(B, H, N, width, dtype=dtype, device=device)
    mu = torch.full((1, H, 1, 1), 0.9, dtype=dtype, device=device)
    gamma = torch.full((1, H, 1, 1), 0.03, dtype=dtype, device=device)
    mask = torch.tensor(
        [[True, True, False, True, False, True, True],
         [True, False, True, True, True, False, True]],
        device=device,
    )
    return z, c, mu, gamma, mask


def test_trace_normalization_fixes_energy_and_preserves_relative_radii() -> None:
    z, _c, _mu, _gamma, mask = _inputs()
    reference = 3.5
    normalized = trace_normalize_basis(
        z,
        mask,
        eps=0.0,
        length_normalize=True,
        length_reference=reference,
    )
    expected_trace = z.shape[-1] * reference
    torch.testing.assert_close(
        normalized.square().sum(dim=(-2, -1)),
        torch.full((z.shape[0], z.shape[1]), expected_trace, dtype=z.dtype),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.count_nonzero(normalized.masked_select(~mask[:, None, :, None])) == 0

    for batch in range(z.shape[0]):
        valid = mask[batch].nonzero(as_tuple=False).flatten()
        i, j = int(valid[0]), int(valid[1])
        raw_ratio = z[batch, :, i].norm(dim=-1) / z[batch, :, j].norm(dim=-1)
        normalized_ratio = (
            normalized[batch, :, i].norm(dim=-1)
            / normalized[batch, :, j].norm(dim=-1)
        )
        torch.testing.assert_close(normalized_ratio, raw_ratio, rtol=1e-12, atol=1e-12)


def test_gain_alpha_solve_budget_is_invariant_to_repeated_sequence_length() -> None:
    z, c, _mu, _gamma, _mask = _inputs()
    gain = torch.tensor(1.3, dtype=z.dtype).view(1, 1, 1, 1)
    alpha = torch.tensor(0.8, dtype=z.dtype).view(1, 1, 1, 1)
    mu = gain.reciprocal()
    gamma = alpha / gain
    repeats = 5

    short = lsso(
        z,
        c,
        mu,
        gamma,
        trace_normalize=True,
        normalization_eps=0.0,
        length_normalize=True,
        length_reference=1.0,
    )
    long = lsso(
        z.repeat_interleave(repeats, dim=2),
        c.repeat_interleave(repeats, dim=2),
        mu,
        gamma,
        trace_normalize=True,
        normalization_eps=0.0,
        length_normalize=True,
        length_reference=1.0,
    )
    torch.testing.assert_close(
        long,
        short.repeat_interleave(repeats, dim=2),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("length_normalize", [False, True])
def test_absorbed_trace_forward_matches_explicit_normalization(
    masked: bool, length_normalize: bool
) -> None:
    z, c, mu, gamma, mask = _inputs()
    active_mask = mask if masked else None
    normalized = trace_normalize_basis(
        z,
        active_mask,
        eps=1e-6,
        length_normalize=length_normalize,
        length_reference=2.5,
    )
    explicit = lsso(
        normalized,
        c,
        mu,
        gamma,
        length_normalize=False,
        valid_mask=active_mask,
    )
    absorbed = lsso(
        z,
        c,
        mu,
        gamma,
        trace_normalize=True,
        normalization_eps=1e-6,
        length_normalize=length_normalize,
        length_reference=2.5,
        valid_mask=active_mask,
    )
    torch.testing.assert_close(absorbed, explicit, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("masked", [False, True])
def test_absorbed_trace_custom_backward_matches_explicit_autograd(masked: bool) -> None:
    z0, c0, mu0, gamma0, mask = _inputs()
    active_mask = mask if masked else None
    probe = torch.randn_like(c0)
    if active_mask is not None:
        probe = probe * active_mask[:, None, :, None]

    def explicit_run():
        z = z0.clone().requires_grad_()
        c = c0.clone().requires_grad_()
        mu = mu0.clone().requires_grad_()
        gamma = gamma0.clone().requires_grad_()
        normalized = trace_normalize_basis(
            z,
            active_mask,
            eps=1e-6,
            length_normalize=True,
            length_reference=2.5,
        )
        y = lsso(
            normalized,
            c,
            mu,
            gamma,
            length_normalize=False,
            valid_mask=active_mask,
        )
        gradients = torch.autograd.grad((y * probe).sum(), (z, c, mu, gamma))
        return y.detach(), gradients

    def absorbed_run():
        z = z0.clone().requires_grad_()
        c = c0.clone().requires_grad_()
        mu = mu0.clone().requires_grad_()
        gamma = gamma0.clone().requires_grad_()
        y = lsso(
            z,
            c,
            mu,
            gamma,
            trace_normalize=True,
            normalization_eps=1e-6,
            length_normalize=True,
            length_reference=2.5,
            valid_mask=active_mask,
        )
        gradients = torch.autograd.grad((y * probe).sum(), (z, c, mu, gamma))
        return y.detach(), gradients

    expected_y, expected_gradients = explicit_run()
    actual_y, actual_gradients = absorbed_run()
    torch.testing.assert_close(actual_y, expected_y, rtol=2e-12, atol=2e-12)
    # The canonical g/alpha chain and statistics-first Gram trace change
    # floating-point association without changing the analytic gradient.
    # FP64 gradcheck below remains the exact derivative guard.
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        tolerance = 2e-6 if actual.numel() <= mu0.numel() else 2e-8
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("masked", [False, True])
def test_trace_custom_backward_passes_gradcheck(masked: bool) -> None:
    torch.manual_seed(2402)
    B, H, N, rank, width = 1, 2, 4, 3, 2
    z = torch.randn(B, H, N, rank, dtype=torch.float64, requires_grad=True)
    c = torch.randn(B, H, N, width, dtype=torch.float64, requires_grad=True)
    mu = torch.full((1, H, 1, 1), 0.8, dtype=torch.float64, requires_grad=True)
    gamma = torch.full((1, H, 1, 1), 0.02, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True, True]]) if masked else None

    assert gradcheck(
        lambda z, c, mu, gamma: lsso(
            z,
            c,
            mu,
            gamma,
            trace_normalize=True,
            normalization_eps=1e-6,
            length_normalize=True,
            length_reference=2.0,
            valid_mask=mask,
        ),
        (z, c, mu, gamma),
        eps=1e-6,
        atol=2e-5,
        rtol=2e-4,
        fast_mode=True,
    )


@pytest.mark.parametrize("masked", [False, True])
def test_zero_epsilon_gradient_is_orthogonal_to_global_radius(masked: bool) -> None:
    z, c, mu, gamma, mask = _inputs()
    z.requires_grad_()
    active_mask = mask if masked else None
    y = lsso(
        z,
        c,
        mu,
        gamma,
        trace_normalize=True,
        normalization_eps=0.0,
        length_reference=2.0,
        valid_mask=active_mask,
    )
    probe = torch.randn_like(y)
    if active_mask is not None:
        probe = probe * active_mask[:, None, :, None]
    grad_z, = torch.autograd.grad((y * probe).sum(), (z,))
    safe_z = z if active_mask is None else torch.where(
        active_mask[:, None, :, None], z, torch.zeros_like(z)
    )
    radial = (grad_z * safe_z).sum(dim=(-2, -1))
    torch.testing.assert_close(radial, torch.zeros_like(radial), rtol=0, atol=1e-7)


def test_trace_normalization_commutes_with_rank_rotary() -> None:
    z, _c, _mu, _gamma, mask = _inputs()
    left = trace_normalize_basis(
        apply_rank_rotary(z), mask, eps=1e-6, length_reference=2.0
    )
    right = apply_rank_rotary(
        trace_normalize_basis(z, mask, eps=1e-6, length_reference=2.0)
    )
    torch.testing.assert_close(left, right, rtol=2e-12, atol=2e-12)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rank", [16, 32, 48, 64])
@pytest.mark.parametrize("masked", [False, True])
def test_cuda_trace_absorption_stress_matches_explicit(
    dtype: torch.dtype, rank: int, masked: bool
) -> None:
    torch.manual_seed(2500 + rank)
    B, H, N, width = 3, 2, 17, 64
    mask = torch.rand(B, N, device="cuda") > 0.25 if masked else None
    if mask is not None:
        mask[:, 0] = True
    z0 = torch.randn(B, H, N, rank, device="cuda", dtype=dtype)
    c0 = torch.randn(B, H, N, width, device="cuda", dtype=dtype)
    mu0 = torch.full((1, H, 1, 1), 0.9, device="cuda", dtype=torch.float32)
    gamma0 = torch.full((1, H, 1, 1), 0.03, device="cuda", dtype=torch.float32)
    probe = torch.randn_like(c0)
    if mask is not None:
        probe = probe * mask[:, None, :, None]

    def run(absorbed: bool):
        z = z0.clone().requires_grad_()
        c = c0.clone().requires_grad_()
        mu = mu0.clone().requires_grad_()
        gamma = gamma0.clone().requires_grad_()
        if absorbed:
            output = lsso(
                z, c, mu, gamma, trace_normalize=True,
                normalization_eps=1e-5, length_reference=2.0,
                valid_mask=mask,
            )
        else:
            normalized = trace_normalize_basis(
                z, mask, eps=1e-5, length_reference=2.0
            )
            output = lsso(
                normalized, c, mu, gamma,
                length_normalize=False, valid_mask=mask,
            )
        gradients = torch.autograd.grad(
            (output * probe).sum(), (z, c, mu, gamma)
        )
        return output.detach(), tuple(value.detach() for value in gradients)

    def run_fp32_reference():
        z = z0.float().detach().requires_grad_()
        c = c0.float().detach().requires_grad_()
        mu = mu0.clone().requires_grad_()
        gamma = gamma0.clone().requires_grad_()
        normalized = trace_normalize_basis(
            z, mask, eps=1e-5, length_reference=2.0
        )
        output = lsso(
            normalized, c, mu, gamma,
            length_normalize=False, valid_mask=mask,
        )
        gradients = torch.autograd.grad(
            (output * probe.float()).sum(), (z, c, mu, gamma)
        )
        return output.detach(), tuple(value.detach() for value in gradients)

    explicit_output, _ = run(False)
    expected_output, expected_gradients = run_fp32_reference()
    actual_output, actual_gradients = run(True)
    tolerance = 6e-2 if dtype == torch.bfloat16 else 1.5e-2
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(),
        rtol=tolerance, atol=tolerance,
    )
    torch.testing.assert_close(
        actual_output.float(), explicit_output.float(),
        rtol=tolerance, atol=tolerance,
    )
    for actual, expected in zip(
        actual_gradients[:2], expected_gradients[:2], strict=True
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=tolerance, atol=tolerance
        )
    # Shared scalar gradients sum many signed token/channel contributions and
    # can be close to cancellation. Use a mixed-precision absolute tolerance
    # while retaining a relative bound away from zero.
    # The maintained path never rounds a materialized normalized basis back to
    # BF16/FP16. Near cancellation, compare shared scalar gradients with an
    # absolute bound appropriate to the two different rounding paths; FP64
    # gradcheck above guards the underlying derivative exactly.
    scalar_atol = 1.0 if dtype == torch.bfloat16 else 2e-1
    scalar_rtol = 8e-2 if dtype == torch.bfloat16 else 3e-2
    for actual, expected in zip(
        actual_gradients[2:], expected_gradients[2:], strict=True
    ):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=scalar_rtol, atol=scalar_atol
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_bfloat16_trace_path_matches_explicit_and_blocks_mask_leaks() -> None:
    torch.manual_seed(2403)
    B, H, N, rank, width = 65, 1, 19, 32, 64
    mask = torch.rand(B, N, device="cuda") > 0.3
    mask[:, 0] = True
    z0 = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c0 = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    padding = ~mask[:, None, :, None]
    z0 = torch.where(padding, torch.full_like(z0, float("nan")), z0)
    c0 = torch.where(padding, torch.full_like(c0, float("nan")), c0)
    mu0 = torch.full((1, H, 1, 1), 0.9, device="cuda", dtype=torch.float32)
    gamma0 = torch.full((1, H, 1, 1), 0.03, device="cuda", dtype=torch.float32)
    probe = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    probe = torch.where(padding, torch.zeros_like(probe), probe)

    def run(absorbed: bool):
        z = z0.clone().requires_grad_()
        c = c0.clone().requires_grad_()
        mu = mu0.clone().requires_grad_()
        gamma = gamma0.clone().requires_grad_()
        if absorbed:
            y = lsso(
                z, c, mu, gamma, trace_normalize=True,
                normalization_eps=1e-5, length_reference=2.0, valid_mask=mask,
            )
        else:
            normalized = trace_normalize_basis(
                z, mask, eps=1e-5, length_reference=2.0
            )
            y = lsso(
                normalized, c, mu, gamma,
                length_normalize=False, valid_mask=mask,
            )
        gradients = torch.autograd.grad((y * probe).sum(), (z, c, mu, gamma))
        return y.detach(), tuple(value.detach() for value in gradients)

    expected_y, expected_gradients = run(False)
    actual_y, actual_gradients = run(True)
    assert torch.isfinite(actual_y).all()
    assert torch.count_nonzero(actual_y.masked_select(padding.expand_as(actual_y))) == 0
    torch.testing.assert_close(actual_y.float(), expected_y.float(), rtol=4e-2, atol=4e-2)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual.float(), expected.float(), rtol=6e-2, atol=6e-2)
    assert torch.count_nonzero(
        actual_gradients[0].masked_select(padding.expand_as(actual_gradients[0]))
    ) == 0
    assert torch.count_nonzero(
        actual_gradients[1].masked_select(padding.expand_as(actual_gradients[1]))
    ) == 0
