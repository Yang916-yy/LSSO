from __future__ import annotations

import copy

import pytest
import torch

from lsso import GroupedLSSO, GroupedRRLSSO, LSSO, RRLSSO
import lsso.modules as lsso_modules
from lsso.modules import _backward_compact_statistics, _compact_statistics, lsso


def _positive_parameters(heads: int, *, dtype: torch.dtype = torch.float64):
    mu = (torch.rand(heads, dtype=dtype) + 0.5).requires_grad_()
    gamma = (torch.rand(heads, dtype=dtype) + 0.1).requires_grad_()
    return mu, gamma


def test_split_n_compact_statistics_matches_single_gemm() -> None:
    torch.manual_seed(1707)
    systems, sequence, rank, width = 5, 37, 4, 7
    u = torch.randn(systems, sequence, rank, dtype=torch.float64)
    c = torch.randn(systems, sequence, width, dtype=torch.float64)
    expected_gram = u.transpose(1, 2) @ u
    expected_cross = u.transpose(1, 2) @ c
    gram, cross = _compact_statistics(
        u, c, solve_dtype=torch.float64, split_n=True, chunk_size=8
    )
    torch.testing.assert_close(gram, expected_gram, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(cross, expected_cross, rtol=2e-15, atol=2e-15)


def test_split_n_backward_statistics_matches_single_gemm() -> None:
    torch.manual_seed(1708)
    systems, sequence, rank, width = 3, 43, 5, 7
    u = torch.randn(systems, sequence, rank, dtype=torch.float64)
    y = torch.randn(systems, sequence, width, dtype=torch.float64)
    p = torch.randn(systems, sequence, width, dtype=torch.float64)
    actual = _backward_compact_statistics(
        u, y, p, split_n=True, chunk_size=8
    )
    expected = (
        y.transpose(1, 2) @ u,
        p.transpose(1, 2) @ u,
        -(p * y).sum(dim=(1, 2)),
    )
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=5e-14, atol=3e-15)


def test_padding_ratio_hint_is_reused_by_custom_backward(monkeypatch) -> None:
    hints = []

    def fake_native(*args, padding_ratio_hint=None, **kwargs):
        hints.append(padding_ratio_hint)
        return None

    monkeypatch.setattr(
        lsso_modules, "try_masked_stats_solve_spd", fake_native
    )
    B, H, N, r, dh = 2, 2, 7, 4, 5
    U = torch.randn(B, H, N, r, requires_grad=True)
    C = torch.randn(B, H, N, dh, requires_grad=True)
    mu = torch.ones(H, requires_grad=True)
    gamma = torch.full((H,), 0.05, requires_grad=True)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0, 0]],
        dtype=torch.bool,
    )

    lsso(
        U,
        C,
        mu,
        gamma,
        valid_mask=mask,
        padding_ratio_hint=0.25,
    ).square().mean().backward()

    assert hints == [0.25, 0.25]


def test_masked_functional_matches_individually_cropped_sequences() -> None:
    """Padding must not change valid outputs or compact solve statistics."""

    torch.manual_seed(0)
    B, H, N, r, dh = 3, 2, 9, 4, 5
    lengths = torch.tensor([9, 6, 2])
    mask = torch.arange(N)[None, :] < lengths[:, None]
    U = torch.randn(B, H, N, r, dtype=torch.float64)
    C = torch.randn(B, H, N, dh, dtype=torch.float64)
    mu, gamma = _positive_parameters(H)

    padded = lsso(
        U,
        C,
        mu,
        gamma,
        valid_mask=mask,
        length_normalize=True,
        length_reference=1.0,
    )

    for batch, length in enumerate(lengths.tolist()):
        cropped = lsso(
            U[batch : batch + 1, :, :length],
            C[batch : batch + 1, :, :length],
            mu,
            gamma,
            length_normalize=True,
            length_reference=1.0,
        )
        torch.testing.assert_close(padded[batch : batch + 1, :, :length], cropped)
        assert torch.count_nonzero(padded[batch, :, length:]) == 0


def test_masked_functional_ignores_padding_values_and_gradients() -> None:
    torch.manual_seed(1)
    B, H, N, r, dh = 2, 3, 8, 4, 6
    mask = torch.tensor(
        [[True, True, True, True, True, False, False, False],
         [True, True, False, False, False, False, False, False]]
    )
    U = torch.randn(B, H, N, r, dtype=torch.float64, requires_grad=True)
    C = torch.randn(B, H, N, dh, dtype=torch.float64, requires_grad=True)
    mu, gamma = _positive_parameters(H)

    output = lsso(U, C, mu, gamma, valid_mask=mask)
    probe = torch.randn_like(output) * mask[:, None, :, None]
    loss = (output * probe).sum()
    gradients = torch.autograd.grad(loss, (U, C, mu, gamma))

    changed_U = U.detach().clone()
    changed_C = C.detach().clone()
    padding = ~mask[:, None, :, None]
    changed_U = torch.where(padding, torch.randn_like(changed_U) * 1000, changed_U)
    changed_C = torch.where(padding, torch.randn_like(changed_C) * 1000, changed_C)
    changed = lsso(changed_U, changed_C, mu.detach(), gamma.detach(), valid_mask=mask)

    torch.testing.assert_close(output.detach(), changed)
    assert torch.count_nonzero(gradients[0] * padding) == 0
    assert torch.count_nonzero(gradients[1] * padding) == 0
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("module_type", [LSSO, RRLSSO, GroupedLSSO, GroupedRRLSSO])
@pytest.mark.parametrize("length_normalize", [False, True])
def test_masked_module_matches_cropped_prefix(
    module_type: type[torch.nn.Module],
    length_normalize: bool,
) -> None:
    torch.manual_seed(2)
    kwargs = dict(
        dim=32,
        num_heads=4,
        rank=8,
        bias=True,
        length_normalize=length_normalize,
        length_reference=1.0,
    )
    if module_type in (GroupedLSSO, GroupedRRLSSO):
        kwargs["num_relation_groups"] = 2
    layer = module_type(**kwargs).double()
    reference = copy.deepcopy(layer)
    x = torch.randn(2, 11, 32, dtype=torch.float64)
    lengths = [11, 4]
    mask = torch.arange(x.shape[1])[None, :] < torch.tensor(lengths)[:, None]

    padded = layer(x, valid_mask=mask)
    for batch, length in enumerate(lengths):
        cropped = reference(x[batch : batch + 1, :length])
        torch.testing.assert_close(
            padded[batch : batch + 1, :length],
            cropped,
            rtol=2e-9,
            atol=2e-9,
        )
        assert torch.count_nonzero(padded[batch, length:]) == 0


@pytest.mark.parametrize("module_type", [LSSO, RRLSSO, GroupedLSSO, GroupedRRLSSO])
def test_masked_module_has_zero_padding_input_gradient(
    module_type: type[torch.nn.Module],
) -> None:
    torch.manual_seed(3)
    kwargs = dict(dim=24, num_heads=4, rank=6, bias=True)
    if module_type in (GroupedLSSO, GroupedRRLSSO):
        kwargs["num_relation_groups"] = 2
    layer = module_type(**kwargs).double()
    x = torch.randn(2, 7, 24, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, True, True, True, True],
         [True, True, True, False, False, False, False]]
    )

    output = layer(x, valid_mask=mask)
    output.square().sum().backward()

    assert x.grad is not None
    assert torch.count_nonzero(x.grad[1, 3:]) == 0
    assert torch.isfinite(x.grad).all()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rrlsso_arbitrary_mask_preserves_original_position_ids() -> None:
    """Removing holes is equivalent only when original rotary positions remain."""

    torch.manual_seed(4)
    layer = RRLSSO(
        dim=32,
        num_heads=4,
        rank=8,
        bias=True,
        length_normalize=True,
    ).double()
    x = torch.randn(1, 8, 32, dtype=torch.float64)
    mask = torch.tensor([[True, False, True, True, False, True, False, True]])
    positions = torch.arange(x.shape[1]).unsqueeze(0)

    padded = layer(x, valid_mask=mask, position_ids=positions)
    kept = mask[0].nonzero(as_tuple=False).flatten()
    cropped = layer(
        x[:, kept],
        position_ids=positions[:, kept],
    )

    torch.testing.assert_close(padded[:, kept], cropped, rtol=2e-9, atol=2e-9)
    assert torch.count_nonzero(padded[:, ~mask[0]]) == 0


def test_all_padding_is_safe_and_zero() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 6, 24, requires_grad=True)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    layer = RRLSSO(dim=24, num_heads=4, rank=6, bias=True)

    output = layer(x, valid_mask=mask)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("module_type", [LSSO, RRLSSO])
def test_masked_cuda_bfloat16_matches_cropped_prefix(
    module_type: type[torch.nn.Module],
) -> None:
    """Exercise fused rotary/statistics paths with a CPU-originating mask."""

    torch.manual_seed(6)
    layer = module_type(
        dim=64,
        num_heads=4,
        rank=16,
        bias=True,
        length_normalize=True,
    ).cuda().to(torch.bfloat16)
    x = torch.randn(2, 13, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Keeping the mask on CPU also checks that the module moves it safely.
    mask = torch.tensor(
        [[True] * 13, [True] * 5 + [False] * 8],
        device="cpu",
    )

    padded = layer(x, valid_mask=mask)
    cropped = layer(x[1:2, :5])
    torch.testing.assert_close(padded[1:2, :5], cropped, rtol=3e-2, atol=3e-2)
    assert torch.count_nonzero(padded[1, 5:]).item() == 0

    padded.float().square().mean().backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad[1, 5:]).item() == 0
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("padding_ratio_hint", [0.1, 0.9])
def test_masked_cuda_nan_padding_cannot_leak_through_forward_or_backward(
    padding_ratio_hint: float,
) -> None:
    """Exercise the complete fused readout/backward path with poisoned padding."""
    torch.manual_seed(1708)
    B, H, N, rank, width = 2, 2, 65, 16, 32
    mask = torch.tensor(
        [[True] * N, [True] * 7 + [False] * (N - 7)], device="cuda"
    )
    padding_u = ~mask[:, None, :, None]
    u = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    u = torch.where(padding_u, torch.full_like(u, float("nan")), u).requires_grad_()
    c = torch.where(padding_u, torch.full_like(c, float("nan")), c).requires_grad_()
    mu = torch.ones(H, device="cuda", requires_grad=True)
    gamma = torch.full((H,), 0.02, device="cuda", requires_grad=True)

    output = lsso(
        u,
        c,
        mu,
        gamma,
        valid_mask=mask,
        padding_ratio_hint=padding_ratio_hint,
    )
    padding_y = padding_u.expand_as(output)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output.masked_select(padding_y)) == 0
    upstream = torch.randn_like(output)
    upstream = torch.where(
        mask[:, None, :, None], upstream, torch.full_like(upstream, float("nan"))
    )
    output.backward(upstream)
    assert u.grad is not None and c.grad is not None
    assert torch.isfinite(u.grad).all() and torch.isfinite(c.grad).all()
    assert torch.count_nonzero(u.grad.masked_select(padding_u.expand_as(u.grad))) == 0
    assert torch.count_nonzero(c.grad.masked_select(padding_y)) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dispatch", ["hybrid", "split_n"])
def test_long_backward_dispatch_matches_native_and_blocks_arbitrary_mask_leaks(
    monkeypatch, dispatch: str
) -> None:
    torch.manual_seed(1709)
    B, H, N, rank, width = 2, 2, 65, 16, 32
    mask = torch.rand(B, N, device="cuda") > 0.45
    mask[:, 0] = True
    u0 = torch.randn(B, H, N, rank, device="cuda", dtype=torch.bfloat16)
    c0 = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    probe = torch.randn(B, H, N, width, device="cuda", dtype=torch.bfloat16)
    probe = probe * mask[:, None, :, None]

    def run(u_value: torch.Tensor, c_value: torch.Tensor):
        u = u_value.detach().clone().requires_grad_()
        c = c_value.detach().clone().requires_grad_()
        mu = torch.ones(H, device="cuda", requires_grad=True)
        gamma = torch.full((H,), 0.02, device="cuda", requires_grad=True)
        output = lsso(
            u, c, mu, gamma, valid_mask=mask, padding_ratio_hint=0.9
        )
        gradients = torch.autograd.grad(
            (output * probe).sum(), (u, c, mu, gamma)
        )
        return output.detach(), tuple(value.detach() for value in gradients)

    native_output, native_gradients = run(u0, c0)
    monkeypatch.setattr(lsso_modules, "_FUSED_BACKWARD_MAX_SEQUENCE", 0)
    if dispatch == "split_n":
        monkeypatch.setattr(lsso_modules, "_SPLIT_BACKWARD_MIN_SEQUENCE", 1)
        monkeypatch.setattr(lsso_modules, "_SPLIT_BACKWARD_MAX_SYSTEMS", B * H)
        monkeypatch.setattr(lsso_modules, "_SPLIT_BACKWARD_CHUNK_SIZE", 8)
    else:
        monkeypatch.setattr(lsso_modules, "_SPLIT_BACKWARD_MIN_SEQUENCE", N + 1)

    output, gradients = run(u0, c0)
    torch.testing.assert_close(output, native_output, rtol=0, atol=0)
    for value, reference in zip(gradients, native_gradients, strict=True):
        torch.testing.assert_close(value, reference, rtol=4e-2, atol=4e-2)

    padding = ~mask[:, None, :, None]
    poisoned_u = torch.where(padding, torch.full_like(u0, float("nan")), u0)
    poisoned_c = torch.where(padding, torch.full_like(c0, float("nan")), c0)
    poisoned_output, poisoned_gradients = run(poisoned_u, poisoned_c)
    assert torch.isfinite(poisoned_output).all()
    assert torch.count_nonzero(
        poisoned_output.masked_select(padding.expand_as(poisoned_output))
    ) == 0
    torch.testing.assert_close(poisoned_output, output, rtol=0, atol=0)
    for value, reference in zip(poisoned_gradients, gradients, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, reference, rtol=4e-2, atol=4e-2)
    assert torch.count_nonzero(
        poisoned_gradients[0].masked_select(padding.expand_as(poisoned_gradients[0]))
    ) == 0
    assert torch.count_nonzero(
        poisoned_gradients[1].masked_select(padding.expand_as(poisoned_gradients[1]))
    ) == 0
