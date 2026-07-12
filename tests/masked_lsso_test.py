from __future__ import annotations

import copy

import pytest
import torch

from lsso import GroupedLSSO, GroupedRRLSSO, LSSO, RRLSSO
from lsso.modules import lsso


def _positive_parameters(heads: int, *, dtype: torch.dtype = torch.float64):
    mu = (torch.rand(heads, dtype=dtype) + 0.5).requires_grad_()
    gamma = (torch.rand(heads, dtype=dtype) + 0.1).requires_grad_()
    return mu, gamma


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
