from __future__ import annotations

import pytest
import torch

from lsso import MixerAdapter
from lsso.mixer_adapter import RotaryMHA


def test_1d_rotary_mha_preserves_qk_relative_shift_and_gradients() -> None:
    torch.manual_seed(0)
    mixer = RotaryMHA(16, 4, dropout=0.0, rotary_1d=True)
    values = torch.randn(2, 7, 16, requires_grad=True)
    output = mixer(values)
    assert output.shape == values.shape
    output.square().mean().backward()
    assert values.grad is not None
    cos, sin = mixer._sequence_factors(values.detach(), values.shape[1])
    q = torch.randn(2, 4, 7, 4)
    rotated = mixer._apply_1d_rotary(q, cos, sin)
    torch.testing.assert_close(
        rotated.square().sum(dim=-1), q.square().sum(dim=-1),
        rtol=1e-5, atol=1e-6,
    )


def test_1d_rotary_mha_requires_even_head_width() -> None:
    with pytest.raises(ValueError, match="even head dimension"):
        RotaryMHA(12, 4, rotary_1d=True)


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_mixer_adapter_common_interface_mask_and_backward(mixer: str) -> None:
    torch.manual_seed(21)
    layer = MixerAdapter(dim=64, num_heads=4, mixer=mixer, rank=16).double()
    x = torch.randn(2, 13, 64, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[True] * 13, [True] * 7 + [False] * 6]
    )
    y = layer(x, valid_mask=mask)
    assert y.shape == x.shape
    assert torch.count_nonzero(y[1, 7:]) == 0
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[1, 7:]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_mixer_adapter_cuda_bfloat16(mixer: str) -> None:
    layer = MixerAdapter(dim=64, num_heads=4, mixer=mixer, rank=16).cuda().bfloat16()
    x = torch.randn(2, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.tensor(
        [[True] * 17, [True] * 9 + [False] * 8], device="cuda"
    )
    y = layer(x, valid_mask=mask)
    assert torch.isfinite(y).all()
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
