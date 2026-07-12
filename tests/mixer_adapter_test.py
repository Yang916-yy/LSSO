from __future__ import annotations

import pytest
import torch

from lsso import (
    MixerAdapter,
    apply_2d_rank_rotary,
    make_2d_position_coords,
)


def test_2d_rotary_preserves_norm_and_relative_kernel() -> None:
    torch.manual_seed(20)
    U = torch.randn(2, 3, 15, 16, dtype=torch.float64)
    coords = make_2d_position_coords((3, 5), dtype=torch.float64)
    rotated = apply_2d_rank_rotary(U, position_coords=coords)
    shifted = apply_2d_rank_rotary(
        U, position_coords=coords + torch.tensor([13.0, -7.0])
    )
    torch.testing.assert_close(rotated.norm(dim=-1), U.norm(dim=-1))
    torch.testing.assert_close(
        rotated @ rotated.transpose(-1, -2),
        shifted @ shifted.transpose(-1, -2),
        atol=1e-10,
        rtol=1e-10,
    )


def test_2d_coords_support_prefix_and_non_square_grid() -> None:
    coords = make_2d_position_coords((2, 3), num_prefix_tokens=2)
    assert coords.shape == (8, 2)
    torch.testing.assert_close(coords[:2], torch.zeros(2, 2))
    torch.testing.assert_close(coords[-1], torch.tensor([2.0, 1.0]))
    x = torch.randn(1, 2, 8, 16)
    assert apply_2d_rank_rotary(
        x, spatial_shape=(2, 3), num_prefix_tokens=2
    ).shape == x.shape


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_mixer_adapter_common_interface_mask_and_backward(mixer: str) -> None:
    torch.manual_seed(21)
    layer = MixerAdapter(dim=64, num_heads=4, mixer=mixer, rank=16).double()
    x = torch.randn(2, 13, 64, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor(
        [[True] * 13, [True] * 7 + [False] * 6]
    )
    y = layer(
        x,
        valid_mask=mask,
        spatial_shape=(3, 4),
        num_prefix_tokens=1,
    )
    assert y.shape == x.shape
    assert torch.count_nonzero(y[1, 7:]) == 0
    y.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[1, 7:]) == 0


def test_adapter_accepts_batched_explicit_window_coordinates() -> None:
    layer = MixerAdapter(dim=64, num_heads=4, mixer="rrlsso", rank=16)
    x = torch.randn(2, 6, 64)
    coords = make_2d_position_coords((2, 3)).expand(2, -1, -1).clone()
    coords[1, :, 0] += 3
    y = layer(x, position_coords=coords)
    assert y.shape == x.shape


def test_2d_rotary_rejects_incompatible_dimension() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        apply_2d_rank_rotary(torch.randn(1, 2, 6, 10), spatial_shape=(2, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_mixer_adapter_cuda_bfloat16(mixer: str) -> None:
    layer = MixerAdapter(dim=64, num_heads=4, mixer=mixer, rank=16).cuda().bfloat16()
    x = torch.randn(2, 17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mask = torch.tensor(
        [[True] * 17, [True] * 9 + [False] * 8], device="cuda"
    )
    y = layer(x, valid_mask=mask, spatial_shape=(4, 4), num_prefix_tokens=1)
    assert torch.isfinite(y).all()
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
