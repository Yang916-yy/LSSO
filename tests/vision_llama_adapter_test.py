from __future__ import annotations

import pytest
import torch

from examples.models.vision_llama import (
    VisionLLaMA,
    create_vision_llama,
    load_official_vision_llama_checkpoint,
)


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_plain_vision_llama_mixers_forward_backward(mixer: str) -> None:
    torch.manual_seed(30)
    model = VisionLLaMA(
        image_size=(32, 48), patch_size=8, num_classes=11,
        dim=64, depth=2, num_heads=4, mixer=mixer, rank=16,
        learned_position=True,
    ).double()
    image = torch.randn(2, 3, 32, 48, dtype=torch.float64, requires_grad=True)
    output = model(image)
    assert output.shape == (2, 11)
    output.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_plain_vision_llama_interpolates_learned_position() -> None:
    model = VisionLLaMA(
        image_size=32, patch_size=8, num_classes=0,
        dim=64, depth=1, num_heads=4, mixer="rrlsso", rank=16,
    )
    assert model(torch.randn(1, 3, 40, 56)).shape == (1, 64)


def test_scale_factory_matches_official_plain_dimensions() -> None:
    small = create_vision_llama("small", mixer="rrlsso", num_classes=0)
    assert small.dim == 384 and len(small.blocks) == 12
    assert small.blocks[0].mixer.mixer.num_heads == 6


def test_official_checkpoint_key_conversion_for_mha() -> None:
    model = VisionLLaMA(
        image_size=16, patch_size=8, num_classes=3,
        dim=64, depth=1, num_heads=4, mixer="mha", rank=16,
        learned_position=False,
    )
    qkv = torch.randn_like(model.blocks[0].mixer.mixer.qkv.weight)
    checkpoint = {
        "model": {
            "blocks.0.attn.qkv.weight": qkv,
            "blocks.0.attn.proj.weight": torch.randn_like(model.blocks[0].mixer.mixer.proj.weight),
            "blocks.0.mlp.c_fc1.weight": torch.randn_like(model.blocks[0].mlp.w1.weight),
        }
    }
    _missing, unexpected = load_official_vision_llama_checkpoint(model, checkpoint)
    assert not unexpected
    torch.testing.assert_close(model.blocks[0].mixer.mixer.qkv.weight, qkv)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_plain_vision_llama_rrlsso_cuda_bfloat16() -> None:
    model = VisionLLaMA(
        image_size=32, patch_size=8, num_classes=5,
        dim=64, depth=2, num_heads=4, mixer="rrlsso", rank=16,
    ).cuda().bfloat16()
    image = torch.randn(2, 3, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = model(image)
    output.float().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(image.grad).all()
