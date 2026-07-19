from __future__ import annotations

import pytest
import torch

from examples.models.vision_llama_pyramid import PyramidVisionLLaMA


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_pyramid_outputs_four_dense_feature_levels(mixer: str) -> None:
    model = PyramidVisionLLaMA(
        image_size=64, num_classes=7,
        dims=(32, 64, 96, 128), depths=(1, 1, 1, 1),
        heads=(2, 4, 6, 8), mixer=mixer, rank=16, window_size=7,
    )
    image = torch.randn(2, 3, 64, 64, requires_grad=True)
    features = model.forward_features(image)
    assert [tuple(item.shape) for item in features] == [
        (2, 32, 16, 16), (2, 64, 8, 8), (2, 96, 4, 4), (2, 128, 2, 2)
    ]
    logits = model(image)
    assert logits.shape == (2, 7)
    logits.mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_pyramid_alternating_global_policy() -> None:
    model = PyramidVisionLLaMA(
        image_size=32, num_classes=0,
        dims=(32, 64, 96, 128), depths=(2, 2, 2, 2),
        heads=(2, 4, 6, 8), mixer="rrlsso", rank=16,
        attention_policy="alternating-global",
    )
    output = model(torch.randn(1, 3, 32, 32))
    assert output.shape == (1, 128)


def test_pyramid_window_padding_non_square_input() -> None:
    model = PyramidVisionLLaMA(
        image_size=64, num_classes=0,
        dims=(32, 64, 96, 128), depths=(1, 1, 1, 1),
        heads=(2, 4, 6, 8), mixer="rrlsso", rank=16, window_size=7,
    )
    features = model.forward_features(torch.randn(1, 3, 64, 80))
    assert features[0].shape[-2:] == (16, 20)
