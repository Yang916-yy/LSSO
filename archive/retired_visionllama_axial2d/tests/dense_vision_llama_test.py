from __future__ import annotations

import pytest
import torch

from examples.models.simple_feature_pyramid import SimpleFeaturePyramid
from examples.models.vision_llama import VisionLLaMA
from examples.models.vision_llama_dense import (
    DenseVisionLLaMA,
    load_dense_vision_llama_checkpoint,
)
from examples.models.windowed_dense_mixer import (
    partition_windows,
    unpartition_windows,
)


def test_window_partition_round_trip_with_padding() -> None:
    x = torch.arange(2 * 5 * 7 * 8).reshape(2, 5 * 7, 8)
    windows, mask, padded = partition_windows(x, (5, 7), 4)
    assert windows.shape == (8, 16, 8)
    assert mask.sum().item() == 2 * 5 * 7
    restored = unpartition_windows(
        windows,
        batch=2,
        spatial_shape=(5, 7),
        window_size=4,
        padded_shape=padded,
    )
    torch.testing.assert_close(restored, x)


def test_divisible_windows_skip_padding_mask() -> None:
    x = torch.randn(2, 8 * 16, 32)
    windows, mask, padded = partition_windows(x, (8, 16), 4)
    assert mask is None and padded == (8, 16)
    restored = unpartition_windows(
        windows,
        batch=2,
        spatial_shape=(8, 16),
        window_size=4,
        padded_shape=padded,
    )
    torch.testing.assert_close(restored, x)


@pytest.mark.parametrize("mixer", ["mha", "lsso", "rrlsso"])
def test_dense_plain_mixers_non_square_forward_backward(mixer: str) -> None:
    model = DenseVisionLLaMA(
        image_size=(32, 48),
        patch_size=8,
        dim=64,
        depth=3,
        num_heads=4,
        mixer=mixer,
        rank=16,
        window_size=3,
        global_block_indices=(1,),
        out_indices=(0, 2),
    ).double()
    image = torch.randn(2, 3, 40, 56, dtype=torch.float64, requires_grad=True)
    features = model.forward_intermediates(image)
    assert [tuple(feature.shape) for feature in features] == [
        (2, 64, 5, 7),
        (2, 64, 5, 7),
    ]
    features[-1].square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()


def test_dense_model_loads_plain_classifier_parameters() -> None:
    kwargs = dict(
        image_size=32,
        patch_size=8,
        dim=64,
        depth=2,
        num_heads=4,
        mixer="rrlsso",
        rank=16,
    )
    classifier = VisionLLaMA(num_classes=10, **kwargs)
    dense = DenseVisionLLaMA(
        num_classes=0,
        window_size=2,
        global_block_indices=(1,),
        **kwargs,
    )
    missing, unexpected = load_dense_vision_llama_checkpoint(
        dense, {"model_state": classifier.state_dict()}
    )
    assert missing == [] and unexpected == []
    torch.testing.assert_close(dense.patch_embed.weight, classifier.patch_embed.weight)
    torch.testing.assert_close(dense.pos_embed, classifier.pos_embed)


def test_dense_rejects_input_not_divisible_by_patch_size() -> None:
    model = DenseVisionLLaMA(
        image_size=32,
        patch_size=8,
        dim=32,
        depth=1,
        num_heads=2,
        mixer="rrlsso",
        rank=8,
    )
    with pytest.raises(ValueError, match="must be divisible"):
        model(torch.randn(1, 3, 33, 40))


def test_edge_window_mask_is_reused() -> None:
    model = DenseVisionLLaMA(
        image_size=32,
        patch_size=8,
        dim=32,
        depth=1,
        num_heads=2,
        mixer="rrlsso",
        rank=8,
        window_size=3,
        global_block_indices=(),
    )
    image = torch.randn(1, 3, 40, 56)
    model(image)
    cache = model.blocks[0]._dense_window_mask_cache
    first_mask = next(iter(cache.values()))
    model(image)
    assert len(cache) == 1 and next(iter(cache.values())) is first_mask


def test_simple_feature_pyramid_shapes_and_backward() -> None:
    pyramid = SimpleFeaturePyramid(64, 32)
    feature = torch.randn(2, 64, 5, 7, requires_grad=True)
    outputs = pyramid(feature)
    assert [tuple(output.shape) for output in outputs] == [
        (2, 32, 20, 28),
        (2, 32, 10, 14),
        (2, 32, 5, 7),
        (2, 32, 2, 3),
    ]
    sum(output.square().mean() for output in outputs).backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()


def test_dense_backbone_and_simple_fpn_integrate() -> None:
    backbone = DenseVisionLLaMA(
        image_size=32,
        patch_size=8,
        dim=64,
        depth=2,
        num_heads=4,
        mixer="rrlsso",
        rank=16,
        window_size=3,
        global_block_indices=(1,),
    )
    pyramid = SimpleFeaturePyramid(64, 24)
    features = pyramid(backbone(torch.randn(1, 3, 48, 64)))
    assert [feature.shape[-2:] for feature in features] == [
        (24, 32), (12, 16), (6, 8), (3, 4)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dense_rrlsso_simple_fpn_cuda_bfloat16() -> None:
    backbone = DenseVisionLLaMA(
        image_size=32,
        patch_size=8,
        dim=64,
        depth=2,
        num_heads=4,
        mixer="rrlsso",
        rank=16,
        window_size=3,
        global_block_indices=(1,),
    ).cuda().bfloat16()
    pyramid = SimpleFeaturePyramid(64, 32).cuda().bfloat16()
    image = torch.randn(
        2, 3, 48, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    outputs = pyramid(backbone(image))
    sum(output.float().mean() for output in outputs).backward()
    assert all(torch.isfinite(output).all() for output in outputs)
    assert image.grad is not None and torch.isfinite(image.grad).all()
