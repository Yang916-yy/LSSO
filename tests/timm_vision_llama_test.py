from __future__ import annotations

import pytest
import timm
import torch

import examples.models  # noqa: F401 - imports registration side effects
from experiments.imagenet_wds_train import shard_cache_name, shard_commands
from examples.models.vision_llama import VisionLLaMA


REGISTERED_MODELS = (
    "vision_llama_small_mha",
    "vision_llama_small_lsso_r32",
    "vision_llama_small_rrlsso_r32",
    "vision_llama_base_mha",
    "vision_llama_base_lsso_r32",
    "vision_llama_base_rrlsso_r32",
    "vision_llama_large_mha",
    "vision_llama_large_lsso_r32",
    "vision_llama_large_rrlsso_r32",
)


def test_all_plain_vision_llama_models_are_registered() -> None:
    available = set(timm.list_models("vision_llama_*"))
    assert set(REGISTERED_MODELS) <= available


@pytest.mark.parametrize(
    "name",
    [
        "vision_llama_large_mha",
        "vision_llama_large_lsso_r32",
        "vision_llama_large_rrlsso_r32",
    ],
)
def test_large_registered_factories_build_with_compact_overrides(name: str) -> None:
    model = timm.create_model(
        name,
        img_size=32,
        patch_size=4,
        num_classes=5,
        dim=64,
        depth=1,
        num_heads=4,
    )
    assert isinstance(model, VisionLLaMA)
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 5)


@pytest.mark.parametrize(
    ("name", "mixer"),
    [
        ("vision_llama_small_mha", "mha"),
        ("vision_llama_small_lsso_r32", "lsso"),
        ("vision_llama_small_rrlsso_r32", "rrlsso"),
    ],
)
def test_timm_create_model_forward_and_metadata(name: str, mixer: str) -> None:
    model = timm.create_model(
        name, pretrained=False, img_size=32, patch_size=4, num_classes=13
    )
    assert isinstance(model, VisionLLaMA)
    assert model.mixer_name == mixer
    assert model.num_classes == 13 and model.num_features == 384
    assert model.default_cfg["input_size"] == (3, 224, 224)
    assert model(torch.randn(1, 3, 32, 32)).shape == (1, 13)


def test_timm_classifier_reset_and_no_weight_decay() -> None:
    model = timm.create_model("vision_llama_small_rrlsso_r32", num_classes=7)
    model.reset_classifier(3)
    assert model.get_classifier().out_features == 3
    assert model.no_weight_decay() == {"cls_token", "pos_embed"}


def test_registered_rank_can_be_passed_explicitly() -> None:
    model = timm.create_model(
        "vision_llama_small_rrlsso_r32",
        rank=32,
        img_size=32,
        patch_size=4,
        dim=64,
        depth=1,
        num_heads=4,
    )
    assert isinstance(model, VisionLLaMA)


def test_imagenet_shard_names_match_repository_layout(tmp_path) -> None:
    train = shard_commands("train", tmp_path, "timm/imagenet-1k-wds")
    validation = shard_commands("validation", tmp_path, "timm/imagenet-1k-wds")
    assert "imagenet1k-train-0000.tar" in train[0]
    assert "imagenet1k-train-1023.tar" in train[-1]
    assert "imagenet1k-validation-00.tar" in validation[0]
    assert "imagenet1k-validation-63.tar" in validation[-1]
    assert train[0].startswith("pipe:curl ")
    assert "--retry 10 --retry-connrefused" in train[0]
    assert "${HF_TOKEN}" in train[0]
    assert shard_cache_name(train[0]) == "imagenet1k-train-0000.tar"


def test_pretrained_requires_explicit_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="no bundled pretrained weights"):
        timm.create_model("vision_llama_small_mha", pretrained=True)
