from __future__ import annotations

import pytest
import torch

from experiments.deit3_official_recipe import (
    official_recipe,
    repeat_samples,
    rrlsso_regularization,
    three_augment_transform,
    validation_transform,
)
from lsso import RRLSSO


def test_official_size_specific_pretraining_profiles() -> None:
    small = official_recipe("deit3_small_patch16_rrlsso", "pretrain")
    base = official_recipe("deit3_base_patch16_rrlsso", "pretrain")
    large = official_recipe("deit3_large_patch16_rrlsso", "pretrain")
    assert (small.image_size, small.peak_lr, small.drop_path_rate) == (224, 4e-3, 0.05)
    assert (base.image_size, base.peak_lr, base.drop_path_rate) == (192, 3e-3, 0.20)
    assert (large.image_size, large.peak_lr, large.drop_path_rate) == (192, 3e-3, 0.45)
    assert all(profile.optimizer == "fusedlamb" for profile in (small, base, large))
    assert all(profile.effective_batch == 2048 for profile in (small, base, large))
    assert all(profile.repeated_aug == 3 and profile.bce_loss for profile in (small, base, large))


def test_official_finetuning_profiles_and_valid_stages() -> None:
    base224 = official_recipe("deit3_base_patch16_rrlsso", "finetune224")
    base384 = official_recipe("deit3_base_patch16_rrlsso", "finetune384")
    assert (base224.image_size, base224.drop_path_rate) == (224, 0.20)
    assert (base384.image_size, base384.drop_path_rate) == (384, 0.15)
    assert base224.optimizer == base384.optimizer == "adamw"
    assert base224.effective_batch == base384.effective_batch == 512
    with pytest.raises(ValueError, match="no small"):
        official_recipe("deit3_small_patch16_rrlsso", "finetune224")


def test_repeated_augmentation_stream_repeats_before_transform() -> None:
    assert list(repeat_samples(iter(["a", "b"]), 3)) == [
        "a", "a", "a", "b", "b", "b"
    ]


def test_official_transforms_have_expected_crop_ratio_and_three_augment() -> None:
    training = three_augment_transform(192)
    validation = validation_transform(192)
    assert len(training.transforms) == 6
    assert validation.transforms[0].size == 192
    assert validation.transforms[1].size == (192, 192)


def test_rrlsso_regularizer_is_zero_at_default_initialization_and_differentiable() -> None:
    module = RRLSSO(dim=64, num_heads=4, rank=8, gain_init=1.0, alpha_init=1.2, alpha_max=3.0)
    penalty, components = rrlsso_regularization(
        module,
        gain_anchor_weight=1e-4,
        alpha_saturation_weight=1e-4,
        alpha_saturation_fraction=0.8,
    )
    assert penalty.item() == pytest.approx(0.0, abs=1e-12)
    assert components["gain_anchor"].item() == pytest.approx(0.0, abs=1e-12)
    assert components["alpha_saturation"].item() == pytest.approx(0.0, abs=1e-12)
    with torch.no_grad():
        module.theta_gain.add_(0.5)
        module.theta_alpha.add_(5.0)
    penalty, _ = rrlsso_regularization(
        module,
        gain_anchor_weight=1e-4,
        alpha_saturation_weight=1e-4,
        alpha_saturation_fraction=0.8,
    )
    penalty.backward()
    assert penalty.item() > 0
    assert module.theta_gain.grad is not None
    assert module.theta_alpha.grad is not None
