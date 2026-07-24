from __future__ import annotations

import pytest
import torch

from experiments.deit3_official_recipe import (
    official_recipe,
    three_augment_transform,
    validation_transform,
    virtual_group_repeated_samples,
)
from experiments.rrlsso_diagnostics import rrlsso_parameter_diagnostics
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
    assert (small.augmentation_group_size, base.augmentation_group_size) == (256, 256)
    assert large.augmentation_group_size == 64
    assert all(profile.repeated_aug == 3 and profile.bce_loss for profile in (small, base, large))


def test_official_finetuning_profiles_and_valid_stages() -> None:
    base224 = official_recipe("deit3_base_patch16_rrlsso", "finetune224")
    base384 = official_recipe("deit3_base_patch16_rrlsso", "finetune384")
    assert (base224.image_size, base224.drop_path_rate) == (224, 0.20)
    assert (base384.image_size, base384.drop_path_rate) == (384, 0.15)
    assert base224.optimizer == base384.optimizer == "adamw"
    assert base224.effective_batch == base384.effective_batch == 512
    assert (base224.augmentation_group_size, base384.augmentation_group_size) == (64, 32)
    with pytest.raises(ValueError, match="no small"):
        official_recipe("deit3_small_patch16_rrlsso", "finetune224")


def test_repeated_augmentation_separates_views_by_virtual_group() -> None:
    output = list(
        virtual_group_repeated_samples(range(8), repeats=3, group_size=4)
    )
    groups = [output[start : start + 4] for start in range(0, len(output), 4)]
    assert groups == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
        [4, 5, 6, 7],
    ]
    assert all(len(set(group)) == len(group) for group in groups)


def test_official_transforms_have_expected_crop_ratio_and_three_augment() -> None:
    training = three_augment_transform(192)
    validation = validation_transform(192)
    assert len(training.transforms) == 6
    assert validation.transforms[0].size == 192
    assert validation.transforms[1].size == (192, 192)


def test_rrlsso_diagnostics_report_unbounded_strength_and_knee() -> None:
    module = RRLSSO(dim=64, num_heads=4, rank=8)
    with torch.no_grad():
        module.theta_alpha.copy_(torch.log(torch.tensor([0.5, 1.0, 2.0, 4.0])))
    diagnostics = rrlsso_parameter_diagnostics(module)
    assert diagnostics["alpha_observed_min"].item() == pytest.approx(0.5)
    assert diagnostics["alpha_observed_max"].item() == pytest.approx(4.0)
    assert diagnostics["beta_mean"].item() == pytest.approx(
        torch.tensor([2.0, 1.0, 0.5, 0.25]).mean().item()
    )
