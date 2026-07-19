"""Official DeiT-III ImageNet-1K recipes and RRLSSO-only regularizers."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar

import torch
from PIL import ImageFilter, ImageOps
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
from torchvision import transforms

from lsso import RRLSSO

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
T = TypeVar("T")


@dataclass(frozen=True)
class DeiT3Recipe:
    size: str
    stage: str
    image_size: int
    epochs: int
    peak_lr: float
    min_lr: float
    warmup_lr: float
    warmup_epochs: int
    weight_decay: float
    drop_path_rate: float
    optimizer: str
    effective_batch: int
    augmentation_group_size: int
    repeated_aug: int
    bce_loss: bool
    label_smoothing: float
    augmentation: str


DEIT3_OFFICIAL_RECIPES: dict[tuple[str, str], DeiT3Recipe] = {
    ("small", "pretrain"): DeiT3Recipe(
        "small", "pretrain", 224, 800, 4e-3, 1e-5, 1e-6, 5,
        0.05, 0.05, "fusedlamb", 2048, 256, 3, True, 0.0, "three_augment",
    ),
    ("base", "pretrain"): DeiT3Recipe(
        "base", "pretrain", 192, 800, 3e-3, 1e-5, 1e-6, 5,
        0.05, 0.20, "fusedlamb", 2048, 256, 3, True, 0.0, "three_augment",
    ),
    ("large", "pretrain"): DeiT3Recipe(
        "large", "pretrain", 192, 800, 3e-3, 1e-5, 1e-6, 5,
        0.05, 0.45, "fusedlamb", 2048, 64, 3, True, 0.0, "three_augment",
    ),
    ("base", "finetune224"): DeiT3Recipe(
        "base", "finetune224", 224, 20, 1e-5, 1e-5, 1e-6, 5,
        0.10, 0.20, "adamw", 512, 64, 1, False, 0.1, "randaugment",
    ),
    ("large", "finetune224"): DeiT3Recipe(
        "large", "finetune224", 224, 20, 1e-5, 1e-5, 1e-6, 5,
        0.10, 0.45, "adamw", 512, 64, 1, False, 0.1, "randaugment",
    ),
    ("small", "finetune384"): DeiT3Recipe(
        "small", "finetune384", 384, 20, 1e-5, 1e-5, 1e-6, 5,
        0.10, 0.00, "adamw", 512, 64, 1, False, 0.1, "randaugment",
    ),
    ("base", "finetune384"): DeiT3Recipe(
        "base", "finetune384", 384, 20, 1e-5, 1e-5, 1e-6, 5,
        0.10, 0.15, "adamw", 512, 32, 1, False, 0.1, "randaugment",
    ),
    ("large", "finetune384"): DeiT3Recipe(
        "large", "finetune384", 384, 20, 1e-5, 1e-5, 1e-6, 5,
        0.10, 0.40, "adamw", 512, 16, 1, False, 0.1, "randaugment",
    ),
}


def model_size(model_name: str) -> str:
    for size in ("small", "base", "large"):
        if f"deit3_{size}_" in model_name:
            return size
    raise ValueError(f"cannot infer DeiT-III size from {model_name!r}")


def official_recipe(model_name: str, stage: str) -> DeiT3Recipe:
    key = (model_size(model_name), stage)
    if key not in DEIT3_OFFICIAL_RECIPES:
        valid = sorted(stage_name for size, stage_name in DEIT3_OFFICIAL_RECIPES if size == key[0])
        raise ValueError(f"official DeiT-III has no {key[0]} {stage!r} recipe; valid stages: {valid}")
    return DEIT3_OFFICIAL_RECIPES[key]


class GaussianBlur:
    def __init__(self, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image):
        return image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization:
    def __call__(self, image):
        return ImageOps.solarize(image)


def three_augment_transform(image_size: int) -> transforms.Compose:
    """Meta DeiT-III 3-Augment: crop/flip, one transform, jitter, normalize."""

    return transforms.Compose(
        [
            RandomResizedCropAndInterpolation(
                image_size, scale=(0.08, 1.0), interpolation="bicubic"
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice(
                [transforms.Grayscale(3), Solarization(), GaussianBlur()]
            ),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def randaugment_finetune_transform(image_size: int):
    return create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=0.3,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.0,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def validation_transform(image_size: int) -> transforms.Compose:
    # Official eval_crop_ratio=1.0.
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def virtual_group_repeated_samples(
    source: Iterable[T], *, repeats: int, group_size: int
) -> Iterator[T]:
    """Emit independent RA views in virtual-device-sized groups.

    Each source block is replayed as a whole rather than repeating individual
    samples consecutively.  Consequently, no virtual augmentation group contains
    two views of the same source sample.  The stage belongs before image decoding
    so its bounded buffer stores compressed WebDataset records.
    """

    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    if group_size < 1:
        raise ValueError(f"group_size must be positive, got {group_size}")
    iterator = iter(source)
    while True:
        block = list(itertools.islice(iterator, group_size))
        if len(block) != group_size:
            return
        for _ in range(repeats):
            yield from block


def rrlsso_regularization(
    model: torch.nn.Module,
    *,
    gain_anchor_weight: float,
    alpha_saturation_weight: float,
    alpha_saturation_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Zero-at-initialization safeguards for the implicit solve dynamics."""

    if not 0 < alpha_saturation_fraction < 1:
        raise ValueError("alpha_saturation_fraction must lie in (0, 1)")
    reference = next(model.parameters())
    gain_penalty = reference.new_zeros((), dtype=torch.float32)
    alpha_penalty = reference.new_zeros((), dtype=torch.float32)
    modules = 0
    for module in model.modules():
        if not isinstance(module, RRLSSO):
            continue
        gain, alpha = module.effective_gain_alpha()
        gain_penalty = gain_penalty + gain.float().log().square().mean()
        relative_alpha = alpha.float() / float(module.alpha_max)
        alpha_penalty = alpha_penalty + torch.relu(
            relative_alpha - alpha_saturation_fraction
        ).square().mean()
        modules += 1
    if modules:
        gain_penalty = gain_penalty / modules
        alpha_penalty = alpha_penalty / modules
    total = gain_anchor_weight * gain_penalty + alpha_saturation_weight * alpha_penalty
    return total, {"gain_anchor": gain_penalty, "alpha_saturation": alpha_penalty}


__all__ = [
    "DEIT3_OFFICIAL_RECIPES",
    "DeiT3Recipe",
    "model_size",
    "official_recipe",
    "randaugment_finetune_transform",
    "virtual_group_repeated_samples",
    "rrlsso_regularization",
    "three_augment_transform",
    "validation_transform",
]
