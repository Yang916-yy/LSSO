"""Official DeiT-III ImageNet-1K recipes and RRLSSO-only regularizers."""

from __future__ import annotations

import itertools
import math
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
    gain_reference: dict[str, torch.Tensor],
    gain_anchor_weight: float,
    alpha_saturation_weight: float,
    alpha_saturation_fraction: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gauge fixing and a logit-space solve-strength saturation barrier."""

    if not 0 < alpha_saturation_fraction < 1:
        raise ValueError("alpha_saturation_fraction must lie in (0, 1)")
    reference = next(model.parameters())
    gain_square_sum = reference.new_zeros((), dtype=torch.float32)
    alpha_square_sum = reference.new_zeros((), dtype=torch.float32)
    gain_count = alpha_count = 0
    threshold_logit = math.log(
        alpha_saturation_fraction / (1.0 - alpha_saturation_fraction)
    )
    seen_gain_keys = set()
    for name, module in model.named_modules():
        if not isinstance(module, RRLSSO):
            continue
        key = name or "<root>"
        if hasattr(module, "theta_gain"):
            if key not in gain_reference:
                raise KeyError(f"missing gain reference for RRLSSO module {key!r}")
            target = gain_reference[key].to(
                device=module.theta_gain.device, dtype=torch.float32
            )
            if target.shape != module.theta_gain.shape:
                raise ValueError(
                    f"gain reference shape mismatch for {key}: "
                    f"{tuple(target.shape)} != {tuple(module.theta_gain.shape)}"
                )
            delta = module.theta_gain.float() - target
            gain_square_sum = gain_square_sum + delta.square().sum()
            gain_count += delta.numel()
            seen_gain_keys.add(key)
        excess_logit = torch.relu(module.theta_alpha.float() - threshold_logit)
        alpha_square_sum = alpha_square_sum + excess_logit.square().sum()
        alpha_count += excess_logit.numel()
    unused = set(gain_reference) - seen_gain_keys
    if unused:
        raise KeyError(f"gain reference contains unknown RRLSSO modules: {sorted(unused)}")
    gain_penalty = gain_square_sum / max(gain_count, 1)
    alpha_penalty = alpha_square_sum / max(alpha_count, 1)
    total = gain_anchor_weight * gain_penalty + alpha_saturation_weight * alpha_penalty
    return total, {"gain_anchor": gain_penalty, "alpha_saturation": alpha_penalty}


def make_rrlsso_gain_reference(
    model: torch.nn.Module,
    *,
    anchor_to_current: bool,
    state: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Build the persistent log-gain gauge reference for one training stage."""

    modules = {
        name or "<root>": module
        for name, module in model.named_modules()
        if isinstance(module, RRLSSO) and hasattr(module, "theta_gain")
    }
    if state is not None and set(state) != set(modules):
        missing = sorted(set(modules) - set(state))
        extra = sorted(set(state) - set(modules))
        raise KeyError(f"invalid gain-reference state; missing={missing}, extra={extra}")
    result = {}
    for key, module in modules.items():
        if state is not None:
            value = state[key]
            if value.shape != module.theta_gain.shape:
                raise ValueError(
                    f"gain reference shape mismatch for {key}: "
                    f"{tuple(value.shape)} != {tuple(module.theta_gain.shape)}"
                )
            value = value.to(module.theta_gain)
        elif anchor_to_current:
            value = module.theta_gain.detach()
        else:
            value = torch.zeros_like(module.theta_gain)
        result[key] = value.detach().clone()
    return result


def rrlsso_parameter_diagnostics(
    model: torch.nn.Module,
    gain_reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Aggregate scale and solve-strength diagnostics without changing gradients."""

    theta_gain = []
    gain_delta = []
    alpha_ratio = []
    for name, module in model.named_modules():
        if not isinstance(module, RRLSSO):
            continue
        if hasattr(module, "theta_gain"):
            key = name or "<root>"
            current = module.theta_gain.detach().float()
            target = gain_reference[key].to(current)
            theta_gain.append(current.flatten())
            gain_delta.append((current - target).flatten())
        alpha_ratio.append(module.theta_alpha.detach().float().sigmoid().flatten())
    reference = next(model.parameters()).detach().new_zeros((), dtype=torch.float32)
    gains = torch.cat(theta_gain) if theta_gain else reference.reshape(1)
    deltas = torch.cat(gain_delta) if gain_delta else reference.reshape(1)
    ratios = torch.cat(alpha_ratio) if alpha_ratio else reference.reshape(1)
    return {
        "gain_log_mean": gains.mean(),
        "gain_log_std": gains.std(unbiased=False),
        "gain_anchor_rms": deltas.square().mean().sqrt(),
        "alpha_ratio_mean": ratios.mean(),
        "alpha_ratio_std": ratios.std(unbiased=False),
        "alpha_fraction_gt_080": (ratios > 0.8).float().mean(),
        "alpha_fraction_gt_095": (ratios > 0.95).float().mean(),
    }


__all__ = [
    "DEIT3_OFFICIAL_RECIPES",
    "DeiT3Recipe",
    "model_size",
    "make_rrlsso_gain_reference",
    "official_recipe",
    "randaugment_finetune_transform",
    "virtual_group_repeated_samples",
    "rrlsso_regularization",
    "rrlsso_parameter_diagnostics",
    "three_augment_transform",
    "validation_transform",
]
