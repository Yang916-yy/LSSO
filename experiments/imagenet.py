"""Distributed ImageNet-1K training with the official DeiT III recipes.

The model itself intentionally lives outside this entrypoint.  The runner calls
``integrations.timm.create_lsso_deit3`` so classification, detection, and
segmentation share one backbone implementation rather than recreating model
math in every experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
import tomllib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "configs" / "imagenet_deit3.toml"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
OFFICIAL_DEIT3_URL = (
    "https://github.com/facebookresearch/deit/blob/"
    "7e160fe43f0252d17191b71cbb5826254114ea5b/README_revenge.md"
)
IMAGENET_CHECKPOINT_FORMAT = 2


def checkpoint_contract_digest(contract: Mapping[str, Any]) -> str:
    """Return the canonical digest stored with every ImageNet checkpoint."""

    try:
        encoded = json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("ImageNet checkpoint contract is not JSON-serializable") from error
    return hashlib.sha256(encoded).hexdigest()


def validate_checkpoint_contract(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the current checkpoint envelope before any tensor is loaded."""

    if checkpoint.get("format_version") != IMAGENET_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint does not use the current ImageNet contract format")
    contract = checkpoint.get("contract")
    if not isinstance(contract, dict) or not all(
        key in contract for key in ("tier", "phase", "model", "operator", "train")
    ):
        raise ValueError("checkpoint is missing its complete ImageNet contract")
    digest = checkpoint.get("contract_digest")
    expected = checkpoint_contract_digest(contract)
    if not isinstance(digest, str) or digest != expected:
        raise ValueError("checkpoint ImageNet contract digest does not match its content")
    return contract


@dataclass(frozen=True)
class DistributedState:
    """Process-local state established by torchrun."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class ImageNetRun:
    """Fully resolved run contract, including the selected official recipe."""

    config_path: Path
    tier: str
    phase: str
    model: dict[str, Any]
    operator: dict[str, Any]
    train: dict[str, Any]
    overrides: tuple[str, ...]

    def checkpoint_contract(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "phase": self.phase,
            "model": self.model,
            "operator": self.operator,
            "train": self.train,
        }

    def checkpoint_contract_digest(self) -> str:
        return checkpoint_contract_digest(self.checkpoint_contract())

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "tier": self.tier,
            "phase": self.phase,
            "model": self.model,
            "operator": self.operator,
            "train": self.train,
            "overrides": list(self.overrides),
            "official_deit3_recipe": OFFICIAL_DEIT3_URL,
            "checkpoint_contract_digest": self.checkpoint_contract_digest(),
        }


class RepeatedAugmentationSampler(Sampler[int]):
    """DeiT's three-view distributed sampler, retained verbatim in behavior."""

    def __init__(
        self,
        dataset: Dataset[Any],
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool = True,
        num_repeats: int = 3,
    ) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if num_repeats < 1:
            raise ValueError("num_repeats must be positive")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        # The 256-image truncation is part of the published DeiT sampler.
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator)
        else:
            indices = torch.arange(len(self.dataset))

        indices = torch.repeat_interleave(indices, repeats=self.num_repeats).tolist()
        padding = self.total_size - len(indices)
        if padding > 0:
            indices += indices[:padding]
        if len(indices) != self.total_size:
            raise RuntimeError("repeated-augmentation sampler produced an invalid size")

        local_indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(local_indices) != self.num_samples:
            raise RuntimeError("repeated-augmentation sampler produced an invalid shard")
        return iter(local_indices[: self.num_selected_samples])

    def __len__(self) -> int:
        return self.num_selected_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


class _GaussianBlur:
    def __init__(self, *, probability: float = 1.0, radius_min: float = 0.1, radius_max: float = 2.0) -> None:
        self.probability = probability
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image: Any) -> Any:
        from PIL import ImageFilter

        if random.random() > self.probability:
            return image
        return image.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class _Solarization:
    def __init__(self, *, probability: float = 1.0) -> None:
        self.probability = probability

    def __call__(self, image: Any) -> Any:
        from PIL import ImageOps

        if random.random() < self.probability:
            return ImageOps.solarize(image)
        return image


class _GrayScale:
    def __init__(self, *, probability: float = 1.0) -> None:
        from torchvision import transforms

        self.probability = probability
        self.transform = transforms.Grayscale(num_output_channels=3)

    def __call__(self, image: Any) -> Any:
        if random.random() < self.probability:
            return self.transform(image)
        return image


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSSO DeiT III S/B/L on ImageNet-1K with torchrun."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tier", choices=("small", "base", "large"), required=True)
    parser.add_argument(
        "--phase",
        choices=("pretrain", "finetune_224"),
        default="pretrain",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="ImageNet root containing train/ and val/ ImageFolder directories.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--implementation",
        choices=("cuda", "reference"),
        help="Override the configured LSSO implementation for a diagnostic run.",
    )
    parser.add_argument("--resume", type=Path, help="Resume an exact previous run.")
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Required 192px pretraining checkpoint for B/L 224px fine-tuning.",
    )
    parser.add_argument(
        "--allow-lamb-fallback",
        action="store_true",
        help="Use timm's non-fused LAMB only when Apex FusedLAMB is unavailable.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the official duration for a diagnostic run; recorded as non-canonical.",
    )
    parser.add_argument("--batch-size", type=int, help="Override the per-GPU batch size.")
    parser.add_argument("--workers", type=int, help="Override ImageFolder worker count.")
    parser.add_argument("--seed", type=int, help="Override the official seed.")
    parser.add_argument("--save-every", type=int, help="Checkpoint interval in epochs.")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval", action="store_true", help="Evaluate --resume without training.")
    return parser.parse_args(argv)


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    return value


def _require_keys(values: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing)}")


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = _positive_float(value, name) if value != 0 else 0.0
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def load_run(args: argparse.Namespace) -> ImageNetRun:
    config_path = args.config.resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    defaults = dict(_as_mapping(raw.get("defaults"), "[defaults]"))
    operator = dict(_as_mapping(raw.get("operator"), "[operator]"))
    tiers = _as_mapping(raw.get("tiers"), "[tiers]")
    tier_values = dict(_as_mapping(tiers.get(args.tier), f"[tiers.{args.tier}]"))
    phases = {
        key: value for key, value in tier_values.items() if isinstance(value, dict)
    }
    phase_values = phases.get(args.phase)
    if phase_values is None:
        available = sorted(phases)
        raise ValueError(
            f"tier {args.tier!r} does not define phase {args.phase!r}; "
            f"available phases: {', '.join(available) or 'none'}"
        )
    phase_values = dict(_as_mapping(phase_values, f"[tiers.{args.tier}.{args.phase}]"))
    tier_values = {
        key: value for key, value in tier_values.items() if not isinstance(value, dict)
    }

    model = {
        "image_size": phase_values["input_size"],
        "patch_size": defaults["patch_size"],
        "num_classes": defaults["num_classes"],
        "mlp_ratio": defaults["mlp_ratio"],
        "layer_scale_init_value": defaults["layer_scale_init_value"],
        "norm_eps": defaults["norm_eps"],
        **tier_values,
    }
    train = {**defaults, **phase_values}
    train.pop("input_size")
    overrides: list[str] = []
    for argument, key in (
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.workers, "workers"),
        (args.seed, "seed"),
        (args.save_every, "save_every"),
    ):
        if argument is not None:
            train[key] = argument
            overrides.append(key)
    if args.implementation is not None:
        operator["implementation"] = args.implementation
        overrides.append("implementation")

    _validate_run(args.tier, args.phase, model, operator, train)
    if args.phase == "finetune_224" and args.init_checkpoint is None and args.resume is None:
        raise ValueError("--phase finetune_224 requires --init-checkpoint on a new run")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if args.eval and args.resume is None:
        raise ValueError("--eval requires --resume")

    return ImageNetRun(
        config_path=config_path,
        tier=args.tier,
        phase=args.phase,
        model=model,
        operator=operator,
        train=train,
        overrides=tuple(overrides),
    )


def _validate_run(
    tier: str,
    phase: str,
    model: Mapping[str, Any],
    operator: Mapping[str, Any],
    train: Mapping[str, Any],
) -> None:
    _require_keys(
        model,
        (
            "image_size",
            "patch_size",
            "num_classes",
            "embed_dim",
            "depth",
            "num_heads",
            "rank",
            "mlp_ratio",
            "layer_scale_init_value",
            "norm_eps",
            "drop_path_rate",
        ),
        f"model contract for {tier}",
    )
    _require_keys(
        operator,
        ("core_mode", "rank_rotary", "bias", "implementation"),
        "[operator]",
    )
    _require_keys(
        train,
        (
            "epochs",
            "batch_size",
            "lr",
            "min_lr",
            "warmup_lr",
            "warmup_epochs",
            "weight_decay",
            "workers",
            "eval_crop_ratio",
            "mixup",
            "cutmix",
            "mixup_prob",
            "mixup_switch_prob",
            "mixup_mode",
            "ema",
            "ema_decay",
            "amp_dtype",
            "save_every",
            "optimizer",
            "augmentation",
            "color_jitter",
            "auto_augment",
            "repeated_aug",
            "bce_loss",
            "label_smoothing",
        ),
        f"training contract for {tier}/{phase}",
    )

    image_size = _positive_int(model["image_size"], "image_size")
    patch_size = _positive_int(model["patch_size"], "patch_size")
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")
    embed_dim = _positive_int(model["embed_dim"], "embed_dim")
    heads = _positive_int(model["num_heads"], "num_heads")
    if embed_dim % heads:
        raise ValueError("embed_dim must be divisible by num_heads")
    _positive_int(model["depth"], "depth")
    _positive_int(model["rank"], "rank")
    _positive_int(model["num_classes"], "num_classes")
    _positive_float(model["mlp_ratio"], "mlp_ratio")
    _positive_float(model["layer_scale_init_value"], "layer_scale_init_value")
    _positive_float(model["norm_eps"], "norm_eps")
    _probability(model["drop_path_rate"], "drop_path_rate")

    if operator["core_mode"] != "dynamic":
        raise ValueError("the ImageNet recipe requires the DYNAMIC LSSO core")
    if operator["rank_rotary"] is not True:
        raise ValueError("the ImageNet recipe requires Rank-Rotary")
    if operator["bias"] is not True:
        raise ValueError("the DeiT III scaffold requires qkv/projection bias")
    if operator["implementation"] not in {"cuda", "reference"}:
        raise ValueError("operator.implementation must be 'cuda' or 'reference'")

    _positive_int(train["epochs"], "epochs")
    _positive_int(train["batch_size"], "batch_size")
    _positive_int(train["workers"], "workers")
    _nonnegative_int(train["warmup_epochs"], "warmup_epochs")
    _positive_int(train["save_every"], "save_every")
    _positive_float(train["lr"], "lr")
    _positive_float(train["warmup_lr"], "warmup_lr")
    _positive_float(train["min_lr"], "min_lr")
    _positive_float(train["weight_decay"], "weight_decay")
    _probability(train["eval_crop_ratio"], "eval_crop_ratio")
    _probability(train["mixup"], "mixup")
    _probability(train["cutmix"], "cutmix")
    _probability(train["mixup_prob"], "mixup_prob")
    _probability(train["mixup_switch_prob"], "mixup_switch_prob")
    _probability(train["label_smoothing"], "label_smoothing")
    _probability(train["ema_decay"], "ema_decay")
    if train["mixup_mode"] != "batch":
        raise ValueError("the published DeiT III recipes use batch-mode Mixup/CutMix")
    if train["amp_dtype"] != "float16":
        raise ValueError("the CUDA LSSO path requires train.amp_dtype = 'float16'")
    if train["optimizer"] not in {"fusedlamb", "adamw"}:
        raise ValueError("optimizer must be 'fusedlamb' or 'adamw'")
    if train["augmentation"] not in {"three_augment", "rand_augment"}:
        raise ValueError("augmentation must be 'three_augment' or 'rand_augment'")
    if not isinstance(train["repeated_aug"], bool):
        raise ValueError("repeated_aug must be boolean")
    if not isinstance(train["bce_loss"], bool):
        raise ValueError("bce_loss must be boolean")
    if not isinstance(train["ema"], bool) or train["ema"] is not True:
        raise ValueError("the published DeiT III recipe requires EMA")

    expected = {
        "small": (384, 12, 6, 32),
        "base": (768, 12, 12, 48),
        "large": (1024, 24, 16, 64),
    }[tier]
    actual = (model["embed_dim"], model["depth"], model["num_heads"], model["rank"])
    if actual != expected:
        raise ValueError(f"{tier} geometry must be {expected}, got {actual}")
    if phase == "pretrain" and train["optimizer"] != "fusedlamb":
        raise ValueError("the 800-epoch pretraining phase requires FusedLAMB")
    if phase == "finetune_224":
        if tier == "small":
            raise ValueError("DeiT III does not define a 224px fine-tuning phase for small")
        if model["image_size"] != 224 or train["optimizer"] != "adamw":
            raise ValueError("the B/L 224px fine-tuning phase requires 224px AdamW")


def initialize_distributed() -> DistributedState:
    keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    present = [key for key in keys if key in os.environ]
    if present and len(present) != len(keys):
        missing = sorted(set(keys) - set(present))
        raise RuntimeError(
            "incomplete torchrun environment; missing " + ", ".join(missing)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("ImageNet DeiT III training requires a CUDA device")

    if not present:
        return DistributedState(
            rank=0,
            world_size=1,
            local_rank=torch.cuda.current_device(),
            device=torch.device("cuda", torch.cuda.current_device()),
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size < 1 or not 0 <= local_rank < torch.cuda.device_count():
        raise RuntimeError("torchrun rank/world-size/device configuration is invalid")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
    )


def finalize_distributed(state: DistributedState) -> None:
    if state.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _barrier(state: DistributedState) -> None:
    if state.enabled:
        dist.barrier()


def seed_everything(seed: int, state: DistributedState) -> None:
    process_seed = int(seed) + state.rank
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)
    try:
        import numpy as np

        np.random.seed(process_seed)
    except ImportError:
        pass


def _seed_worker(_: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def build_three_augment(image_size: int, color_jitter: float) -> Any:
    """Reproduce DeiT III's public ``augment.py`` 3-Augment transform."""

    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    transforms_list: list[Any] = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomChoice(
            [_GrayScale(), _Solarization(), _GaussianBlur()]
        ),
    ]
    if color_jitter > 0:
        transforms_list.append(
            transforms.ColorJitter(color_jitter, color_jitter, color_jitter)
        )
    transforms_list.extend(
        (
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )
    return transforms.Compose(transforms_list)


def build_train_transform(run: ImageNetRun) -> Any:
    train = run.train
    image_size = int(run.model["image_size"])
    if train["augmentation"] == "three_augment":
        return build_three_augment(image_size, float(train["color_jitter"]))

    from timm.data import create_transform

    return create_transform(
        input_size=image_size,
        is_training=True,
        color_jitter=(None if float(train["color_jitter"]) == 0 else float(train["color_jitter"])),
        auto_augment=str(train["auto_augment"]),
        interpolation="bicubic",
        re_prob=float(train["reprob"]),
        re_mode="pixel",
        re_count=1,
    )


def build_eval_transform(run: ImageNetRun) -> Any:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    image_size = int(run.model["image_size"])
    resize_size = int(image_size / float(run.train["eval_crop_ratio"]))
    return transforms.Compose(
        (
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        )
    )


def build_loaders(
    run: ImageNetRun,
    data_root: Path,
    state: DistributedState,
) -> tuple[DataLoader[Any], DataLoader[Any], Sampler[int]]:
    from torchvision.datasets import ImageFolder

    root = data_root.resolve()
    train_root = root / "train"
    val_root = root / "val"
    if not train_root.is_dir() or not val_root.is_dir():
        raise FileNotFoundError(
            f"expected ImageNet ImageFolder directories {train_root} and {val_root}"
        )
    train_dataset = ImageFolder(train_root, transform=build_train_transform(run))
    val_dataset = ImageFolder(val_root, transform=build_eval_transform(run))
    if len(train_dataset.classes) != int(run.model["num_classes"]):
        raise ValueError(
            f"ImageNet train directory has {len(train_dataset.classes)} classes, "
            f"expected {run.model['num_classes']}"
        )
    if len(val_dataset.classes) != len(train_dataset.classes):
        raise ValueError("ImageNet train/val class directory sets do not agree")

    if bool(run.train["repeated_aug"]):
        train_sampler: Sampler[int] = RepeatedAugmentationSampler(
            train_dataset,
            num_replicas=state.world_size,
            rank=state.rank,
            shuffle=True,
            num_repeats=3,
        )
    elif state.enabled:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=state.world_size,
            rank=state.rank,
            shuffle=True,
        )
    else:
        train_sampler = RandomSampler(train_dataset)

    # The public DeiT command does not enable --dist-eval, so every process
    # evaluates the full validation set and reductions preserve the same metric.
    val_sampler: Sampler[int] = SequentialSampler(val_dataset)
    workers = int(run.train["workers"])
    generator = torch.Generator()
    generator.manual_seed(int(run.train["seed"]) + state.rank)
    common = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(run.train["batch_size"]),
        sampler=train_sampler,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, int(1.5 * int(run.train["batch_size"]))),
        sampler=val_sampler,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler


def prepare_operator_backend(run: ImageNetRun, device: torch.device) -> None:
    if run.operator["implementation"] == "cuda":
        from lsso.ball import cuda

        cuda.load(device=device)


def build_model(run: ImageNetRun) -> nn.Module:
    """Call the shared DeiT III LSSO backbone factory owned by integrations."""

    try:
        from integrations.timm import create_lsso_deit3
    except ImportError as error:
        raise RuntimeError(
            "ImageNet training requires integrations.timm.create_lsso_deit3; "
            "the shared DeiT III backbone adapter is not available."
        ) from error

    model = run.model
    return create_lsso_deit3(
        image_size=int(model["image_size"]),
        patch_size=int(model["patch_size"]),
        num_classes=int(model["num_classes"]),
        embed_dim=int(model["embed_dim"]),
        depth=int(model["depth"]),
        num_heads=int(model["num_heads"]),
        rank=int(model["rank"]),
        mlp_ratio=float(model["mlp_ratio"]),
        core_mode=str(run.operator["core_mode"]),
        rank_rotary=bool(run.operator["rank_rotary"]),
        bias=bool(run.operator["bias"]),
        implementation=str(run.operator["implementation"]),
        drop_path_rate=float(model["drop_path_rate"]),
        layer_scale_init_value=float(model["layer_scale_init_value"]),
        norm_eps=float(model["norm_eps"]),
        no_embed_class=True,
    )


def _checkpoint_model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("checkpoint must contain a tensor state dict under 'model'")
    return copy.deepcopy(state)


def _position_extra_tokens(tokens: int) -> int:
    for extra in (0, 1, 2):
        side = math.isqrt(tokens - extra)
        if side * side == tokens - extra:
            return extra
    raise ValueError(f"position embedding length {tokens} does not encode a square patch grid")


def interpolate_position_embedding(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Bicubically resize a learned 2D patch table for DeiT III fine-tuning."""

    if source.ndim != 3 or target.ndim != 3 or source.shape[0] != target.shape[0]:
        raise ValueError("position embeddings must have shape [batch, tokens, channels]")
    if source.shape[-1] != target.shape[-1]:
        raise ValueError("position embeddings must have the same channel dimension")
    source_extra = _position_extra_tokens(source.shape[1])
    target_extra = _position_extra_tokens(target.shape[1])
    if source_extra != target_extra:
        raise ValueError("source and target position embeddings disagree on extra tokens")
    source_side = math.isqrt(source.shape[1] - source_extra)
    target_side = math.isqrt(target.shape[1] - target_extra)
    extra = source[:, :source_extra]
    patch = source[:, source_extra:]
    patch = patch.reshape(
        source.shape[0], source_side, source_side, source.shape[-1]
    ).permute(0, 3, 1, 2)
    patch = functional.interpolate(
        patch.float(),
        size=(target_side, target_side),
        mode="bicubic",
        align_corners=False,
    )
    patch = patch.permute(0, 2, 3, 1).reshape(
        source.shape[0], target_side * target_side, -1
    )
    result = torch.cat((extra.float(), patch), dim=1).to(dtype=target.dtype)
    return result


def load_finetune_checkpoint(model: nn.Module, path: Path, run: ImageNetRun) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("fine-tuning checkpoint must be a mapping")
    checkpoint_contract = validate_checkpoint_contract(checkpoint)
    source_tier = checkpoint_contract["tier"]
    if source_tier != run.tier:
        raise ValueError(
            f"fine-tuning checkpoint tier is {source_tier!r}, expected {run.tier!r}"
        )
    if checkpoint_contract["phase"] != "pretrain":
        raise ValueError("fine-tuning checkpoint must originate from ImageNet pretraining")
    source_model = checkpoint_contract["model"]
    if not isinstance(source_model, dict) or not isinstance(
        checkpoint_contract["operator"], dict
    ):
        raise ValueError("fine-tuning checkpoint has an invalid ImageNet model contract")
    shared_model_keys = (
        "patch_size",
        "num_classes",
        "mlp_ratio",
        "layer_scale_init_value",
        "norm_eps",
        "embed_dim",
        "depth",
        "num_heads",
        "rank",
        "drop_path_rate",
    )
    if any(source_model.get(key) != run.model.get(key) for key in shared_model_keys):
        raise ValueError("fine-tuning checkpoint model contract does not match the run")
    if checkpoint_contract["operator"] != run.operator:
        raise ValueError("fine-tuning checkpoint operator contract does not match the run")
    source = _checkpoint_model_state(checkpoint)
    destination = model.state_dict()
    source_position_key = _position_embedding_key(source)
    destination_position_key = _position_embedding_key(destination)
    if source_position_key is not None and destination_position_key is not None:
        if source[source_position_key].shape != destination[destination_position_key].shape:
            source[source_position_key] = interpolate_position_embedding(
                source[source_position_key], destination[destination_position_key]
            )
    incompatible = model.load_state_dict(source, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "fine-tuning checkpoint does not match the current model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


def _position_embedding_key(state: Mapping[str, torch.Tensor]) -> str | None:
    keys = [key for key in state if key == "pos_embed" or key.endswith(".pos_embed")]
    if not keys:
        return None
    if len(keys) != 1:
        raise ValueError(f"model has ambiguous learned position embeddings: {keys}")
    return keys[0]


def _no_weight_decay(model: nn.Module) -> set[str]:
    parameter_names = {name for name, _ in model.named_parameters()}
    candidate = getattr(model, "no_weight_decay", None)
    if candidate is None:
        # The shared wrapper delegates to timm's ViT encoder, so retain the
        # published no-decay treatment without teaching the experiment about
        # any other model parameters.
        return {
            name
            for name in parameter_names
            if name == "pos_embed"
            or name.endswith(".pos_embed")
            or name == "cls_token"
            or name.endswith(".cls_token")
        }
    names = candidate()
    if not isinstance(names, (set, frozenset)) or not all(
        isinstance(name, str) for name in names
    ):
        raise TypeError("model.no_weight_decay() must return a set of parameter names")
    resolved: set[str] = set()
    for name in names:
        if name in parameter_names:
            resolved.add(name)
            continue
        matches = [candidate for candidate in parameter_names if candidate.endswith(f".{name}")]
        if len(matches) != 1:
            raise ValueError(f"model.no_weight_decay() returned unknown name {name!r}")
        resolved.add(matches[0])
    return resolved


def build_optimizer(
    model: nn.Module,
    run: ImageNetRun,
    *,
    allow_lamb_fallback: bool,
) -> tuple[torch.optim.Optimizer, str]:
    from timm.optim import Lamb, param_groups_weight_decay

    groups = param_groups_weight_decay(
        model,
        weight_decay=float(run.train["weight_decay"]),
        no_weight_decay_list=_no_weight_decay(model),
    )
    lr = float(run.train["lr"])
    if run.train["optimizer"] == "adamw":
        return (
            torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1.0e-8),
            "torch.adamw",
        )

    try:
        from apex.optimizers import FusedLAMB

        return (
            FusedLAMB(
                groups,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1.0e-8,
                adam_w_mode=True,
            ),
            "apex.fused_lamb",
        )
    except ImportError as error:
        if not allow_lamb_fallback:
            raise RuntimeError(
                "the canonical DeiT III pretraining recipe requires Apex FusedLAMB; "
                "install Apex or pass --allow-lamb-fallback to record a non-fused LAMB run"
            ) from error
        return (
            Lamb(
                groups,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1.0e-6,
                max_grad_norm=1.0,
            ),
            "timm.lamb (explicit fallback)",
        )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    run: ImageNetRun,
) -> Any:
    from timm.scheduler import CosineLRScheduler

    return CosineLRScheduler(
        optimizer,
        t_initial=int(run.train["epochs"]),
        lr_min=float(run.train["min_lr"]),
        cycle_mul=1.0,
        cycle_decay=1.0,
        cycle_limit=1,
        warmup_t=int(run.train["warmup_epochs"]),
        warmup_lr_init=float(run.train["warmup_lr"]),
        warmup_prefix=False,
        t_in_epochs=True,
    )


def build_mixup_and_loss(run: ImageNetRun) -> tuple[Any | None, nn.Module]:
    from timm.data import Mixup
    from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

    mixup_active = (
        float(run.train["mixup"]) > 0
        or float(run.train["cutmix"]) > 0
    )
    mixup = None
    if mixup_active:
        mixup = Mixup(
            mixup_alpha=float(run.train["mixup"]),
            cutmix_alpha=float(run.train["cutmix"]),
            cutmix_minmax=None,
            prob=float(run.train["mixup_prob"]),
            switch_prob=float(run.train["mixup_switch_prob"]),
            mode=str(run.train["mixup_mode"]),
            label_smoothing=float(run.train["label_smoothing"]),
            num_classes=int(run.model["num_classes"]),
        )
    if bool(run.train["bce_loss"]):
        return mixup, nn.BCEWithLogitsLoss()
    if mixup_active:
        return mixup, SoftTargetCrossEntropy()
    if float(run.train["label_smoothing"]) > 0:
        return mixup, LabelSmoothingCrossEntropy(
            smoothing=float(run.train["label_smoothing"])
        )
    return mixup, nn.CrossEntropyLoss()


def build_ema(model: nn.Module, run: ImageNetRun) -> Any:
    from timm.utils import ModelEma

    return ModelEma(model, decay=float(run.train["ema_decay"]), device="", resume="")


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _reduce_metrics(
    values: tuple[float, int, int, int], state: DistributedState
) -> tuple[float, float, float]:
    loss_sum, correct1, correct5, count = values
    packed = torch.tensor(
        (loss_sum, float(correct1), float(correct5), float(count)),
        device=state.device,
        dtype=torch.float64,
    )
    if state.enabled:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    if packed[3].item() == 0:
        raise RuntimeError("no samples were processed")
    return (
        packed[0].item() / packed[3].item(),
        100.0 * packed[1].item() / packed[3].item(),
        100.0 * packed[2].item() / packed[3].item(),
    )


def _autocast() -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    sampler: Sampler[int],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    mixup: Any | None,
    ema: Any,
    *,
    epoch: int,
    state: DistributedState,
    run: ImageNetRun,
    print_freq: int,
) -> tuple[float, float, float]:
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)  # type: ignore[union-attr]
    model.train()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    count = 0
    started = time.perf_counter()
    for step, (images, targets) in enumerate(loader):
        images = images.to(state.device, non_blocking=True)
        targets = targets.to(state.device, non_blocking=True)
        metric_targets = targets
        if mixup is not None:
            images, targets = mixup(images, targets)
        if bool(run.train["bce_loss"]):
            targets = targets.gt(0.0).to(dtype=targets.dtype)

        optimizer.zero_grad(set_to_none=True)
        with _autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch + 1}, step {step + 1}"
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        batch = images.shape[0]
        loss_sum += float(loss.detach()) * batch
        if metric_targets.ndim == 1:
            correct1 += int((logits.detach().argmax(dim=1) == metric_targets).sum())
            correct5 += int(
                logits.detach().topk(k=min(5, logits.shape[1]), dim=1).indices.eq(
                    metric_targets[:, None]
                ).any(dim=1).sum()
            )
        count += batch
        if state.is_main and print_freq > 0 and (step + 1) % print_freq == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "train_step",
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "steps": len(loader),
                        "loss": float(loss.detach()),
                        "lr": optimizer.param_groups[0]["lr"],
                        "images_per_second": count / max(elapsed, 1.0e-9),
                    }
                ),
                flush=True,
            )
    return _reduce_metrics((loss_sum, correct1, correct5, count), state)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    state: DistributedState,
) -> tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    count = 0
    for images, targets in loader:
        images = images.to(state.device, non_blocking=True)
        targets = targets.to(state.device, non_blocking=True)
        with _autocast():
            logits = model(images)
            loss = criterion(logits, targets)
        batch = images.shape[0]
        loss_sum += float(loss) * batch
        correct1 += int((logits.argmax(dim=1) == targets).sum())
        correct5 += int(
            logits.topk(k=min(5, logits.shape[1]), dim=1).indices.eq(
                targets[:, None]
            ).any(dim=1).sum()
        )
        count += batch
    return _reduce_metrics((loss_sum, correct1, correct5, count), state)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _record_runtime_metadata(
    output: Path,
    *,
    parameter_count: int,
    resolved_optimizer: str,
    run: ImageNetRun,
    allow_lamb_fallback: bool,
) -> None:
    path = output / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object")
    metadata["model"] = {"trainable_parameters": parameter_count}
    metadata["optimizer_resolved"] = resolved_optimizer
    metadata["lamb_fallback_permitted"] = allow_lamb_fallback
    metadata["recipe_fidelity"] = _recipe_fidelity(
        run,
        resolved_optimizer=resolved_optimizer,
        allow_lamb_fallback=allow_lamb_fallback,
    )
    _atomic_json(path, metadata)


def _recipe_fidelity(
    run: ImageNetRun,
    *,
    resolved_optimizer: str,
    allow_lamb_fallback: bool,
) -> str:
    if (
        not run.overrides
        and not allow_lamb_fallback
        and (
            resolved_optimizer == "apex.fused_lamb"
            or run.train["optimizer"] == "adamw"
        )
    ):
        return "deit3-derived"
    return "explicitly-modified"


def _prepare_output(
    output: Path,
    args: argparse.Namespace,
    run: ImageNetRun,
    state: DistributedState,
) -> Path:
    output = output.resolve()
    failure: str | None = None
    if state.is_main:
        try:
            output.mkdir(parents=True, exist_ok=True)
            if args.resume is None and any(output.iterdir()):
                raise FileExistsError(
                    f"refusing to overwrite a non-empty output directory: {output}"
                )
            metadata = {
                "event": "start",
                "run": run.as_dict(),
                "launcher": {
                    "world_size": state.world_size,
                    "rank": state.rank,
                    "local_rank": state.local_rank,
                },
                "environment": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(state.device),
                    "git_revision": _git_revision(),
                },
                "args": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
            }
            try:
                import timm

                metadata["environment"]["timm"] = timm.__version__
            except ImportError:
                metadata["environment"]["timm"] = None
            _atomic_json(output / "metadata.json", metadata)
        except Exception as error:  # Broadcast rank-zero setup failures to torchrun peers.
            failure = f"{type(error).__name__}: {error}"
    if state.enabled:
        payload: list[str | None] = [failure]
        dist.broadcast_object_list(payload, src=0)
        failure = payload[0]
    if failure is not None:
        raise RuntimeError(f"unable to initialize output directory: {failure}")
    _barrier(state)
    return output


def _checkpoint(
    *,
    epoch: int,
    model: nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    run: ImageNetRun,
    best_acc1: float,
) -> dict[str, Any]:
    contract = run.checkpoint_contract()
    return {
        "format_version": IMAGENET_CHECKPOINT_FORMAT,
        "epoch": epoch,
        "best_acc1": best_acc1,
        "contract": contract,
        "contract_digest": checkpoint_contract_digest(contract),
        "model": _unwrap_model(model).state_dict(),
        "model_ema": ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }


def _load_resume(
    path: Path,
    *,
    model: nn.Module,
    ema: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    run: ImageNetRun,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint must be a mapping")
    if validate_checkpoint_contract(checkpoint) != run.checkpoint_contract():
        raise ValueError("resume checkpoint contract does not match the requested run")
    _unwrap_model(model).load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
    ema_state = checkpoint.get("model_ema")
    if not isinstance(ema_state, dict):
        raise ValueError("resume checkpoint is missing model_ema")
    ema.module.load_state_dict(ema_state, strict=True)
    for key, owner in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        state = checkpoint.get(key)
        if not isinstance(state, dict):
            raise ValueError(f"resume checkpoint is missing {key}")
        owner.load_state_dict(state)
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("resume checkpoint has an invalid epoch")
    best_acc1 = checkpoint.get("best_acc1")
    if not isinstance(best_acc1, (float, int)):
        raise ValueError("resume checkpoint has an invalid best_acc1")
    return epoch + 1, float(best_acc1)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run = load_run(args)
    state = initialize_distributed()
    try:
        seed_everything(int(run.train["seed"]), state)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        prepare_operator_backend(run, state.device)
        output = _prepare_output(args.output, args, run, state)
        train_loader, val_loader, train_sampler = build_loaders(
            run, args.data_root, state
        )
        model = build_model(run)
        if args.init_checkpoint is not None:
            load_finetune_checkpoint(model, args.init_checkpoint, run)
        model.to(state.device)
        ema = build_ema(model, run)
        model_without_ddp = model
        if state.enabled:
            model = DistributedDataParallel(model, device_ids=[state.local_rank])

        optimizer, resolved_optimizer = build_optimizer(
            model_without_ddp,
            run,
            allow_lamb_fallback=args.allow_lamb_fallback,
        )
        scheduler = build_scheduler(optimizer, run)
        scaler = torch.amp.GradScaler("cuda")
        mixup, criterion = build_mixup_and_loss(run)
        criterion.to(state.device)
        start_epoch = 0
        best_acc1 = float("-inf")
        if args.resume is not None:
            start_epoch, best_acc1 = _load_resume(
                args.resume,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                run=run,
            )
            # This matches the public DeiT entrypoint after restoring scheduler state.
            scheduler.step(start_epoch)

        parameter_count = sum(
            parameter.numel() for parameter in model_without_ddp.parameters() if parameter.requires_grad
        )
        if state.is_main:
            _record_runtime_metadata(
                output,
                parameter_count=parameter_count,
                resolved_optimizer=resolved_optimizer,
                run=run,
                allow_lamb_fallback=args.allow_lamb_fallback,
            )
            print(
                json.dumps(
                    {
                        "event": "start",
                        "tier": run.tier,
                        "phase": run.phase,
                        "parameters": parameter_count,
                        "train_samples": len(train_loader.dataset),
                        "val_samples": len(val_loader.dataset),
                        "steps_per_epoch": len(train_loader),
                        "optimizer": resolved_optimizer,
                        "world_size": state.world_size,
                    }
                ),
                flush=True,
            )
        if args.eval:
            val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, state=state)
            if state.is_main:
                record = {
                    "event": "evaluation",
                    "checkpoint": str(args.resume),
                    "val_loss": val_loss,
                    "val_acc1": val_acc1,
                    "val_acc5": val_acc5,
                }
                _append_jsonl(output / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            return

        if start_epoch >= int(run.train["epochs"]):
            if state.is_main:
                print(json.dumps({"event": "complete", "already_complete": True}), flush=True)
            return
        for epoch in range(start_epoch, int(run.train["epochs"])):
            epoch_started = time.perf_counter()
            train_loss, train_acc1, train_acc5 = train_epoch(
                model,
                train_loader,
                train_sampler,
                criterion,
                optimizer,
                scaler,
                mixup,
                ema,
                epoch=epoch,
                state=state,
                run=run,
                print_freq=args.print_freq,
            )
            # timm's epoch scheduler receives the next epoch at the end of
            # this one. The constructor already provides epoch zero's 1e-6 LR.
            scheduler.step(epoch + 1)
            val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, state=state)
            record = {
                "event": "epoch",
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc5": train_acc5,
                "val_loss": val_loss,
                "val_acc1": val_acc1,
                "val_acc5": val_acc5,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - epoch_started,
            }
            if state.is_main:
                is_best = val_acc1 > best_acc1
                if is_best:
                    best_acc1 = val_acc1
                checkpoint = _checkpoint(
                    epoch=epoch,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    run=run,
                    best_acc1=best_acc1,
                )
                if (epoch + 1) % int(run.train["save_every"]) == 0:
                    torch.save(checkpoint, output / "checkpoint_last.pt")
                if is_best:
                    torch.save(checkpoint, output / "checkpoint_best.pt")
                record["best_val_acc1"] = best_acc1
                _append_jsonl(output / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)
            _barrier(state)
        if state.is_main:
            print(json.dumps({"event": "complete", "best_val_acc1": best_acc1}), flush=True)
    finally:
        finalize_distributed(state)


if __name__ == "__main__":
    main()
