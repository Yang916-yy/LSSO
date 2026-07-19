"""Single-GPU DeiT-III/RRLSSO ImageNet-1K training over WebDataset shards.

The stage profiles reproduce Meta's per-size DeiT-III recipes: Small is
pre-trained at 224px, while Base/Large are pre-trained at 192px and optionally
refined at 224px or 384px.  Every stage saves atomic ``last.pt`` and ``best.pt``
checkpoints and can resume at epoch boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from timm import create_model
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from timm.utils import ModelEmaV2
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import examples.models  # noqa: E402,F401  registers project models
from examples.models.deit3_rrlsso import DEIT3_RRLSSO_MODELS  # noqa: E402
from experiments.deit3_official_recipe import (  # noqa: E402
    make_rrlsso_gain_reference,
    official_recipe,
    randaugment_finetune_transform,
    rrlsso_regularization,
    rrlsso_parameter_diagnostics,
    three_augment_transform,
    validation_transform,
    virtual_group_repeated_samples,
)
from lsso.mathdx_backend import is_mathdx_available, mathdx_load_error  # noqa: E402

TRAIN_SAMPLES = 1_281_167
VAL_SAMPLES = 50_000
TRAIN_SHARDS = 1024
VAL_SHARDS = 64
CHECKPOINT_SCHEMA = 2


def shard_commands(split: str, cache_dir: Path, repo: str) -> list[str]:
    count = TRAIN_SHARDS if split == "train" else VAL_SHARDS
    prefix = "train" if split == "train" else "validation"
    digits = 4 if split == "train" else 2
    helper = ROOT / "tools" / "hf_wds_stream.py"
    return [
        "pipe:"
        + " ".join(
            (
                shlex.quote(sys.executable),
                shlex.quote(str(helper)),
                "--repo",
                shlex.quote(repo),
                "--filename",
                shlex.quote(f"imagenet1k-{prefix}-{index:0{digits}d}.tar"),
                "--cache-dir",
                shlex.quote(str(cache_dir)),
            )
        )
        for index in range(count)
    ]


def local_webdataset(
    shards: list[str], *, shardshuffle: int | bool, seed: int
) -> wds.DataPipeline:
    pipeline: list[object] = [wds.SimpleShardList(shards, seed=seed)]
    if shardshuffle:
        pipeline.append(wds.detshuffle(int(shardshuffle), seed=seed))
    pipeline.extend(
        [wds.split_by_worker, wds.tarfile_to_samples(handler=wds.reraise_exception)]
    )
    return wds.DataPipeline(*pipeline)


def local_shards(root: Path, split: str) -> list[str]:
    manifest = root / "dataset.json"
    if not manifest.is_file():
        raise RuntimeError(f"local WebDataset is incomplete; missing {manifest}")
    state = json.loads(manifest.read_text(encoding="utf-8"))
    expected = TRAIN_SAMPLES if split == "train" else VAL_SAMPLES
    if int(state[f"{split}_samples"]) != expected:
        raise RuntimeError(f"invalid {split} sample count in {manifest}")
    shards = sorted(str(path) for path in root.glob(f"imagenet1k-{split}-*.tar"))
    if not shards:
        raise RuntimeError(f"no local {split} shards under {root}")
    return shards


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = (
        three_augment_transform(args.image_size)
        if args.augmentation == "three_augment"
        else randaugment_finetune_transform(args.image_size)
    )
    val_transform = validation_transform(args.image_size)
    if args.local_wds_dir:
        local_root = Path(args.local_wds_dir)
        train_shards = local_shards(local_root, "train")
        val_shards = local_shards(local_root, "validation")
        factory = local_webdataset
    else:
        cache_dir = Path(args.cache_dir)
        train_shards = shard_commands("train", cache_dir, args.hf_repo)
        val_shards = shard_commands("validation", cache_dir, args.hf_repo)

        def factory(shards, *, shardshuffle, seed):
            return wds.WebDataset(
                shards,
                resampled=False,
                shardshuffle=shardshuffle,
                detshuffle=True,
                seed=seed,
                handler=wds.reraise_exception,
            )

    if args.shard_limit:
        train_shards = train_shards[: args.shard_limit]
        val_shards = val_shards[: args.shard_limit]
    train = factory(
        train_shards,
        shardshuffle=min(100, len(train_shards)) if len(train_shards) > 1 else False,
        seed=args.seed,
    ).compose(
        wds.detshuffle(args.shuffle_buffer, seed=args.seed + 1009),
        partial(
            virtual_group_repeated_samples,
            repeats=args.repeated_aug,
            group_size=args.runtime_augmentation_group_size,
        ),
        wds.decode("pil"),
        wds.to_tuple("jpg;jpeg;png", "cls"),
        wds.map_tuple(train_transform, int),
        wds.batched(args.batch_size, partial=False),
    )
    validation = factory(val_shards, shardshuffle=False, seed=args.seed).compose(
        wds.decode("pil"),
        wds.to_tuple("jpg;jpeg;png", "cls"),
        wds.map_tuple(val_transform, int),
        wds.batched(args.eval_batch_size, partial=True),
    )
    official_updates = TRAIN_SAMPLES // args.effective_batch
    steps = args.steps_per_epoch or official_updates * args.grad_accum
    train_loader = wds.WebLoader(
        train,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    ).with_epoch(steps)
    val_loader = wds.WebLoader(
        validation,
        batch_size=None,
        num_workers=args.eval_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    return train_loader, val_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def resize_position_embedding(
    state: dict[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Bicubically resize DeiT-III patch positions across train resolutions."""

    state = dict(state)
    source = state.get("pos_embed")
    target = getattr(model, "pos_embed", None)
    if source is None or target is None or source.shape == target.shape:
        return state
    if source.ndim != 3 or target.ndim != 3 or source.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"cannot resize pos_embed from {tuple(source.shape)} to {tuple(target.shape)}"
        )
    source_tokens, target_tokens = source.shape[1], target.shape[1]
    source_side, target_side = math.isqrt(source_tokens), math.isqrt(target_tokens)
    if source_side**2 != source_tokens or target_side**2 != target_tokens:
        raise ValueError("DeiT-III position embeddings must contain square patch grids")
    positions = source.float().reshape(1, source_side, source_side, -1).permute(0, 3, 1, 2)
    positions = F.interpolate(
        positions,
        size=(target_side, target_side),
        mode="bicubic",
        align_corners=False,
    )
    state["pos_embed"] = positions.permute(0, 2, 3, 1).reshape_as(target).to(source.dtype)
    return state


def create_training_model(args: argparse.Namespace) -> torch.nn.Module:
    kwargs: dict[str, Any] = {
        "pretrained": False,
        "img_size": args.image_size,
        "num_classes": 1000,
        "drop_path_rate": args.drop_path_rate,
    }
    if args.model in DEIT3_RRLSSO_MODELS:
        kwargs.update(
            rank=args.rank,
            gain_init=args.gain_init,
            alpha_init=args.alpha_init,
            alpha_max=args.alpha_max,
            basis_normalization="trace",
            length_normalize=True,
            length_reference=1.0,
        )
    return create_model(args.model, **kwargs)


def atomic_save(state: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_steps: int = 0,
) -> tuple[float, float]:
    model.eval()
    loss_sum = correct = total = 0.0
    for step, (images, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        count = labels.numel()
        loss_sum += loss.item() * count
        correct += (logits.argmax(-1) == labels).sum().item()
        total += count
    model.train()
    if not total:
        raise RuntimeError("ImageNet validation stream produced no batches")
    return loss_sum / total, correct / total


def create_official_optimizer(
    model: torch.nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, str]:
    optimizer_name = args.optimizer
    if optimizer_name == "fusedlamb":
        try:
            optimizer = create_optimizer_v2(
                model,
                opt="fusedlamb",
                lr=args.lr,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )
            return optimizer, "apex_fusedlamb"
        except ImportError as error:
            if not args.allow_unfused_lamb:
                raise RuntimeError(
                    "official DeiT-III pre-training requires NVIDIA Apex FusedLAMB; "
                    "install Apex or use --allow-unfused-lamb only for local smoke tests"
                ) from error
            optimizer_name = "lamb"
    optimizer = create_optimizer_v2(
        model,
        opt=optimizer_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    return optimizer, optimizer_name


def apply_virtual_group_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    mixup: Mixup,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply batch-mode Mixup/CutMix as independent virtual GPU groups."""

    if images.shape[0] != labels.shape[0]:
        raise ValueError("images and labels must have the same batch dimension")
    if images.shape[0] % group_size:
        raise ValueError(
            f"physical batch {images.shape[0]} is not divisible by virtual group {group_size}"
        )
    mixed_labels = []
    for start in range(0, images.shape[0], group_size):
        stop = start + group_size
        # timm Mixup mutates the image slice in place, so no image concatenation is needed.
        _, group_labels = mixup(images[start:stop], labels[start:stop])
        mixed_labels.append(group_labels)
    return images, torch.cat(mixed_labels, dim=0)


def create_model_ema(
    model: torch.nn.Module,
    *,
    enabled: bool,
    decay: float,
    resume_checkpoint: dict[str, Any] | None = None,
) -> ModelEmaV2 | None:
    """Create a stage-local EMA, restoring it only for a true stage resume."""

    if not enabled:
        return None
    model_ema = ModelEmaV2(model, decay=decay)
    if resume_checkpoint is not None and resume_checkpoint.get("model_ema") is not None:
        model_ema.module.load_state_dict(resume_checkpoint["model_ema"])
    return model_ema


def _solve_parameter_groups(
    model: torch.nn.Module,
) -> dict[str, list[torch.nn.Parameter]]:
    groups: dict[str, list[torch.nn.Parameter]] = {"gain": [], "alpha": []}
    for name, parameter in model.named_parameters():
        if name.endswith("theta_gain"):
            groups["gain"].append(parameter)
        elif name.endswith("theta_alpha"):
            groups["alpha"].append(parameter)
    return groups


def regularization_gradient_norms(
    regularization: torch.Tensor,
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """Measure explicit-regularizer gradients without populating ``.grad``."""

    groups = _solve_parameter_groups(model)
    parameters = groups["gain"] + groups["alpha"]
    gradients = torch.autograd.grad(
        regularization,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    split = len(groups["gain"])
    result = {}
    for key, values in (
        ("gain", gradients[:split]),
        ("alpha", gradients[split:]),
    ):
        squares = [value.detach().float().square().sum() for value in values if value is not None]
        result[key] = torch.stack(squares).sum().sqrt() if squares else regularization.new_zeros(())
    return result


def solve_gradient_norms(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    groups = _solve_parameter_groups(model)
    reference = next(model.parameters()).detach().new_zeros((), dtype=torch.float32)
    result = {}
    for key, parameters in groups.items():
        squares = [
            parameter.grad.detach().float().square().sum()
            for parameter in parameters
            if parameter.grad is not None
        ]
        result[key] = torch.stack(squares).sum().sqrt() if squares else reference.clone()
    return result


def solve_parameter_snapshot(model: torch.nn.Module) -> dict[str, list[torch.Tensor]]:
    return {
        key: [parameter.detach().clone() for parameter in parameters]
        for key, parameters in _solve_parameter_groups(model).items()
    }


def solve_update_norms(
    model: torch.nn.Module,
    before: dict[str, list[torch.Tensor]],
) -> dict[str, torch.Tensor]:
    groups = _solve_parameter_groups(model)
    reference = next(model.parameters()).detach().new_zeros((), dtype=torch.float32)
    result = {}
    for key, parameters in groups.items():
        squares = [
            (parameter.detach().float() - old.float()).square().sum()
            for parameter, old in zip(parameters, before[key], strict=True)
        ]
        result[key] = torch.stack(squares).sum().sqrt() if squares else reference.clone()
    return result


def resolve_run_mode(args: argparse.Namespace, last: Path) -> str:
    """Resolve one unambiguous checkpoint state before mutating the output."""

    has_last = last.is_file()
    best = last.with_name("best.pt")
    has_checkpoint = has_last or best.is_file()
    existing_checkpoint = last if has_last else best
    if args.init_checkpoint:
        if args.resume:
            raise ValueError(
                "--init-checkpoint starts a new refinement and requires --no-resume"
            )
        if args.stage == "pretrain":
            raise ValueError("--init-checkpoint is only valid for refinement stages")
        if not Path(args.init_checkpoint).is_file():
            raise FileNotFoundError(f"initialization checkpoint not found: {args.init_checkpoint}")
        if has_checkpoint and not args.overwrite_output:
            raise FileExistsError(
                f"{existing_checkpoint} already exists; resume it, choose another output, or pass "
                "--overwrite-output explicitly"
            )
        return "init_finetune"
    if args.resume and has_last:
        return "resume_pretrain" if args.stage == "pretrain" else "resume_finetune"
    if args.stage != "pretrain":
        raise FileNotFoundError(
            f"no resumable checkpoint at {last}; start refinement with "
            "--no-resume --init-checkpoint PRETRAIN.pt"
        )
    if has_checkpoint and not args.overwrite_output:
        raise FileExistsError(
            f"--no-resume would overwrite {existing_checkpoint}; pass "
            "--overwrite-output explicitly"
        )
    return "new_pretrain"


def checkpoint_metadata(
    args: argparse.Namespace,
    *,
    gain_reference_origin: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "training_stage": args.stage,
        "model_name": args.model,
        "image_size": args.image_size,
        "rank": args.rank,
        "alpha_max": args.alpha_max,
        "init_weights": "raw",
        "gain_reference_origin": gain_reference_origin,
    }


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = checkpoint.get("run_metadata")
    if not isinstance(metadata, dict) or metadata.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError(
            f"resume requires checkpoint schema {CHECKPOINT_SCHEMA}; restart this stage "
            "from its preceding-stage initialization checkpoint"
        )
    expected = {
        "training_stage": args.stage,
        "model_name": args.model,
        "image_size": args.image_size,
        "rank": args.rank,
        "alpha_max": args.alpha_max,
        "init_weights": "raw",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    expected_origin = "pretrain_zero" if args.stage == "pretrain" else "raw_finetune_init"
    if metadata.get("gain_reference_origin") != expected_origin:
        mismatches["gain_reference_origin"] = (
            metadata.get("gain_reference_origin"),
            expected_origin,
        )
    if mismatches:
        raise RuntimeError(f"resume checkpoint metadata mismatch: {mismatches}")
    if checkpoint.get("rrlsso_gain_reference") is None:
        raise RuntimeError("resume checkpoint is missing its persistent gain reference")
    return metadata


def validate_initialization_checkpoint(
    checkpoint: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if "model" not in checkpoint:
        raise RuntimeError("initialization checkpoint has no raw model state")
    metadata = checkpoint.get("run_metadata")
    if metadata is None:
        return
    expected_source_stage = {
        "finetune224": "pretrain",
        "finetune384": "finetune224",
    }[args.stage]
    expected = {
        "training_stage": expected_source_stage,
        "model_name": args.model,
        "rank": args.rank,
        "alpha_max": args.alpha_max,
        "init_weights": "raw",
    }
    mismatches = {
        key: (metadata.get(key), expected_value)
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"initialization checkpoint metadata mismatch: {mismatches}")


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal ImageNet training requires CUDA")
    if not args.local_wds_dir and not os.environ.get("HF_TOKEN"):
        raise RuntimeError("set HF_TOKEN after accepting timm/imagenet-1k-wds access")
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    mathdx_available = is_mathdx_available()
    print(
        f"mathdx_backend={'loaded' if mathdx_available else 'unavailable'}"
        + (f" error={mathdx_load_error()}" if not mathdx_available else ""),
        flush=True,
    )
    if args.require_mathdx and not mathdx_available:
        raise RuntimeError("--require-mathdx was set but its backend is unavailable")
    actual_effective_batch = args.batch_size * args.grad_accum
    if actual_effective_batch != args.effective_batch and not args.allow_batch_mismatch:
        raise ValueError(
            f"official {args.model}/{args.stage} effective batch is {args.effective_batch}, "
            f"got {args.batch_size} x {args.grad_accum} = {actual_effective_batch}"
        )
    amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
    output = Path(args.output)
    last = output / "last.pt"
    run_mode = resolve_run_mode(args, last)
    print(f"checkpoint_mode={run_mode}", flush=True)

    train_loader, val_loader = make_loaders(args)
    model = create_training_model(args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"model={args.model} parameters={parameter_count:,} image_size={args.image_size} "
        f"stage={args.stage}",
        flush=True,
    )
    optimizer, optimizer_impl = create_official_optimizer(model, args)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.epochs,
        lr_min=args.min_lr,
        warmup_t=args.warmup_epochs,
        warmup_lr_init=args.warmup_lr,
        t_in_epochs=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    mixup = Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=args.label_smoothing,
        num_classes=1000,
    )
    criterion: torch.nn.Module = (
        torch.nn.BCEWithLogitsLoss() if args.bce_loss else SoftTargetCrossEntropy()
    )
    official_updates = TRAIN_SAMPLES // args.effective_batch
    steps_per_epoch = args.steps_per_epoch or official_updates * args.grad_accum
    start_epoch = global_update = 0
    best_acc = -1.0
    resume_checkpoint: dict[str, Any] | None = None
    if run_mode in {"resume_pretrain", "resume_finetune"}:
        resume_checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        run_metadata = validate_resume_checkpoint(resume_checkpoint, args)
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
        scaler.load_state_dict(resume_checkpoint.get("scaler", {}))
        start_epoch = int(resume_checkpoint["epoch"])
        global_update = int(resume_checkpoint["global_update"])
        best_acc = float(resume_checkpoint["best_acc"])
        restore_rng_state(resume_checkpoint.get("rng_state"))
        print(f"resumed epoch={start_epoch} update={global_update}", flush=True)
    elif run_mode == "init_finetune":
        initialization = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=False
        )
        validate_initialization_checkpoint(initialization, args)
        model_state = resize_position_embedding(initialization["model"], model)
        model.load_state_dict(model_state)
        print(f"initialized model from {args.init_checkpoint}", flush=True)
        run_metadata = checkpoint_metadata(
            args, gain_reference_origin="raw_finetune_init"
        )
    else:
        run_metadata = checkpoint_metadata(args, gain_reference_origin="pretrain_zero")
    model_ema = create_model_ema(
        model,
        enabled=args.ema,
        decay=args.ema_decay,
        resume_checkpoint=resume_checkpoint,
    )
    saved_gain_reference = (
        resume_checkpoint.get("rrlsso_gain_reference")
        if resume_checkpoint is not None
        else None
    )
    gain_reference = make_rrlsso_gain_reference(
        model,
        anchor_to_current=args.stage != "pretrain",
        state=saved_gain_reference,
    )
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "checkpoint_mode": run_mode,
        "run_metadata": run_metadata,
        "parameter_count": parameter_count,
        "optimizer_impl": optimizer_impl,
        "actual_effective_batch": actual_effective_batch,
        "official_updates_per_epoch": official_updates,
        "virtual_groups_per_update": (
            actual_effective_batch // args.runtime_augmentation_group_size
        ),
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"recipe=official_deit3 optimizer={optimizer_impl} lr={args.lr:g} "
        f"drop_path={args.drop_path_rate:g} repeated_aug={args.repeated_aug} "
        f"amp={args.amp_dtype} effective_batch={actual_effective_batch} "
        f"virtual_group={args.runtime_augmentation_group_size} "
        f"rrlsso_reg=(gain={args.rrlsso_gain_reg:g},alpha={args.rrlsso_alpha_reg:g})",
        flush=True,
    )

    metrics = output / "metrics.csv"
    mode = "a" if start_epoch and metrics.exists() else "w"
    fields = (
        "epoch", "train_loss", "train_data_loss", "train_regularization", "lr",
        "alpha_penalty", "gain_log_mean", "gain_log_std", "gain_anchor_rms",
        "alpha_ratio_mean", "alpha_ratio_std", "alpha_fraction_gt_080",
        "alpha_fraction_gt_095", "global_update", "val_loss", "val_acc",
        "seconds", "peak_gb",
    )
    if args.rrlsso_extended_diagnostics:
        fields = fields + (
            "gain_grad_norm", "alpha_grad_norm", "gain_reg_grad_ratio",
            "alpha_reg_grad_ratio", "gain_update_norm", "alpha_update_norm",
        )
    with metrics.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            loss_sum = data_loss_sum = regularization_sum = 0.0
            alpha_penalty_sum = 0.0
            sampled_regularizer_gradients = None
            sampled_total_gradients = None
            sampled_updates = None
            sampled_parameters_before = None
            observed_steps = 0
            for step, (images, labels) in enumerate(train_loader):
                if step >= steps_per_epoch:
                    break
                observed_steps = step + 1
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                images, labels = apply_virtual_group_mixup(
                    images,
                    labels,
                    mixup,
                    args.runtime_augmentation_group_size,
                )
                if args.bce_loss:
                    labels = labels.gt(0.0).to(labels.dtype)
                with torch.autocast("cuda", dtype=amp_dtype):
                    data_loss = criterion(model(images), labels)
                    regularization, regularization_components = rrlsso_regularization(
                        model,
                        gain_reference=gain_reference,
                        gain_anchor_weight=args.rrlsso_gain_reg,
                        alpha_saturation_weight=args.rrlsso_alpha_reg,
                        alpha_saturation_fraction=args.rrlsso_alpha_saturation,
                    )
                    total_loss = data_loss + regularization
                    loss = total_loss / args.grad_accum
                if args.rrlsso_extended_diagnostics and step + 1 == steps_per_epoch:
                    sampled_regularizer_gradients = regularization_gradient_norms(
                        regularization, model
                    )
                scaler.scale(loss).backward()
                loss_sum += total_loss.detach().item()
                data_loss_sum += data_loss.detach().item()
                regularization_sum += regularization.detach().item()
                alpha_penalty_sum += regularization_components["alpha_saturation"].detach().item()
                if (step + 1) % args.grad_accum == 0:
                    sample_diagnostics = (
                        args.rrlsso_extended_diagnostics
                        and step + 1 == steps_per_epoch
                    )
                    if args.clip_grad or sample_diagnostics:
                        scaler.unscale_(optimizer)
                    if sample_diagnostics:
                        sampled_total_gradients = solve_gradient_norms(model)
                        sampled_parameters_before = solve_parameter_snapshot(model)
                    if args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    if sample_diagnostics:
                        sampled_updates = solve_update_norms(model, sampled_parameters_before)
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                    if model_ema is not None:
                        model_ema.update(model)
                if (step + 1) % args.log_interval == 0:
                    print(
                        f"epoch={epoch + 1} step={step + 1}/{steps_per_epoch} "
                        f"loss={loss_sum / (step + 1):.4f} "
                        f"reg={regularization_sum / (step + 1):.6g}",
                        flush=True,
                    )
            if observed_steps == 0:
                raise RuntimeError("ImageNet training stream produced no batches")
            if observed_steps % args.grad_accum:
                if args.clip_grad or args.rrlsso_extended_diagnostics:
                    scaler.unscale_(optimizer)
                if args.rrlsso_extended_diagnostics:
                    sampled_total_gradients = solve_gradient_norms(model)
                    sampled_parameters_before = solve_parameter_snapshot(model)
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                if args.rrlsso_extended_diagnostics:
                    sampled_updates = solve_update_norms(model, sampled_parameters_before)
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                if model_ema is not None:
                    model_ema.update(model)

            val_loss, val_acc = evaluate(
                model, val_loader, device, amp_dtype, args.max_val_steps
            )
            parameter_diagnostics = rrlsso_parameter_diagnostics(model, gain_reference)
            diagnostic_values = {
                key: float(value.item()) for key, value in parameter_diagnostics.items()
            }
            gradient_values = {}
            if args.rrlsso_extended_diagnostics:
                assert sampled_regularizer_gradients is not None
                assert sampled_total_gradients is not None
                assert sampled_updates is not None
                gradient_values = {
                    "gain_grad_norm": float(sampled_total_gradients["gain"].item()),
                    "alpha_grad_norm": float(sampled_total_gradients["alpha"].item()),
                    "gain_reg_grad_ratio": float(
                        (sampled_regularizer_gradients["gain"] / sampled_total_gradients["gain"].clamp_min(1e-30)).item()
                    ),
                    "alpha_reg_grad_ratio": float(
                        (sampled_regularizer_gradients["alpha"] / sampled_total_gradients["alpha"].clamp_min(1e-30)).item()
                    ),
                    "gain_update_norm": float(sampled_updates["gain"].item()),
                    "alpha_update_norm": float(sampled_updates["alpha"].item()),
                }
            current_lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch + 1,
                "train_loss": loss_sum / observed_steps,
                "train_data_loss": data_loss_sum / observed_steps,
                "train_regularization": regularization_sum / observed_steps,
                "lr": current_lr,
                "alpha_penalty": alpha_penalty_sum / observed_steps,
                **diagnostic_values,
                **gradient_values,
                "global_update": global_update,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seconds": time.time() - started,
                "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            }
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            scheduler.step(epoch + 1)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "model_ema": model_ema.module.state_dict() if model_ema is not None else None,
                "run_metadata": run_metadata,
                "rrlsso_gain_reference": {
                    key: value.detach().cpu() for key, value in gain_reference.items()
                },
                "epoch": epoch + 1,
                "global_update": global_update,
                "best_acc": best_acc,
                "rng_state": rng_state(),
                "args": vars(args),
            }
            atomic_save(state, last)
            if is_best:
                atomic_save(state, output / "best.pt")


def apply_stage_defaults(args: argparse.Namespace) -> argparse.Namespace:
    recipe = official_recipe(args.model, args.stage)
    for name in (
        "image_size", "peak_lr", "min_lr", "warmup_lr", "warmup_epochs",
        "weight_decay", "drop_path_rate", "optimizer", "effective_batch",
        "augmentation_group_size",
        "repeated_aug", "bce_loss", "label_smoothing", "augmentation",
    ):
        value = getattr(recipe, name)
        setattr(args, "lr" if name == "peak_lr" else name, value)
    args.epochs = args.epochs if args.epochs is not None else recipe.epochs
    if args.batch_size is None:
        if args.stage == "pretrain":
            args.batch_size = 128 if recipe.size == "large" else 512
        elif args.stage == "finetune384":
            args.batch_size = {"small": 64, "base": 32, "large": 16}[recipe.size]
        else:
            args.batch_size = 128 if recipe.size == "large" else 512
    if args.grad_accum is None:
        if args.effective_batch % args.batch_size:
            raise ValueError(
                f"physical batch {args.batch_size} does not divide official effective "
                f"batch {args.effective_batch}"
            )
        args.grad_accum = args.effective_batch // args.batch_size
    if args.batch_size % args.augmentation_group_size:
        if not args.allow_batch_mismatch:
            raise ValueError(
                f"physical batch {args.batch_size} must be a multiple of the official "
                f"virtual augmentation group {args.augmentation_group_size}"
            )
        args.runtime_augmentation_group_size = args.batch_size
    else:
        args.runtime_augmentation_group_size = args.augmentation_group_size
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if not args.output:
        suffix = f"pretrain{args.image_size}" if args.stage == "pretrain" else args.stage
        args.output = str(Path("runs/imagenet1k") / args.model / suffix)
    if args.init_checkpoint and args.resume:
        raise ValueError(
            "--init-checkpoint starts a new refinement and requires --no-resume"
        )
    if args.stage == "pretrain" and args.init_checkpoint:
        raise ValueError("--init-checkpoint is only valid for refinement stages")
    if args.stage != "pretrain" and not args.resume and not args.init_checkpoint:
        raise ValueError("the finetune stage requires --init-checkpoint when --no-resume is used")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="deit3_base_patch16_rrlsso", choices=tuple(DEIT3_RRLSSO_MODELS)
    )
    parser.add_argument(
        "--stage", choices=("pretrain", "finetune224", "finetune384"), default="pretrain"
    )
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--gain-init", type=float, default=1.0)
    parser.add_argument("--alpha-init", type=float, default=1.2)
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument("--hf-repo", default="timm/imagenet-1k-wds")
    parser.add_argument("--cache-dir", default="/local_nvme/imagenet-wds")
    parser.add_argument("--local-wds-dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--shard-limit", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.set_defaults(mixup=0.8, cutmix=1.0)
    parser.add_argument("--clip-grad", type=float, default=0.0)
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.99996)
    parser.add_argument("--rrlsso-gain-reg", type=float, default=1e-4)
    parser.add_argument("--rrlsso-alpha-reg", type=float, default=1e-4)
    parser.add_argument("--rrlsso-alpha-saturation", type=float, default=0.8)
    parser.add_argument("--rrlsso-extended-diagnostics", action="store_true")
    parser.add_argument("--allow-unfused-lamb", action="store_true")
    parser.add_argument("--allow-batch-mismatch", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--require-mathdx", action="store_true")
    return apply_stage_defaults(parser.parse_args(argv))


if __name__ == "__main__":
    train(parse_args())
