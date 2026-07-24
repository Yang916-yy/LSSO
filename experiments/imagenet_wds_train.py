"""Single-GPU DeiT-III/RRLSSO ImageNet-1K training over WebDataset shards.

The stage profiles reproduce Meta's per-size DeiT-III recipes: Small is
pre-trained at 224px, while Base/Large are pre-trained at 192px and optionally
refined at 224px or 384px.  Every stage saves atomic ``last.pt`` and ``best.pt``
checkpoints and can resume at epoch boundaries.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import random
import shlex
import subprocess
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
    official_recipe,
    randaugment_finetune_transform,
    three_augment_transform,
    validation_transform,
    virtual_group_repeated_samples,
)
from experiments.rrlsso_diagnostics import (  # noqa: E402
    rrlsso_parameter_diagnostics,
    scalar_diagnostics_to_floats,
)
from lsso.mathdx_backend import is_mathdx_available, mathdx_load_error  # noqa: E402

TRAIN_SAMPLES = 1_281_167
VAL_SAMPLES = 50_000
TRAIN_SHARDS = 1024
VAL_SHARDS = 64
CHECKPOINT_SCHEMA = 4


def hf_shard_filenames(split: str) -> list[str]:
    count = TRAIN_SHARDS if split == "train" else VAL_SHARDS
    prefix = "train" if split == "train" else "validation"
    digits = 4 if split == "train" else 2
    return [
        f"imagenet1k-{prefix}-{index:0{digits}d}.tar" for index in range(count)
    ]


def hf_split_cache_complete(
    split: str, cache_dir: Path, *, shard_limit: int = 0
) -> bool:
    filenames = hf_shard_filenames(split)
    if shard_limit:
        filenames = filenames[:shard_limit]
    return all(
        (cache_dir / filename).is_file()
        and (cache_dir / filename).stat().st_size > 0
        for filename in filenames
    )


def shard_commands(split: str, cache_dir: Path, repo: str) -> list[str]:
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
                shlex.quote(filename),
                "--cache-dir",
                shlex.quote(str(cache_dir)),
            )
        )
        for filename in hf_shard_filenames(split)
    ]


def complete_hf_split_cache(
    split: str,
    cache_dir: Path,
    repo: str,
    *,
    workers: int,
    shard_limit: int = 0,
    attempts: int = 5,
) -> None:
    """Block until one HF split is completely present in the atomic cache.

    This deliberately invokes the same proven one-shard streaming helper used
    by WebDataset.  Redirecting stdout merely drains its pipe while it fills the
    cache; no second download implementation is introduced.
    """

    filenames = hf_shard_filenames(split)
    if shard_limit:
        filenames = filenames[:shard_limit]
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        name
        for name in filenames
        if not (cache_dir / name).is_file() or (cache_dir / name).stat().st_size == 0
    ]
    if not missing:
        print(
            f"hf_cache_ready split={split} "
            f"complete={len(filenames)}/{len(filenames)}",
            flush=True,
        )
        return

    helper = ROOT / "tools" / "hf_wds_stream.py"
    cache_workers = min(max(1, workers), len(missing))
    barrier_started = time.monotonic()
    print(
        f"hf_cache_fill split={split} complete={len(filenames) - len(missing)}/"
        f"{len(filenames)} missing={len(missing)} workers={cache_workers}",
        flush=True,
    )

    def download(filename: str) -> str:
        command = [
            sys.executable,
            str(helper),
            "--repo",
            repo,
            "--filename",
            filename,
            "--cache-dir",
            str(cache_dir),
        ]
        last_error = ""
        for attempt in range(1, attempts + 1):
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode == 0 and (cache_dir / filename).is_file():
                return filename
            last_error = result.stderr.strip()
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 15))
        raise RuntimeError(
            f"failed to cache {filename} after {attempts} attempts: {last_error}"
        )

    completed = len(filenames) - len(missing)
    milestones = iter(
        threshold
        for threshold in sorted({
            math.ceil(len(filenames) * fraction)
            for fraction in (0.25, 0.5, 0.75)
        })
        if completed < threshold < len(filenames)
    )
    next_milestone = next(milestones, None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=cache_workers) as executor:
        futures = [executor.submit(download, filename) for filename in missing]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if next_milestone is not None and completed >= next_milestone:
                print(
                    f"hf_cache_progress split={split} "
                    f"complete={completed}/{len(filenames)}",
                    flush=True,
                )
                next_milestone = next(milestones, None)

    partials = [
        cache_dir / f"{filename}.partial"
        for filename in filenames
        if (cache_dir / f"{filename}.partial").is_file()
    ]
    if partials:
        raise RuntimeError(
            f"HF {split} cache barrier left {len(partials)} partial shards"
        )
    print(
        f"hf_cache_complete split={split} complete={len(filenames)}/{len(filenames)} "
        f"seconds={time.monotonic() - barrier_started:.1f}",
        flush=True,
    )


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


def make_loaders(
    args: argparse.Namespace, *, train_seed_offset: int = 0
) -> tuple[DataLoader, DataLoader]:
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
        train_cache_hot = hf_split_cache_complete(
            "train", cache_dir, shard_limit=args.shard_limit
        )
        train_shards = (
            [str(cache_dir / filename) for filename in hf_shard_filenames("train")]
            if train_cache_hot
            else shard_commands("train", cache_dir, args.hf_repo)
        )
        val_shards = shard_commands("validation", cache_dir, args.hf_repo)

        def factory(shards, *, shardshuffle, seed):
            return wds.WebDataset(
                shards,
                resampled=False,
                shardshuffle=shardshuffle,
                detshuffle=True,
                seed=seed,
                handler=wds.reraise_exception,
                # A bounded gate may intentionally expose fewer shards than
                # DataLoader workers. Empty workers are valid there; the one
                # owning the shard still supplies the requested epoch length.
                # Formal training keeps WebDataset's strict empty check.
                empty_check=not bool(args.shard_limit),
            )

    if args.shard_limit:
        train_shards = train_shards[: args.shard_limit]
        val_shards = val_shards[: args.shard_limit]
    train = factory(
        train_shards,
        shardshuffle=min(100, len(train_shards)) if len(train_shards) > 1 else False,
        seed=args.seed + train_seed_offset,
    ).compose(
        wds.detshuffle(
            args.shuffle_buffer, seed=args.seed + 1009 + train_seed_offset
        ),
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
        # Cold remote streams must terminate before the cache barrier and
        # validation. Once every train shard is local, use direct tar paths and
        # retain the worker pool across all subsequent epochs.
        persistent_workers=args.workers > 0
        and (bool(args.local_wds_dir) or train_cache_hot),
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
            length_normalize=True,
            length_reference=1.0,
        )
    return create_model(args.model, **kwargs)


def atomic_save(state: dict[str, Any], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def migrate_metrics_without_alpha_std(
    metrics: Path,
    fields: tuple[str, ...],
) -> None:
    """Atomically remove the one retired diagnostic column before appending."""

    if not metrics.is_file():
        return
    with metrics.open(newline="") as handle:
        reader = csv.DictReader(handle)
        existing = tuple(reader.fieldnames or ())
        if existing == fields:
            return
        legacy = list(fields)
        legacy.insert(legacy.index("alpha_observed_min"), "alpha_std")
        if existing != tuple(legacy):
            raise RuntimeError(
                f"unsupported metrics schema in {metrics}: {existing}"
            )
        rows = list(reader)
    temporary = metrics.with_suffix(metrics.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, metrics)
    print(f"metrics_schema_migrated removed=alpha_std path={metrics}", flush=True)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    max_steps: int = 0,
) -> tuple[float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    correct = torch.zeros((), device=device, dtype=torch.int64)
    total = 0
    for step, (images, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        count = labels.numel()
        loss_sum.add_(loss.detach().float(), alpha=count)
        correct.add_((logits.argmax(-1) == labels).sum())
        total += count
    model.train()
    if not total:
        raise RuntimeError("ImageNet validation stream produced no batches")
    loss_value, correct_value = torch.stack(
        (loss_sum, correct.to(torch.float32))
    ).cpu().tolist()
    return loss_value / total, correct_value / total


def create_official_optimizer(
    model: torch.nn.Module, args: argparse.Namespace
) -> tuple[torch.optim.Optimizer, str]:
    optimizer_name = args.optimizer
    if optimizer_name == "fusedlamb":
        apex_available = False
        if torch.cuda.is_available():
            try:
                import apex.optimizers  # noqa: F401
            except ImportError:
                pass
            else:
                apex_available = True
        if apex_available:
            optimizer = create_optimizer_v2(
                model,
                opt="fusedlamb",
                lr=args.lr,
                weight_decay=args.weight_decay,
                eps=1e-8,
            )
            return optimizer, "apex_fusedlamb"
        else:
            if not args.allow_unfused_lamb:
                raise RuntimeError(
                    "official DeiT-III pre-training requires CUDA and NVIDIA Apex "
                    "FusedLAMB; install Apex or use --allow-unfused-lamb only for "
                    "local smoke tests"
                )
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


def checkpoint_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "training_stage": args.stage,
        "model_name": args.model,
        "image_size": args.image_size,
        "rank": args.rank,
        "init_weights": "raw",
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
        "init_weights": "raw",
    }
    mismatches = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"resume checkpoint metadata mismatch: {mismatches}")
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
        run_metadata = checkpoint_metadata(args)
    else:
        run_metadata = checkpoint_metadata(args)
    model_ema = create_model_ema(
        model,
        enabled=args.ema,
        decay=args.ema_decay,
        resume_checkpoint=resume_checkpoint,
    )
    train_loader, val_loader = make_loaders(
        args, train_seed_offset=start_epoch
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
        f"virtual_group={args.runtime_augmentation_group_size}",
        flush=True,
    )

    metrics = output / "metrics.csv"
    mode = "a" if start_epoch and metrics.exists() else "w"
    fields = (
        "epoch", "train_loss", "lr", "gain_log_mean", "gain_log_std",
        "alpha_mean", "alpha_observed_min",
        "alpha_observed_max", "beta_mean",
        "global_update", "val_loss", "val_acc",
        "seconds", "peak_gb",
    )
    if mode == "a":
        migrate_metrics_without_alpha_std(metrics, fields)
    with metrics.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            # Apex's deprecated FusedLAMB does not expose PyTorch's
            # set_to_none keyword. Clearing through the model preserves the
            # desired None-gradient semantics for every optimizer backend.
            model.zero_grad(set_to_none=True)
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            loss_sum = torch.zeros((), device=device, dtype=torch.float32)
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
                    loss = data_loss / args.grad_accum
                scaler.scale(loss).backward()
                loss_sum.add_(data_loss.detach().float())
                if (step + 1) % args.grad_accum == 0:
                    if args.clip_grad:
                        scaler.unscale_(optimizer)
                    if args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad(set_to_none=True)
                    global_update += 1
                    if model_ema is not None:
                        model_ema.update(model)
                if (step + 1) % args.log_interval == 0:
                    logged_loss = float(loss_sum.item()) / (step + 1)
                    print(
                        f"epoch={epoch + 1} step={step + 1}/{steps_per_epoch} "
                        f"loss={logged_loss:.4f}",
                        flush=True,
                    )
            if observed_steps == 0:
                raise RuntimeError("ImageNet training stream produced no batches")
            if observed_steps % args.grad_accum:
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                model.zero_grad(set_to_none=True)
                global_update += 1
                if model_ema is not None:
                    model_ema.update(model)

            if not args.local_wds_dir:
                # The finite RA epoch consumes only part of the shuffled train
                # stream.  Finish the train cache after its non-persistent
                # worker pool has exited, and do not allow validation downloads
                # to overlap any train-shard transfer.
                complete_hf_split_cache(
                    "train",
                    Path(args.cache_dir),
                    args.hf_repo,
                    workers=args.workers,
                    shard_limit=args.shard_limit,
                )
                if not train_loader.pipeline[0].persistent_workers:
                    # The first cold epoch has now filled the atomic cache.
                    # Rebuild once onto direct local tar paths; this new pool
                    # remains alive from the next epoch onward.
                    train_loader, _ = make_loaders(
                        args, train_seed_offset=epoch + 1
                    )
                    if args.workers > 0:
                        assert train_loader.pipeline[0].persistent_workers
                    print(
                        "hf_cache_hot train_loader=direct-local "
                        f"persistent_workers={args.workers > 0}",
                        flush=True,
                    )
            val_loss, val_acc = evaluate(
                model, val_loader, device, amp_dtype, args.max_val_steps
            )
            parameter_diagnostics = rrlsso_parameter_diagnostics(model)
            diagnostic_values = scalar_diagnostics_to_floats(parameter_diagnostics)
            train_loss_value = float(loss_sum.item()) / observed_steps
            current_lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss_value,
                "lr": current_lr,
                **diagnostic_values,
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
