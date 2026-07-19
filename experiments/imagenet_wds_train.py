"""Single-GPU DeiT-III/RRLSSO ImageNet-1K training over WebDataset shards.

The default protocol is an 800-epoch 192px stage followed by an independent
20-epoch 224px refinement stage.  Both stages save atomic ``last.pt`` and
``best.pt`` checkpoints and can resume at epoch boundaries.
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from timm import create_model
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import examples.models  # noqa: E402,F401  registers project models
from examples.models.deit3_rrlsso import DEIT3_RRLSSO_MODELS  # noqa: E402
from lsso.mathdx_backend import is_mathdx_available, mathdx_load_error  # noqa: E402

TRAIN_SAMPLES = 1_281_167
VAL_SAMPLES = 50_000
TRAIN_SHARDS = 1024
VAL_SHARDS = 64


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
        pipeline.append(wds.shuffle(int(shardshuffle), seed=seed))
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
    train_transform = create_transform(
        input_size=args.image_size,
        is_training=True,
        color_jitter=0.4,
        auto_augment="rand-m9-mstd0.5-inc1",
        interpolation="bicubic",
        re_prob=0.25,
        re_mode="pixel",
        re_count=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    resize = int(round(args.image_size / 0.875))
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
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
        wds.shuffle(args.shuffle_buffer),
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
    steps = args.steps_per_epoch or TRAIN_SAMPLES // args.batch_size
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


def cosine_lr(update: int, total: int, warmup: int, peak: float, floor: float) -> float:
    if update < warmup:
        return peak * (update + 1) / max(1, warmup)
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


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
    max_steps: int = 0,
) -> tuple[float, float]:
    model.eval()
    loss_sum = correct = total = 0.0
    for step, (images, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
    if args.lr <= 0:
        effective_batch = args.batch_size * args.grad_accum
        args.lr = 4e-3 * effective_batch / 4096
        print(
            f"auto_lr={args.lr:.8g} effective_batch={effective_batch} reference=4e-3@4096",
            flush=True,
        )

    train_loader, val_loader = make_loaders(args)
    model = create_training_model(args).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"model={args.model} parameters={parameter_count:,} image_size={args.image_size} "
        f"stage={args.stage}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    mixup = Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=args.label_smoothing,
        num_classes=1000,
    )
    criterion = SoftTargetCrossEntropy()
    steps_per_epoch = args.steps_per_epoch or TRAIN_SAMPLES // args.batch_size
    updates_per_epoch = math.ceil(steps_per_epoch / args.grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = updates_per_epoch * args.warmup_epochs
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {"parameter_count": parameter_count}
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )

    start_epoch = global_update = 0
    best_acc = -1.0
    last = output / "last.pt"
    if args.resume and last.is_file():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        global_update = int(checkpoint["global_update"])
        best_acc = float(checkpoint["best_acc"])
        restore_rng_state(checkpoint.get("rng_state"))
        print(f"resumed epoch={start_epoch} update={global_update}", flush=True)
    elif args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model_state = resize_position_embedding(checkpoint["model"], model)
        model.load_state_dict(model_state)
        print(f"initialized model from {args.init_checkpoint}", flush=True)
    elif args.stage == "finetune":
        raise FileNotFoundError(
            f"no resumable checkpoint at {last}; pass the 192px best.pt with "
            "--init-checkpoint"
        )

    metrics = output / "metrics.csv"
    mode = "a" if start_epoch and metrics.exists() else "w"
    fields = ("epoch", "train_loss", "val_loss", "val_acc", "seconds", "peak_gb")
    with metrics.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            started = time.time()
            torch.cuda.reset_peak_memory_stats()
            loss_sum = 0.0
            observed_steps = 0
            for step, (images, labels) in enumerate(train_loader):
                if step >= steps_per_epoch:
                    break
                observed_steps = step + 1
                images, labels = mixup(images, labels)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = criterion(model(images), labels) / args.grad_accum
                loss.backward()
                loss_sum += loss.item() * args.grad_accum
                if (step + 1) % args.grad_accum == 0:
                    if args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    lr = cosine_lr(
                        global_update, total_updates, warmup_updates, args.lr, args.min_lr
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                if (step + 1) % args.log_interval == 0:
                    print(
                        f"epoch={epoch + 1} step={step + 1}/{steps_per_epoch} "
                        f"loss={loss_sum / (step + 1):.4f}",
                        flush=True,
                    )
            if observed_steps == 0:
                raise RuntimeError("ImageNet training stream produced no batches")
            if observed_steps % args.grad_accum:
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                lr = cosine_lr(
                    global_update, total_updates, warmup_updates, args.lr, args.min_lr
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            val_loss, val_acc = evaluate(model, val_loader, device, args.max_val_steps)
            row = {
                "epoch": epoch + 1,
                "train_loss": loss_sum / observed_steps,
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
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
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
    pretrain = args.stage == "pretrain"
    args.epochs = args.epochs if args.epochs is not None else (800 if pretrain else 20)
    args.image_size = args.image_size if args.image_size is not None else (192 if pretrain else 224)
    args.warmup_epochs = (
        args.warmup_epochs if args.warmup_epochs is not None else (20 if pretrain else 5)
    )
    args.weight_decay = (
        args.weight_decay if args.weight_decay is not None else (0.05 if pretrain else 0.1)
    )
    args.min_lr = args.min_lr if args.min_lr is not None else (1e-5 if pretrain else 1e-6)
    if args.lr is None:
        args.lr = 0.0 if pretrain else 1e-5
    if not args.output:
        suffix = "pretrain192" if pretrain else "finetune224"
        args.output = str(Path("runs/imagenet1k") / args.model / suffix)
    if not pretrain and not args.resume and not args.init_checkpoint:
        raise ValueError("the finetune stage requires --init-checkpoint when --no-resume is used")
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="deit3_base_patch16_192_rrlsso", choices=tuple(DEIT3_RRLSSO_MODELS)
    )
    parser.add_argument("--stage", choices=("pretrain", "finetune"), default="pretrain")
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--eval-workers", type=int, default=2)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--shard-limit", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-mathdx", action="store_true")
    return apply_stage_defaults(parser.parse_args(argv))


if __name__ == "__main__":
    train(parse_args())
