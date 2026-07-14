"""Single-GPU ImageNet-1K training with progressive WebDataset shard caching."""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from timm.data import Mixup, create_transform
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import examples.models  # noqa: F401
from examples.models.vision_llama import VisionLLaMA
from lsso.mathdx_backend import is_mathdx_available, mathdx_load_error
from timm import create_model

TRAIN_SAMPLES = 1_281_167
VAL_SAMPLES = 50_000
TRAIN_SHARDS = 1024
VAL_SHARDS = 64


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def start_prefetcher(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
    if args.prefetch_workers <= 0:
        return None
    helper = ROOT / "tools" / "hf_wds_prefetch.py"
    command = [
        sys.executable,
        str(helper),
        "--repo",
        args.hf_repo,
        "--cache-dir",
        args.cache_dir,
        "--workers",
        str(args.prefetch_workers),
        "--download-attempts",
        str(args.download_attempts),
        "--shard-limit",
        str(args.shard_limit),
        "--seed",
        str(args.seed),
        "--parent-pid",
        str(os.getpid()),
    ]
    process = subprocess.Popen(command, env=os.environ.copy())
    atexit.register(_terminate_process, process)
    print(
        f"Xet prefetcher pid={process.pid} workers={args.prefetch_workers}",
        flush=True,
    )
    return process


def shard_commands(
    split: str,
    cache_dir: Path,
    repo: str,
    max_downloads: int = 8,
    download_attempts: int = 0,
) -> list[str]:
    count = TRAIN_SHARDS if split == "train" else VAL_SHARDS
    prefix = "train" if split == "train" else "validation"
    digits = 4 if split == "train" else 2
    helper = ROOT / "tools" / "hf_wds_stream.py"
    return [
        "pipe:"
        + " ".join(
            shlex.quote(value)
            for value in (
                sys.executable,
                str(helper),
                "--repo",
                repo,
                "--filename",
                f"imagenet1k-{prefix}-{index:0{digits}d}.tar",
                "--cache-dir",
                str(cache_dir),
                "--max-downloads",
                str(max_downloads),
                "--download-attempts",
                str(download_attempts),
            )
        )
        for index in range(count)
    ]


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
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_commands = shard_commands(
        "train", Path(args.cache_dir), args.hf_repo,
        args.max_downloads, args.download_attempts
    )
    val_commands = shard_commands(
        "validation", Path(args.cache_dir), args.hf_repo,
        args.max_downloads, args.download_attempts
    )
    if args.shard_limit:
        train_commands = train_commands[:args.shard_limit]
        val_commands = val_commands[:args.shard_limit]
    train = (
        wds.WebDataset(
            train_commands,
            resampled=False,
            shardshuffle=min(100, len(train_commands)) if len(train_commands) > 1 else False,
            seed=args.seed,
        )
        .shuffle(args.shuffle_buffer)
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(train_transform, int)
        .batched(args.batch_size, partial=False)
    )
    validation = (
        wds.WebDataset(
            val_commands,
            shardshuffle=False,
        )
        .decode("pil")
        .to_tuple("jpg;jpeg;png", "cls")
        .map_tuple(val_transform, int)
        .batched(args.eval_batch_size, partial=True)
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
        # Validation runs once per epoch.  Keeping its worker pool alive in
        # addition to the persistent training pool wastes host RAM for almost
        # the entire epoch, especially with large decoded ImageNet batches.
        persistent_workers=False,
    )
    return train_loader, val_loader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(update: int, total: int, warmup: int, peak: float, floor: float) -> float:
    if update < warmup:
        return peak * (update + 1) / max(1, warmup)
    progress = min(1.0, (update - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device, max_steps: int = 0
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
    return loss_sum / total, correct / total


def train(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("formal ImageNet training requires CUDA")
    if not os.environ.get("HF_TOKEN"):
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
        raise RuntimeError("--require-mathdx was set but the fused CUDA backend could not be loaded")
    if args.lr <= 0:
        effective_batch = args.batch_size * args.grad_accum
        args.lr = 4e-3 * effective_batch / 4096
        print(
            f"auto_lr={args.lr:.8g} effective_batch={effective_batch} "
            "reference=4e-3@4096",
            flush=True,
        )
    prefetcher = start_prefetcher(args)
    train_loader, val_loader = make_loaders(args)
    model = create_model(
        args.model,
        pretrained=False,
        img_size=args.image_size,
        num_classes=1000,
        rank=args.rank,
        length_normalize=True,
        length_reference=1.0,
    ).to(device)
    if not isinstance(model, VisionLLaMA):
        raise TypeError(f"expected VisionLLaMA, got {type(model).__name__}")
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
    (output / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    start_epoch = global_update = 0
    best_acc = -1.0
    last = output / "last.pt"
    if args.resume and last.is_file():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        global_update = checkpoint["global_update"]
        best_acc = checkpoint["best_acc"]
        print(f"resumed epoch={start_epoch} update={global_update}", flush=True)

    metrics = output / "metrics.csv"
    mode = "a" if start_epoch and metrics.exists() else "w"
    with metrics.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "val_loss", "val_acc", "seconds", "peak_gb"))
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
                observed_steps = step + 1
                images, labels = mixup(images, labels)
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = criterion(model(images), labels) / args.grad_accum
                loss.backward()
                loss_sum += loss.item() * args.grad_accum
                if (step + 1) % args.grad_accum == 0 or step + 1 == steps_per_epoch:
                    if args.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    lr = cosine_lr(global_update, total_updates, warmup_updates, args.lr, args.min_lr)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_update += 1
                if (step + 1) % args.log_interval == 0:
                    print(f"epoch={epoch + 1} step={step + 1}/{steps_per_epoch} loss={loss_sum / (step + 1):.4f}", flush=True)
            if observed_steps == 0:
                raise RuntimeError("ImageNet training stream produced no batches")
            if observed_steps % args.grad_accum:
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                lr = cosine_lr(global_update, total_updates, warmup_updates, args.lr, args.min_lr)
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
                "args": vars(args),
            }
            torch.save(state, last)
            if is_best:
                torch.save(state, output / "best.pt")
    if prefetcher is not None:
        _terminate_process(prefetcher)
        atexit.unregister(_terminate_process)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="vision_llama_base_rrlsso_r32")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--hf-repo", default="timm/imagenet-1k-wds")
    parser.add_argument("--cache-dir", default="/local_nvme/imagenet-wds")
    parser.add_argument("--output", default="runs/imagenet1k/vision_llama_base_rrlsso_r32")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=2,
        help="Validation loader workers, independent of --workers; workers exit after each evaluation.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=8,
        help="Maximum concurrent Hugging Face shard downloads across data workers.",
    )
    parser.add_argument(
        "--prefetch-workers",
        type=int,
        default=7,
        help=(
            "Background shard prefetch threads; 0 disables prefetching. The default "
            "reserves one of eight download slots for an on-demand data worker."
        ),
    )
    parser.add_argument(
        "--download-attempts",
        type=int,
        default=0,
        help="Attempts per shard; 0 retries transient network errors indefinitely.",
    )
    parser.add_argument("--shard-limit", type=int, default=0, help="Limit each split for smoke tests")
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)
    parser.add_argument(
        "--lr", type=float, default=0.0,
        help="Peak LR; <=0 linearly scales 4e-3 from effective batch 4096.",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--require-mathdx",
        action="store_true",
        help="Fail instead of silently using the PyTorch fallback when the fused backend is unavailable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
