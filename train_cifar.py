from __future__ import annotations

import argparse
import json
import random
import tarfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from examples.models.vit import VisionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer/LSSO vision encoder.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "food101"], default="cifar10")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--mixer", choices=["mha", "lsso", "lsso-no-global"], default="lsso")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma-max", type=float, default=0.1)
    parser.add_argument("--theta-gamma-init", type=float, default=-6.0)
    parser.add_argument("--fixed-mu-gamma", action="store_true")
    parser.add_argument("--no-u-rms-norm", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-custom-backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-triton-backward", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_food101(data_dir: Path) -> Path:
    root = data_dir / "food-101"
    if (root / "meta" / "train.txt").exists() and (root / "images").exists():
        return root

    archive = data_dir / "food-101.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"Food101 archive not found: {archive}")

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive} -> {data_dir}", flush=True)
    with tarfile.open(archive, "r:gz") as tar:
        base = data_dir.resolve()
        for member in tar.getmembers():
            target = (data_dir / member.name).resolve()
            if not str(target).startswith(str(base)):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tar.extractall(data_dir)
    return root


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int]:
    if args.dataset == "food101":
        root = ensure_food101(Path(args.data_dir))
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_tf = transforms.Compose(
            [
                transforms.Resize(args.image_size + 32),
                transforms.RandomCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_tf = transforms.Compose(
            [
                transforms.Resize(args.image_size + 32),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_set = datasets.Food101(root=root.parent, split="train", transform=train_tf, download=False)
        test_set = datasets.Food101(root=root.parent, split="test", transform=test_tf, download=False)
        num_classes = 101
    else:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

        dataset_cls = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        num_classes = 10 if args.dataset == "cifar10" else 100
        train_set = dataset_cls(args.data_dir, train=True, transform=train_tf, download=True)
        test_set = dataset_cls(args.data_dir, train=False, transform=test_tf, download=True)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory and torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        persistent_workers=args.num_workers > 0,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=False,
        **loader_kwargs,
    )
    return train_loader, test_loader, num_classes


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == target).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for step, (images, target) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch = images.shape[0]
        total_loss += loss.item() * batch
        total_acc += accuracy(logits.detach(), target) * batch
        total_count += batch
        if max_batches and step >= max_batches:
            break

    return {"loss": total_loss / total_count, "acc": total_acc / total_count}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    set_lsso_diagnostics_enabled(model, True)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    try:
        for step, (images, target) in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, target)

            batch = images.shape[0]
            total_loss += loss.item() * batch
            total_acc += accuracy(logits, target) * batch
            total_count += batch
            if max_batches and step >= max_batches:
                break
    finally:
        set_lsso_diagnostics_enabled(model, False)

    metrics = {"loss": total_loss / total_count, "acc": total_acc / total_count}
    metrics.update(collect_lsso_diagnostics(model))
    return metrics


def set_lsso_diagnostics_enabled(model: nn.Module, enabled: bool) -> None:
    if not hasattr(model, "lsso_layers"):
        return
    for layer in model.lsso_layers():
        layer.record_diagnostics = enabled


def collect_lsso_diagnostics(model: nn.Module) -> dict[str, float]:
    if not hasattr(model, "lsso_layers"):
        return {}

    gamma_over_mu = []
    effective_rank = []
    correction_ratio = []

    for layer in model.lsso_layers():
        diag = layer.last_diagnostics
        if diag is None:
            continue
        gamma_over_mu.append(diag.gamma_over_mu.flatten())
        effective_rank.append(diag.effective_rank.flatten())
        correction_ratio.append(diag.correction_ratio.flatten())

    if not gamma_over_mu:
        return {}

    return {
        "diag_gamma_over_mu": torch.cat(gamma_over_mu).mean().item(),
        "diag_effective_rank": torch.cat(effective_rank).mean().item(),
        "diag_correction_ratio": torch.cat(correction_ratio).mean().item(),
    }


def freeze_lsso_scales(model: nn.Module) -> None:
    if not hasattr(model, "lsso_layers"):
        return
    for layer in model.lsso_layers():
        layer.theta_mu.requires_grad_(False)
        layer.theta_gamma.requires_grad_(False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"

    train_loader, test_loader, num_classes = build_loaders(args)
    print(
        f"dataset={args.dataset} train={len(train_loader.dataset)} "
        f"eval={len(test_loader.dataset)} classes={num_classes}",
        flush=True,
    )
    model = VisionEncoder(
        image_size=args.image_size,
        num_classes=num_classes,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mixer=args.mixer,
        rank=args.rank,
        patch_size=args.patch_size,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        gamma_max=args.gamma_max,
        theta_gamma_init=args.theta_gamma_init,
        normalize_u=not args.no_u_rms_norm,
        use_custom_backward=args.use_custom_backward,
        use_triton_backward=args.use_triton_backward,
    ).to(device)
    if args.fixed_mu_gamma:
        freeze_lsso_scales(model)
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"model params={params:,} trainable={trainable_params:,} device={device} amp={use_amp} "
        f"use_custom_backward={args.use_custom_backward} "
        f"use_triton_backward={args.use_triton_backward} "
        f"fixed_mu_gamma={args.fixed_mu_gamma} "
        f"u_rms_norm={not args.no_u_rms_norm}",
        flush=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_"
        f"{args.dataset}_{args.mixer}_r{args.rank}_g{args.gamma_max}_tgi{args.theta_gamma_init}_"
        f"d{args.dim}_L{args.depth}_h{args.num_heads}_s{args.seed}"
        f"{'_fixedscale' if args.fixed_mu_gamma else ''}"
        f"{'_nou' if args.no_u_rms_norm else ''}"
    )
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{run_name}.jsonl"
    ckpt_path = run_dir / f"{run_name}.pt"

    best_acc = 0.0
    with log_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"args": vars(args)}, sort_keys=True) + "\n")
        for epoch in range(1, args.epochs + 1):
            print(f"epoch {epoch}/{args.epochs} train", flush=True)
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                criterion,
                device,
                use_amp,
                max_batches=args.max_train_batches,
            )
            print(f"epoch {epoch}/{args.epochs} eval", flush=True)
            eval_metrics = evaluate(
                model,
                test_loader,
                criterion,
                device,
                use_amp,
                max_batches=args.max_eval_batches,
            )
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"eval_{k}": v for k, v in eval_metrics.items()},
            }
            print(json.dumps(row, sort_keys=True))
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()

            if eval_metrics["acc"] > best_acc:
                best_acc = eval_metrics["acc"]
                torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch}, ckpt_path)

    print(f"best_acc={best_acc:.4f}")
    print(f"log={log_path}")
    print(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
