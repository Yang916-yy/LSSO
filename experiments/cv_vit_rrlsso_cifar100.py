from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer

from lsso import GroupedRRLSSO, RRLSSO


class RRLSSOSelfAttention(nn.Module):
    """Drop-in replacement for torchvision ViT EncoderBlock.self_attention."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        rank: int,
        dropout: float,
        bias: bool,
        gamma_max: float = 1.2,
        theta_gamma_init: float = 0.5,
        relation_groups: int | None = None,
        length_normalize: bool = True,
        length_reference: float = 1.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if relation_groups is None:
            self.lsso = RRLSSO(
                dim=embed_dim,
                num_heads=num_heads,
                rank=rank,
                dropout=dropout,
                causal=False,
                bias=bias,
                gamma_max=gamma_max,
                theta_gamma_init=theta_gamma_init,
                length_normalize=length_normalize,
                length_reference=length_reference,
            )
        else:
            self.lsso = GroupedRRLSSO(
                dim=embed_dim,
                num_heads=num_heads,
                num_relation_groups=relation_groups,
                rank=rank,
                dropout=dropout,
                bias=bias,
                gamma_max=gamma_max,
                theta_gamma_init=theta_gamma_init,
                length_normalize=length_normalize,
                length_reference=length_reference,
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        del key, value, need_weights, attn_mask, kwargs
        return self.lsso(query), None


def replace_vit_attention_with_rrlsso(
    model: nn.Module,
    *,
    rank: int,
    gamma_max: float = 1.2,
    theta_gamma_init: float = 0.5,
    relation_groups: int | None = None,
    length_normalize: bool = True,
    length_reference: float = 1.0,
) -> int:
    count = 0
    for block in model.encoder.layers:
        old = block.self_attention
        block.self_attention = RRLSSOSelfAttention(
            embed_dim=int(old.embed_dim),
            num_heads=int(old.num_heads),
            rank=rank,
            dropout=float(old.dropout),
            bias=old.in_proj_bias is not None,
            gamma_max=gamma_max,
            theta_gamma_init=theta_gamma_init,
            relation_groups=relation_groups,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        count += 1
    return count


def build_model(
    kind: str,
    *,
    num_classes: int,
    rank: int,
    image_size: int,
    patch_size: int,
    gamma_max: float = 1.2,
    theta_gamma_init: float = 0.5,
    relation_groups: int = 4,
    length_normalize: bool = True,
    length_reference: float = 1.0,
) -> nn.Module:
    model = VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=num_classes,
    )
    if kind == "rrlsso":
        replaced = replace_vit_attention_with_rrlsso(
            model,
            rank=rank,
            gamma_max=gamma_max,
            theta_gamma_init=theta_gamma_init,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        print(f"replaced_attention_layers={replaced}")
    elif kind == "grouped-rrlsso":
        replaced = replace_vit_attention_with_rrlsso(
            model,
            rank=rank,
            gamma_max=gamma_max,
            theta_gamma_init=theta_gamma_init,
            relation_groups=relation_groups,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        print(
            f"replaced_attention_layers={replaced} "
            f"relation_groups={relation_groups}"
        )
    elif kind != "mha":
        raise ValueError(f"unknown model kind {kind!r}")
    return model


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(args.image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    root = Path(args.data_dir)
    train_set = datasets.CIFAR100(root=root, train=True, transform=train_tf, download=True)
    test_set = datasets.CIFAR100(root=root, train=False, transform=test_tf, download=True)
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        generator=train_generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, test_loader


def cosine_lr(step: int, *, total_steps: int, base_lr: float, warmup_steps: int, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    pred = logits.argmax(dim=-1)
    correct = int((pred == targets).sum().detach().cpu())
    return correct, int(targets.numel())


@torch.no_grad()
def global_strength_stats(model: nn.Module) -> tuple[float, float, float]:
    ratios: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, (RRLSSO, GroupedRRLSSO)):
            mu = F.softplus(module.theta_mu.float()) + module.eps
            gamma = module.gamma_max * torch.sigmoid(module.theta_gamma.float())
            ratios.append((gamma / mu).flatten())
    if not ratios:
        return 0.0, 0.0, 0.0
    values = torch.cat(ratios)
    return float(values.mean()), float(values.min()), float(values.max())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_steps: int = 0,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    total = 0
    correct = 0
    for step, (images, labels) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        c, n = accuracy(logits, labels)
        correct += c
        total += n
        loss_sum += float(loss.detach().cpu()) * n
    model.train()
    return loss_sum / max(1, total), correct / max(1, total)


def train_one(args: argparse.Namespace, kind: str) -> None:
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    train_loader, test_loader = make_loaders(args)
    resolved_length_reference = (
        args.length_reference
        if args.length_reference > 0
        else (args.image_size // args.patch_size) ** 2 + 1
    )
    model = build_model(
        kind,
        num_classes=100,
        rank=args.rank,
        image_size=args.image_size,
        patch_size=args.patch_size,
        gamma_max=args.gamma_max,
        theta_gamma_init=args.theta_gamma_init,
        relation_groups=args.relation_groups,
        length_normalize=args.length_normalize,
        length_reference=resolved_length_reference,
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model={kind} params={n_params:,} device={device} dtype={dtype} "
        f"length_normalize={args.length_normalize} "
        f"length_reference={resolved_length_reference}"
    )
    strength_mean, strength_min, strength_max = global_strength_stats(model)
    print(
        f"initial_gamma_over_mu mean={strength_mean:.6f} "
        f"min={strength_min:.6f} max={strength_max:.6f}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_epochs * len(train_loader)))
    out_dir = Path(args.out_dir) / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "kind": kind,
                "params": n_params,
                "resolved_length_reference": resolved_length_reference,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    metrics_path = out_dir / "metrics.csv"
    fields = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "epoch_sec",
        "peak_mem_gb",
        "gamma_over_mu_mean",
        "gamma_over_mu_min",
        "gamma_over_mu_max",
    ]
    if args.save_checkpoints:
        torch.save(
            {
                "kind": kind,
                "epoch": 0,
                "model_state": model.state_dict(),
                "args": vars(args),
                "params": n_params,
            },
            out_dir / "init.pt",
        )

    global_step = 0
    best_val_acc = -1.0
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(args.epochs):
            start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loss_sum = 0.0
            correct = 0
            total = 0
            for step, (images, labels) in enumerate(train_loader):
                if args.max_train_steps_per_epoch and step >= args.max_train_steps_per_epoch:
                    break
                lr = cosine_lr(global_step, total_steps=total_steps, base_lr=args.lr, warmup_steps=warmup_steps, min_lr=args.min_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                c, n = accuracy(logits, labels)
                correct += c
                total += n
                loss_sum += float(loss.detach().cpu()) * n
                global_step += 1
            val_loss, val_acc = evaluate(
                model,
                test_loader,
                device=device,
                dtype=dtype,
                max_steps=args.max_eval_steps_per_epoch,
            )
            strength_mean, strength_min, strength_max = global_strength_stats(model)
            peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
            row = {
                "epoch": epoch + 1,
                "train_loss": loss_sum / max(1, total),
                "train_acc": correct / max(1, total),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_sec": time.time() - start,
                "peak_mem_gb": peak,
                "gamma_over_mu_mean": strength_mean,
                "gamma_over_mu_min": strength_min,
                "gamma_over_mu_max": strength_max,
            }
            writer.writerow(row)
            f.flush()
            print(kind, row, flush=True)
            if args.save_checkpoints:
                ckpt = {
                    "kind": kind,
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "args": vars(args),
                    "params": n_params,
                    "metrics": row,
                }
                torch.save(ckpt, out_dir / "last.pt")
                if row["val_acc"] > best_val_acc:
                    best_val_acc = row["val_acc"]
                    torch.save(ckpt, out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official torchvision ViT-B/4 MHA vs bidirectional RRLSSO on CIFAR-100.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mha", "rrlsso", "grouped-rrlsso"],
        default=["mha", "rrlsso"],
    )
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument("--out-dir", default="runs/cv_vit_b4_cifar100_rrlsso")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--relation-groups", type=int, default=4)
    parser.add_argument("--gamma-max", type=float, default=1.2)
    parser.add_argument("--theta-gamma-init", type=float, default=0.5)
    parser.add_argument(
        "--length-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--length-reference",
        type=float,
        default=1.0,
        help=(
            "Mean-statistics reference (1.0 is sequence-length invariant); "
            "<=0 uses the configured ViT token count for legacy-scale runs."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-steps-per-epoch", type=int, default=0)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for kind in args.models:
        train_one(args, kind)


if __name__ == "__main__":
    main()
