from __future__ import annotations

import argparse
import csv
import json
import tarfile
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lsso import GroupedRRLSSO, LSSO, RRLSSO


class TimmRRLSSOAttention(nn.Module):
    """RRLSSO replacement for a timm ViT attention module."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        rank: int,
        dropout: float,
        bias: bool,
        gain_init: float,
        alpha_init: float,
        solve_parameterization: str = "gain_alpha",
        alpha_max: float = 3.0,
        basis_normalization: str = "trace",
        length_normalize: bool,
        length_reference: float,
    ) -> None:
        super().__init__()
        self.rrlsso = RRLSSO(
            dim=embed_dim,
            num_heads=num_heads,
            rank=rank,
            dropout=dropout,
            bias=bias,
            gain_init=gain_init,
            alpha_init=alpha_init,
            solve_parameterization=solve_parameterization,
            alpha_max=alpha_max,
            basis_normalization=basis_normalization,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if attn_mask is not None:
            raise ValueError("the DeiT-III Food101 path does not use an attention mask")
        if is_causal:
            raise ValueError("the bidirectional RRLSSO replacement is not causal")
        return self.rrlsso(x)


def replace_timm_attention_with_rrlsso(
    model: nn.Module,
    *,
    rank: int,
    gain_init: float,
    alpha_init: float,
    solve_parameterization: str = "gain_alpha",
    alpha_max: float = 3.0,
    basis_normalization: str = "trace",
    length_normalize: bool,
    length_reference: float,
) -> int:
    count = 0
    for block in model.blocks:
        old = block.attn
        block.attn = TimmRRLSSOAttention(
            embed_dim=int(old.qkv.in_features),
            num_heads=int(old.num_heads),
            rank=rank,
            dropout=float(old.attn_drop.p),
            bias=old.qkv.bias is not None,
            gain_init=gain_init,
            alpha_init=alpha_init,
            solve_parameterization=solve_parameterization,
            alpha_max=alpha_max,
            basis_normalization=basis_normalization,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        count += 1
    return count


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
        gain_init: float = 1.0,
        alpha_init: float = 1.2,
        solve_parameterization: str = "gain_alpha",
        alpha_max: float = 3.0,
        basis_normalization: str = "trace",
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
                bias=bias,
                gain_init=gain_init,
                alpha_init=alpha_init,
                solve_parameterization=solve_parameterization,
                alpha_max=alpha_max,
                basis_normalization=basis_normalization,
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
                gain_init=gain_init,
                alpha_init=alpha_init,
                solve_parameterization=solve_parameterization,
                alpha_max=alpha_max,
                basis_normalization=basis_normalization,
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
    gain_init: float = 1.0,
    alpha_init: float = 1.2,
    solve_parameterization: str = "gain_alpha",
    alpha_max: float = 3.0,
    basis_normalization: str = "trace",
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
            gain_init=gain_init,
            alpha_init=alpha_init,
            solve_parameterization=solve_parameterization,
            alpha_max=alpha_max,
            basis_normalization=basis_normalization,
            relation_groups=relation_groups,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        count += 1
    return count


def build_model(
    kind: str,
    *,
    backbone: str = "torchvision-vit-b",
    num_classes: int,
    rank: int,
    image_size: int,
    patch_size: int,
    gain_init: float = 1.0,
    alpha_init: float = 1.2,
    solve_parameterization: str = "gain_alpha",
    alpha_max: float = 3.0,
    basis_normalization: str = "trace",
    relation_groups: int = 4,
    length_normalize: bool = True,
    length_reference: float = 1.0,
) -> nn.Module:
    if backbone == "deit3-base":
        if image_size != 224 or patch_size != 16:
            raise ValueError("deit3-base currently requires image_size=224 and patch_size=16")
        if kind not in {"mha", "rrlsso"}:
            raise ValueError(f"DeiT-III does not support model kind {kind!r}")
        model = timm.create_model(
            "deit3_base_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )
        if kind == "rrlsso":
            replaced = replace_timm_attention_with_rrlsso(
                model,
                rank=rank,
                gain_init=gain_init,
                alpha_init=alpha_init,
                solve_parameterization=solve_parameterization,
                alpha_max=alpha_max,
                basis_normalization=basis_normalization,
                length_normalize=length_normalize,
                length_reference=length_reference,
            )
            print(
                f"replaced_attention_layers={replaced} "
                "rank_rotary=ordinary",
                flush=True,
            )
        return model
    if backbone != "torchvision-vit-b":
        raise ValueError(f"unknown backbone {backbone!r}")
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
            gain_init=gain_init,
            alpha_init=alpha_init,
            solve_parameterization=solve_parameterization,
            alpha_max=alpha_max,
            basis_normalization=basis_normalization,
            length_normalize=length_normalize,
            length_reference=length_reference,
        )
        print(f"replaced_attention_layers={replaced}")
    elif kind == "grouped-rrlsso":
        replaced = replace_vit_attention_with_rrlsso(
            model,
            rank=rank,
            gain_init=gain_init,
            alpha_init=alpha_init,
            solve_parameterization=solve_parameterization,
            alpha_max=alpha_max,
            basis_normalization=basis_normalization,
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
    root = Path(args.data_dir)
    if args.dataset == "cifar100":
        dataset_dir = root / "cifar-100-python"
        if not dataset_dir.is_dir() and args.data_archive:
            archive = Path(args.data_archive)
            if not archive.is_file():
                raise FileNotFoundError(f"CIFAR-100 archive not found: {archive}")
            root.mkdir(parents=True, exist_ok=True)
            print(f"extracting_cifar100 archive={archive} destination={root}", flush=True)
            with tarfile.open(archive, mode="r:gz") as handle:
                handle.extractall(root, filter="data")
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
        train_set = datasets.CIFAR100(root=root, train=True, transform=train_tf, download=False)
        test_set = datasets.CIFAR100(
            root=root, train=False, transform=test_tf, download=False
        )
    elif args.dataset == "food101":
        dataset_dir = root / "food-101"
        if not dataset_dir.is_dir():
            archive = Path(args.data_archive) if args.data_archive else None
            if archive is None or not archive.is_file():
                raise FileNotFoundError(
                    "Food-101 is missing. Pass --data-archive /path/to/food-101.tar.gz "
                    "or extract the archive under --data-dir."
                )
            root.mkdir(parents=True, exist_ok=True)
            print(f"extracting_food101 archive={archive} destination={root}", flush=True)
            with tarfile.open(archive, mode="r:gz") as handle:
                handle.extractall(root, filter="data")
        # ImageNet normalization and one identical strong, image-scale-aware
        # recipe for both MHA and RRLSSO.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    args.image_size, scale=(0.6, 1.0), ratio=(0.75, 4.0 / 3.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(num_ops=2, magnitude=9),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing(
                    p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0.0
                ),
            ]
        )
        resize_size = int(round(args.image_size / 0.875))
        test_tf = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        train_set = datasets.Food101(root=root, split="train", transform=train_tf)
        test_set = datasets.Food101(root=root, split="test", transform=test_tf)
    else:
        raise ValueError(f"unknown dataset {args.dataset!r}")
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
        if isinstance(module, (LSSO, RRLSSO, GroupedRRLSSO)):
            _gain, alpha = module.effective_gain_alpha()
            ratios.append(alpha.float().flatten())
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
    num_classes = 101 if args.dataset == "food101" else 100
    model = build_model(
        kind,
        backbone=args.backbone,
        num_classes=num_classes,
        rank=args.rank,
        image_size=args.image_size,
        patch_size=args.patch_size,
        gain_init=args.gain_init,
        alpha_init=args.alpha_init,
        solve_parameterization=args.solve_parameterization,
        alpha_max=args.alpha_max,
        basis_normalization=args.basis_normalization,
        relation_groups=args.relation_groups,
        length_normalize=args.length_normalize,
        length_reference=resolved_length_reference,
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"dataset={args.dataset} backbone={args.backbone} model={kind} "
        f"params={n_params:,} device={device} dtype={dtype} "
        f"length_normalize={args.length_normalize} "
        f"length_reference={resolved_length_reference} "
        f"solve_parameterization={args.solve_parameterization} "
        f"basis_normalization={args.basis_normalization} "
        "rank_rotary=ordinary"
    )
    strength_mean, strength_min, strength_max = global_strength_stats(model)
    print(
        f"initial_gamma_over_mu mean={strength_mean:.6f} "
        f"min={strength_min:.6f} max={strength_max:.6f}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    updates_per_epoch = (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps
    total_steps = args.epochs * updates_per_epoch
    warmup_steps = max(1, int(args.warmup_epochs * updates_per_epoch))
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
    resume_path = out_dir / "last.pt"
    start_epoch = 0
    global_step = 0
    best_val_acc = -1.0
    if args.auto_resume and resume_path.is_file():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step", start_epoch * updates_per_epoch))
        best_val_acc = float(checkpoint.get("metrics", {}).get("val_acc", -1.0))
        print(f"resumed model={kind} checkpoint={resume_path} epoch={start_epoch}", flush=True)
    elif args.save_checkpoints:
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

    metrics_mode = "a" if start_epoch > 0 and metrics_path.is_file() else "w"
    with metrics_path.open(metrics_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if metrics_mode == "w":
            writer.writeheader()
        for epoch in range(start_epoch, args.epochs):
            start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            loss_sum = 0.0
            correct = 0
            total = 0
            optimizer.zero_grad(set_to_none=True)
            micro_steps = min(
                len(train_loader),
                args.max_train_steps_per_epoch or len(train_loader),
            )
            for step, (images, labels) in enumerate(train_loader):
                if args.max_train_steps_per_epoch and step >= args.max_train_steps_per_epoch:
                    break
                if step % args.grad_accum_steps == 0:
                    lr = cosine_lr(global_step, total_steps=total_steps, base_lr=args.lr, warmup_steps=warmup_steps, min_lr=args.min_lr)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=(device.type == "cuda" and dtype != torch.float32)):
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)
                (loss / args.grad_accum_steps).backward()
                do_update = (
                    (step + 1) % args.grad_accum_steps == 0
                    or step + 1 == micro_steps
                )
                if do_update:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                c, n = accuracy(logits, labels)
                correct += c
                total += n
                loss_sum += float(loss.detach().cpu()) * n
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
                    "global_step": global_step,
                    "args": vars(args),
                    "params": n_params,
                    "metrics": row,
                }
                torch.save(ckpt, out_dir / "last.pt")
                if row["val_acc"] > best_val_acc:
                    best_val_acc = row["val_acc"]
                    torch.save(ckpt, out_dir / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled ViT mixer comparison.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["mha", "lsso", "rrlsso", "grouped-rrlsso"],
        default=["mha", "rrlsso"],
    )
    parser.add_argument("--dataset", choices=["cifar100", "food101"], default="cifar100")
    parser.add_argument(
        "--backbone",
        choices=[
            "torchvision-vit-b", "deit3-base",
        ],
        default="torchvision-vit-b",
    )
    parser.add_argument("--data-dir", default="data/torchvision")
    parser.add_argument(
        "--data-archive",
        default="",
        help="Optional CIFAR-100 or Food-101 tar.gz archive extracted under --data-dir.",
    )
    parser.add_argument("--out-dir", default="runs/cv_vit_b4_cifar100_rrlsso")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--relation-groups", type=int, default=4)
    parser.add_argument("--gain-init", type=float, default=1.0)
    parser.add_argument("--alpha-init", type=float, default=1.2)
    parser.add_argument(
        "--solve-parameterization",
        choices=["gain_alpha", "fixed_gain_alpha"],
        default="gain_alpha",
    )
    parser.add_argument("--alpha-max", type=float, default=3.0)
    parser.add_argument(
        "--basis-normalization",
        choices=["trace", "token_rms"],
        default="trace",
        help="Relation-basis normalization used by LSSO/RRLSSO mixers.",
    )
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
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1")
    for kind in args.models:
        train_one(args, kind)


if __name__ == "__main__":
    main()
