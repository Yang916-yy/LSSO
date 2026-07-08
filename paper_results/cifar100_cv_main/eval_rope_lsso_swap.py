from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from examples.models.common import MLP
from examples.models.vit import VisionEncoder
from lsso import RoPELSSO
from train_cifar import build_loaders, evaluate


class RoPEEncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        rank: int,
        mlp_ratio: float,
        dropout: float,
        gamma_max: float,
        theta_gamma_init: float,
        normalize_u: bool,
        rope_base: float,
        rope_scale: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mixer = RoPELSSO(
            dim=dim,
            num_heads=num_heads,
            rank=rank,
            dropout=dropout,
            gamma_max=gamma_max,
            theta_gamma_init=theta_gamma_init,
            normalize_u=normalize_u,
            rope_base=rope_base,
            rope_scale=rope_scale,
        )
        self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_mask = None if key_padding_mask is None else ~key_padding_mask
        x = x + self.dropout(self.mixer(self.norm1(x), valid_mask=valid_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask[:, :, None], 0.0)
        return x


class RoPEVisionEncoder(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_chans: int = 3,
        num_classes: int,
        dim: int,
        depth: int,
        num_heads: int,
        rank: int,
        mlp_ratio: float,
        dropout: float,
        gamma_max: float,
        theta_gamma_init: float,
        normalize_u: bool,
        rope_base: float,
        rope_scale: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_embed = nn.Conv2d(
            in_chans,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                RoPEEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    rank=rank,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    gamma_max=gamma_max,
                    theta_gamma_init=theta_gamma_init,
                    normalize_u=normalize_u,
                    rope_base=rope_base,
                    rope_scale=rope_scale,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def lsso_layers(self) -> list[RoPELSSO]:
        return [block.mixer for block in self.blocks]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CIFAR checkpoints with native LSSO and RoPE-LSSO swap."
    )
    parser.add_argument("--checkpoint-dir", default="paper_results/cifar100_cv_main/checkpoints")
    parser.add_argument("--output", default="paper_results/cifar100_cv_main/rope_lsso_swap_eval.tsv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--include-baselines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline-models",
        default="mha,nystrom",
        help="Comma-separated baseline checkpoint mixers to evaluate when --include-baselines is enabled.",
    )
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--rope-scale", type=float, default=1.0)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[str, int, str]:
    name = path.name
    seed = 0
    for part in name.split("_"):
        if part.startswith("s") and part[1:].isdigit():
            seed = int(part[1:])
            break
    return (name, seed, str(path))


def model_args_from_checkpoint(
    ckpt_args: dict,
    *,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    max_eval_batches: int,
    device: str,
) -> argparse.Namespace:
    args = argparse.Namespace(**ckpt_args)
    args.data_dir = data_dir
    args.batch_size = batch_size
    args.num_workers = num_workers
    args.max_eval_batches = max_eval_batches
    args.device = device
    args.pin_memory = True
    return args


def build_model(args: argparse.Namespace, num_classes: int, eval_model: str) -> nn.Module:
    common = dict(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_classes=num_classes,
        dim=args.dim,
        depth=args.depth,
        num_heads=args.num_heads,
        rank=args.rank,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        gamma_max=args.gamma_max,
        theta_gamma_init=args.theta_gamma_init,
        normalize_u=not getattr(args, "no_u_rms_norm", False),
    )
    if eval_model == "rope-lsso":
        return RoPEVisionEncoder(
            **common,
            rope_base=getattr(args, "rope_base", 10000.0),
            rope_scale=getattr(args, "rope_scale", 1.0),
        )
    return VisionEncoder(mixer=args.mixer, **common)


def evaluate_checkpoint(
    path: Path,
    *,
    cli_args: argparse.Namespace,
    eval_model: str,
    test_loader,
    num_classes: int,
) -> dict[str, str | int | float]:
    checkpoint = torch.load(path, map_location="cpu")
    ckpt_args = dict(checkpoint["args"])
    model_args = model_args_from_checkpoint(
        ckpt_args,
        data_dir=cli_args.data_dir,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        max_eval_batches=cli_args.max_eval_batches,
        device=cli_args.device,
    )
    model_args.rope_base = cli_args.rope_base
    model_args.rope_scale = cli_args.rope_scale
    model = build_model(model_args, num_classes, eval_model).to(cli_args.device)
    model.load_state_dict(checkpoint["model"], strict=cli_args.strict)
    criterion = nn.CrossEntropyLoss(label_smoothing=getattr(model_args, "label_smoothing", 0.0))
    metrics = evaluate(
        model,
        test_loader,
        criterion,
        torch.device(cli_args.device),
        cli_args.amp and torch.device(cli_args.device).type == "cuda",
        max_batches=cli_args.max_eval_batches,
    )
    return {
        "checkpoint": path.name,
        "source_model": ckpt_args.get("mixer", ""),
        "eval_model": eval_model,
        "rank": int(ckpt_args.get("rank", 0)),
        "seed": int(ckpt_args.get("seed", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "loss": float(metrics["loss"]),
        "acc": float(metrics["acc"]),
        "gamma_over_mu": float(metrics.get("diag_gamma_over_mu", float("nan"))),
        "effective_rank": float(metrics.get("diag_effective_rank", float("nan"))),
        "correction_ratio": float(metrics.get("diag_correction_ratio", float("nan"))),
    }


def main() -> None:
    cli_args = parse_args()
    device = torch.device(cli_args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    checkpoint_dir = Path(cli_args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found in {checkpoint_dir}")

    first = torch.load(checkpoints[0], map_location="cpu")
    first_args = model_args_from_checkpoint(
        dict(first["args"]),
        data_dir=cli_args.data_dir,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        max_eval_batches=cli_args.max_eval_batches,
        device=cli_args.device,
    )
    _, test_loader, num_classes = build_loaders(first_args)

    rows = []
    baseline_models = {item.strip() for item in cli_args.baseline_models.split(",") if item.strip()}
    for path in checkpoints:
        ckpt = torch.load(path, map_location="cpu")
        mixer = ckpt["args"].get("mixer", "")
        eval_models: list[str] = []
        if mixer == "lsso":
            eval_models.extend(["lsso", "rope-lsso"])
        elif cli_args.include_baselines and mixer in baseline_models:
            eval_models.append(mixer)

        for eval_model in eval_models:
            print(f"evaluating {path.name} as {eval_model}", flush=True)
            row = evaluate_checkpoint(
                path,
                cli_args=cli_args,
                eval_model=eval_model,
                test_loader=test_loader,
                num_classes=num_classes,
            )
            rows.append(row)
            print(
                f"  acc={row['acc']:.4f} loss={row['loss']:.4f} "
                f"gamma/mu={row['gamma_over_mu']:.4f}",
                flush=True,
            )

    output = Path(cli_args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "checkpoint",
        "source_model",
        "eval_model",
        "rank",
        "seed",
        "epoch",
        "loss",
        "acc",
        "gamma_over_mu",
        "effective_rank",
        "correction_ratio",
    ]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
