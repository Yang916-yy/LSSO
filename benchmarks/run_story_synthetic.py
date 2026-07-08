from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models.baselines import NystromAttention, PerformerAttention
from examples.models.common import MLP
from lsso import LSSO


@dataclass(frozen=True)
class ModelSpec:
    mixer: str
    rank: int

    @property
    def name(self) -> str:
        if self.mixer == "mha":
            return "mha"
        if self.mixer == "nystrom":
            return f"nystrom-r{self.rank}"
        if self.mixer == "performer":
            return f"performer-r{self.rank}"
        return f"lsso-r{self.rank}"


class StoryBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mixer: str,
        rank: int,
        mlp_ratio: float,
        dropout: float,
        gamma_max: float,
        theta_gamma_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        if mixer == "mha":
            self.mixer = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self._uses_mha = True
        elif mixer == "nystrom":
            self.mixer = NystromAttention(dim, num_heads, num_landmarks=rank)
            self._uses_mha = False
        elif mixer == "performer":
            self.mixer = PerformerAttention(dim, num_heads, nb_features=rank)
            self._uses_mha = False
        elif mixer == "lsso":
            self.mixer = LSSO(
                dim=dim,
                num_heads=num_heads,
                rank=rank,
                dropout=dropout,
                gamma_max=gamma_max,
                theta_gamma_init=theta_gamma_init,
            )
            self._uses_mha = False
        else:
            raise ValueError(f"unknown mixer: {mixer}")
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.norm1(x)
        if self._uses_mha:
            key_padding_mask = None if valid_mask is None else ~valid_mask
            mixed, _ = self.mixer(z, z, z, key_padding_mask=key_padding_mask, need_weights=False)
        else:
            mixed = self.mixer(z, valid_mask=valid_mask)
        x = x + self.dropout(mixed)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        if valid_mask is not None:
            x = x * valid_mask[:, :, None].to(dtype=x.dtype)
        return x


class ContinuousEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        spec: ModelSpec,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        gamma_max: float,
        theta_gamma_init: float,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len + 1, dim))
        self.blocks = nn.ModuleList(
            [
                StoryBlock(dim, num_heads, spec.mixer, spec.rank, mlp_ratio, dropout, gamma_max, theta_gamma_init)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, : x.shape[1] + 1]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])

    def lsso_layers(self) -> list[LSSO]:
        return [block.mixer for block in self.blocks if isinstance(block.mixer, LSSO)]


class TokenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        num_classes: int,
        spec: ModelSpec,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        gamma_max: float,
        theta_gamma_init: float,
        pad_id: int = 0,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.pool = pool
        self.token_embed = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [
                StoryBlock(dim, num_heads, spec.mixer, spec.rank, mlp_ratio, dropout, gamma_max, theta_gamma_init)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).view(1, seq_len).expand(bsz, seq_len)
        valid_mask = input_ids.ne(self.pad_id)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = x * valid_mask[:, :, None].to(dtype=x.dtype)
        for block in self.blocks:
            x = block(x, valid_mask=valid_mask)
        x = self.norm(x)
        if self.pool == "last":
            last = valid_mask.long().sum(dim=1).sub(1).clamp_min(0)
            pooled = x[torch.arange(bsz, device=x.device), last]
        else:
            pooled = x[:, 0]
        return self.head(pooled)

    def lsso_layers(self) -> list[LSSO]:
        return [block.mixer for block in self.blocks if isinstance(block.mixer, LSSO)]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_specs(text: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for raw in text.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item == "mha":
            specs.append(ModelSpec("mha", 0))
            continue
        if ":" in item:
            name, value = item.split(":", 1)
            specs.append(ModelSpec(name, int(value)))
            continue
        if item.startswith("lsso-r"):
            specs.append(ModelSpec("lsso", int(item.split("r", 1)[1])))
            continue
        if item.startswith("nystrom-r"):
            specs.append(ModelSpec("nystrom", int(item.split("r", 1)[1])))
            continue
        if item.startswith("performer-r"):
            specs.append(ModelSpec("performer", int(item.split("r", 1)[1])))
            continue
        raise ValueError(f"cannot parse model spec: {raw}")
    return specs


def make_intrinsic_rank_data(
    train_samples: int,
    val_samples: int,
    seq_len: int,
    input_dim: int,
    intrinsic_rank: int,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    total = train_samples + val_samples
    x = torch.randn(total, seq_len, input_dim, generator=gen)
    w_u = torch.randn(input_dim, intrinsic_rank, generator=gen) / math.sqrt(input_dim)
    w_c = torch.randn(input_dim, num_classes, generator=gen) / math.sqrt(input_dim)
    readout = torch.randn(num_classes, generator=gen)

    with torch.no_grad():
        u = x @ w_u
        u = u * torch.rsqrt((u * u).mean(dim=-1, keepdim=True) + 1e-5)
        c = x @ w_c
        eye = torch.eye(seq_len).expand(total, seq_len, seq_len)
        a = 0.7 * eye + 0.08 * torch.bmm(u, u.transpose(1, 2))
        y = torch.linalg.solve(a.float(), c.float())
        if num_classes == 2:
            score = y.mean(dim=1) @ readout
            threshold = score[:train_samples].median()
            labels = (score > threshold).long()
        else:
            labels = y.mean(dim=1).argmax(dim=-1)

    return (
        x[:train_samples].contiguous(),
        labels[:train_samples].contiguous(),
        x[train_samples:].contiguous(),
        labels[train_samples:].contiguous(),
    )


def make_mqar_data(
    samples: int,
    seq_len: int,
    num_pairs: int,
    num_keys: int,
    num_values: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    min_len = 1 + 2 * num_pairs + 4
    if min_len > seq_len:
        raise ValueError(f"seq_len={seq_len} too short for num_pairs={num_pairs}; need at least {min_len}")
    gen = torch.Generator().manual_seed(seed)
    cls_id, query_id, cand_id = 1, 2, 3
    key_offset = 4
    value_offset = key_offset + num_keys
    vocab_size = value_offset + num_values
    x = torch.zeros(samples, seq_len, dtype=torch.long)
    labels = torch.zeros(samples, dtype=torch.long)
    for i in range(samples):
        keys = torch.randperm(num_keys, generator=gen)[:num_pairs] + key_offset
        values = torch.randint(0, num_values, (num_pairs,), generator=gen) + value_offset
        q_idx = int(torch.randint(0, num_pairs, (1,), generator=gen))
        positive = bool(torch.randint(0, 2, (1,), generator=gen))
        correct = values[q_idx]
        if positive:
            candidate = correct
        else:
            other = int(torch.randint(0, num_pairs - 1, (1,), generator=gen))
            if other >= q_idx:
                other += 1
            candidate = values[other]
        labels[i] = int(positive)
        tokens = [cls_id]
        for key, value in zip(keys.tolist(), values.tolist(), strict=True):
            tokens.extend([key, value])
        tokens.extend([query_id, int(keys[q_idx]), cand_id, int(candidate)])
        x[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return x, labels, vocab_size


def model_mixer_macs(spec: ModelSpec, seq_len: int, dim: int, heads: int, depth: int, mlp_ratio: float) -> int:
    if spec.mixer == "mha":
        per_layer = 4 * seq_len * dim * dim + 2 * seq_len * seq_len * dim
    elif spec.mixer == "performer":
        features = spec.rank
        per_layer = 4 * seq_len * dim * dim + 4 * seq_len * dim * features + seq_len * heads * features
    elif spec.mixer == "nystrom":
        landmarks = min(spec.rank, seq_len)
        conv_kernel = 65
        per_layer = 4 * seq_len * dim * dim
        per_layer += 2 * seq_len * landmarks * dim
        per_layer += heads * landmarks * landmarks * (dim // heads)
        per_layer += heads * landmarks * landmarks * landmarks
        per_layer += conv_kernel * seq_len * dim
    elif spec.mixer == "lsso":
        r = spec.rank
        per_layer = seq_len * dim * (heads * r + dim) + seq_len * dim * dim
        per_layer += heads * seq_len * r * r
        per_layer += 2 * seq_len * r * dim
        per_layer += dim * r * r
        per_layer += heads * r * r * r
    else:
        raise ValueError(spec.mixer)
    return int(depth * per_layer)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].to(device, non_blocking=True)
        yb = y[start : start + batch_size].to(device, non_blocking=True)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += yb.numel()
    return total_loss / total, correct / total


@torch.no_grad()
def collect_lsso_diagnostics(model: nn.Module, sample: torch.Tensor, device: torch.device) -> dict[str, float | None]:
    layers = model.lsso_layers() if hasattr(model, "lsso_layers") else []
    if not layers:
        return {"gamma_over_mu": None, "effective_rank": None, "correction_ratio": None}
    old = [layer.record_diagnostics for layer in layers]
    for layer in layers:
        layer.record_diagnostics = True
    model.eval()
    _ = model(sample.to(device))
    gamma, rank, corr = [], [], []
    for layer in layers:
        diag = layer.last_diagnostics
        if diag is not None:
            gamma.append(diag.gamma_over_mu.mean().item())
            rank.append(diag.effective_rank.mean().item())
            corr.append(diag.correction_ratio.mean().item())
    for layer, value in zip(layers, old, strict=True):
        layer.record_diagnostics = value
    return {
        "gamma_over_mu": sum(gamma) / len(gamma) if gamma else None,
        "effective_rank": sum(rank) / len(rank) if rank else None,
        "correction_ratio": sum(corr) / len(corr) if corr else None,
    }


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    model.train()
    last_loss = 0.0
    last_acc = 0.0
    for step in range(1, steps + 1):
        idx = torch.randint(0, train_x.shape[0], (batch_size,))
        xb = train_x[idx].to(device, non_blocking=True)
        yb = train_y[idx].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        last_loss = loss.item()
        last_acc = (logits.detach().argmax(dim=-1) == yb).float().mean().item()
    val_loss, val_acc = evaluate(model, val_x, val_y, batch_size * 2, device)
    return {"train_loss": last_loss, "train_acc": last_acc, "val_loss": val_loss, "val_acc": val_acc}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_intrinsic(args: argparse.Namespace, specs: list[ModelSpec], device: torch.device) -> list[dict]:
    rows = []
    for intrinsic_rank in args.intrinsic_ranks:
        train_x, train_y, val_x, val_y = make_intrinsic_rank_data(
            args.train_samples,
            args.val_samples,
            args.seq_len,
            args.input_dim,
            intrinsic_rank,
            args.num_classes,
            args.seed + intrinsic_rank * 17,
        )
        for spec in specs:
            set_seed(args.seed + intrinsic_rank * 101 + spec.rank)
            model = ContinuousEncoder(
                input_dim=args.input_dim,
                seq_len=args.seq_len,
                num_classes=args.num_classes,
                spec=spec,
                dim=args.dim,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                dropout=args.dropout,
                gamma_max=args.gamma_max,
                theta_gamma_init=args.theta_gamma_init,
            )
            t0 = time.perf_counter()
            metrics = train_model(model, train_x, train_y, val_x, val_y, args.steps, args.batch_size, args.lr, args.weight_decay, device, args.amp)
            seconds = time.perf_counter() - t0
            diag = collect_lsso_diagnostics(model, val_x[: min(64, len(val_x))], device)
            row = {
                "task": "intrinsic_rank",
                "difficulty": intrinsic_rank,
                "model": spec.name,
                "mixer": spec.mixer,
                "rank": spec.rank,
                "seq_len": args.seq_len + 1,
                "seed": args.seed,
                "steps": args.steps,
                "params_m": count_params(model) / 1e6,
                "mixer_macs_g": model_mixer_macs(spec, args.seq_len + 1, args.dim, args.num_heads, args.depth, args.mlp_ratio) / 1e9,
                "seconds": seconds,
                **metrics,
                **diag,
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
    return rows


def run_mqar(args: argparse.Namespace, specs: list[ModelSpec], device: torch.device) -> list[dict]:
    rows = []
    for num_pairs in args.mqar_pairs:
        train_x, train_y, vocab_size = make_mqar_data(
            args.train_samples,
            args.seq_len,
            num_pairs,
            args.num_keys,
            args.num_values,
            args.seed + num_pairs * 19,
        )
        val_x, val_y, _ = make_mqar_data(
            args.val_samples,
            args.seq_len,
            num_pairs,
            args.num_keys,
            args.num_values,
            args.seed + 10000 + num_pairs * 19,
        )
        for spec in specs:
            set_seed(args.seed + num_pairs * 101 + spec.rank)
            model = TokenEncoder(
                vocab_size=vocab_size,
                seq_len=args.seq_len,
                num_classes=2,
                spec=spec,
                dim=args.dim,
                depth=args.depth,
                num_heads=args.num_heads,
                mlp_ratio=args.mlp_ratio,
                dropout=args.dropout,
                gamma_max=args.gamma_max,
                theta_gamma_init=args.theta_gamma_init,
                pool=args.mqar_pool,
            )
            t0 = time.perf_counter()
            metrics = train_model(model, train_x, train_y, val_x, val_y, args.steps, args.batch_size, args.lr, args.weight_decay, device, args.amp)
            seconds = time.perf_counter() - t0
            diag = collect_lsso_diagnostics(model, val_x[: min(64, len(val_x))], device)
            row = {
                "task": "mqar",
                "difficulty": num_pairs,
                "model": spec.name,
                "mixer": spec.mixer,
                "rank": spec.rank,
                "seq_len": args.seq_len,
                "seed": args.seed,
                "steps": args.steps,
                "params_m": count_params(model) / 1e6,
                "mixer_macs_g": model_mixer_macs(spec, args.seq_len, args.dim, args.num_heads, args.depth, args.mlp_ratio) / 1e9,
                "seconds": seconds,
                **metrics,
                **diag,
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            rows.append(row)
    return rows


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Story-building synthetic experiments for LSSO.")
    parser.add_argument("--task", choices=["intrinsic_rank", "mqar", "both"], default="both")
    parser.add_argument("--models", default="mha,nystrom:32,lsso:16,lsso:32")
    parser.add_argument("--out-dir", default="runs/story_synthetic_smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--intrinsic-ranks", type=parse_ints, default=parse_ints("8,32"))
    parser.add_argument("--mqar-pairs", type=parse_ints, default=parse_ints("24,56"))
    parser.add_argument("--mqar-pool", choices=["cls", "last"], default="last")
    parser.add_argument("--num-keys", type=int, default=512)
    parser.add_argument("--num-values", type=int, default=512)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_specs(args.models)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rows: list[dict] = []
    config_path = out_dir / "config.json"
    config = vars(args).copy()
    config["models"] = [spec.name for spec in specs]
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    if args.task in {"intrinsic_rank", "both"}:
        intrinsic_rows = run_intrinsic(args, specs, device)
        write_rows(out_dir / "intrinsic_rank_summary.tsv", intrinsic_rows)
        rows.extend(intrinsic_rows)
    if args.task in {"mqar", "both"}:
        mqar_rows = run_mqar(args, specs, device)
        write_rows(out_dir / "mqar_summary.tsv", mqar_rows)
        rows.extend(mqar_rows)
    write_rows(out_dir / "summary.tsv", rows)


if __name__ == "__main__":
    main()
