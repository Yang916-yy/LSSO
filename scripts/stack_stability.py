from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models.common import EncoderBlock


@dataclass
class StackResult:
    mixer: str
    depth: int
    rank: int
    batch_size: int
    seq_len: int
    dim: int
    checkpoint: bool
    ok: bool
    elapsed_s: float
    max_memory_mb: float
    loss: float
    input_grad_rms: float
    param_grad_rms: float
    output_rms: float
    output_max_abs: float
    activation_rms_min: float
    activation_rms_max: float
    activation_rms_last: float
    error: str = ""


class DeepEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mixer: str,
        rank: int,
        mlp_ratio: float,
        gamma_max: float,
        theta_gamma_init: float,
        use_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mixer=mixer,
                    rank=rank,
                    mlp_ratio=mlp_ratio,
                    dropout=0.0,
                    gamma_max=gamma_max,
                    theta_gamma_init=theta_gamma_init,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
        activation_rms = []
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            activation_rms.append(x.detach().float().square().mean().sqrt().item())
        return self.norm(x), activation_rms


def rms_of_grads(params) -> float:
    total_sq = 0.0
    total_count = 0
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total_sq += grad.square().sum().item()
        total_count += grad.numel()
    if total_count == 0:
        return float("nan")
    return math.sqrt(total_sq / total_count)


def run_case(args: argparse.Namespace, mixer: str, depth: int) -> StackResult:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(args.device)
    rank = args.rank if mixer != "mha" else 0
    model = DeepEncoder(
        dim=args.dim,
        depth=depth,
        num_heads=args.num_heads,
        mixer=mixer,
        rank=args.rank,
        mlp_ratio=args.mlp_ratio,
        gamma_max=args.gamma_max,
        theta_gamma_init=args.theta_gamma_init,
        use_checkpoint=args.checkpoint,
    ).to(device)
    model.train()

    x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device, requires_grad=True)
    target = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)

    start = time.perf_counter()
    try:
        with torch.amp.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            y, activation_rms = model(x)
            loss = F.mse_loss(y.float(), target.float())
        loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()

        output = y.detach().float()
        ok = (
            torch.isfinite(loss.detach()).item()
            and torch.isfinite(output).all().item()
            and torch.isfinite(x.grad.detach()).all().item()
        )
        error = ""
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.synchronize()
        activation_rms = []
        loss = torch.tensor(float("nan"))
        output = torch.empty(0)
        ok = False
        error = str(exc).splitlines()[0]

    elapsed = time.perf_counter() - start
    max_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if device.type == "cuda"
        else 0.0
    )

    return StackResult(
        mixer=mixer,
        depth=depth,
        rank=rank,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dim=args.dim,
        checkpoint=args.checkpoint,
        ok=ok,
        elapsed_s=elapsed,
        max_memory_mb=max_memory_mb,
        loss=float(loss.detach().cpu()) if torch.isfinite(loss).item() else float("nan"),
        input_grad_rms=x.grad.detach().float().square().mean().sqrt().item()
        if x.grad is not None
        else float("nan"),
        param_grad_rms=rms_of_grads(model.parameters()),
        output_rms=output.square().mean().sqrt().item() if output.numel() else float("nan"),
        output_max_abs=output.abs().max().item() if output.numel() else float("nan"),
        activation_rms_min=min(activation_rms) if activation_rms else float("nan"),
        activation_rms_max=max(activation_rms) if activation_rms else float("nan"),
        activation_rms_last=activation_rms[-1] if activation_rms else float("nan"),
        error=error,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixers", nargs="+", default=["mha", "lsso"])
    parser.add_argument("--depths", nargs="+", type=int, default=[12, 24, 48, 96, 192])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--gamma-max", type=float, default=0.3)
    parser.add_argument("--theta-gamma-init", type=float, default=-4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    results = []
    for mixer in args.mixers:
        for depth in args.depths:
            result = run_case(args, mixer, depth)
            row = asdict(result)
            print(json.dumps(row, sort_keys=True), flush=True)
            results.append(row)
            if not result.ok:
                break

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
