"""Archived causal-prefix benchmark; not runnable against the supported API."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lsso import LSSO
from lsso.causal_triton import triton_available


class CausalSDPAMixer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, N, D)
        return self.out(y)


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark causal prefix-LSSO.")
    parser.add_argument("--seq-lens", type=parse_ints, default=parse_ints("128,256,512,1024"))
    parser.add_argument("--ranks", type=parse_ints, default=parse_ints("16,32"))
    parser.add_argument("--chunk-sizes", type=parse_ints, default=parse_ints("64,128"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", default="runs/causal_prefix_lsso_benchmark.tsv")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(
    module: nn.Module,
    x: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    amp: bool,
    device: torch.device,
) -> tuple[float, float, float]:
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)

    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        inp = x.detach().clone().requires_grad_(True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            y = module(inp)
            loss = y.square().mean()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        inp = x.detach().clone().requires_grad_(True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            y = module(inp)
            loss = y.square().mean()
        loss.backward()
        optimizer.step()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    params_m = sum(p.numel() for p in module.parameters()) / 1e6
    return elapsed_ms, peak_mb, params_m


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    rows = []
    for seq_len in args.seq_lens:
        x = torch.randn(args.batch_size, seq_len, args.dim, device=device)
        modules: list[tuple[str, nn.Module]] = [
            ("causal_sdpa_mha", CausalSDPAMixer(args.dim, args.heads)),
        ]
        for rank in args.ranks:
            modules.extend(
                [
                    (f"lsso_r{rank}_noncausal", LSSO(args.dim, args.heads, rank=rank)),
                    (f"lsso_r{rank}_causal_prefix_materialized", LSSO(args.dim, args.heads, rank=rank, causal=True)),
                ]
            )
            for chunk_size in args.chunk_sizes:
                modules.append(
                    (
                        f"lsso_r{rank}_causal_prefix_chunk{chunk_size}",
                        LSSO(
                            args.dim,
                            args.heads,
                            rank=rank,
                            causal=True,
                            causal_chunk_size=chunk_size,
                        ),
                    )
                )
                if triton_available() and device.type == "cuda":
                    modules.append(
                        (
                            f"lsso_r{rank}_causal_prefix_triton_chunk{chunk_size}",
                            LSSO(
                                args.dim,
                                args.heads,
                                rank=rank,
                                causal=True,
                                causal_chunk_size=chunk_size,
                                causal_backend="triton",
                            ),
                        )
                    )
        for name, module in modules:
            module = module.to(device)
            elapsed_ms, peak_mb, params_m = measure(
                module,
                x,
                warmup=args.warmup,
                iters=args.iters,
                amp=args.amp,
                device=device,
            )
            row = {
                "seq_len": seq_len,
                "model": name,
                "batch_size": args.batch_size,
                "dim": args.dim,
                "heads": args.heads,
                "ms_fwd_bwd_step": elapsed_ms,
                "peak_alloc_mb": peak_mb,
                "params_m": params_m,
            }
            print(row, flush=True)
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
