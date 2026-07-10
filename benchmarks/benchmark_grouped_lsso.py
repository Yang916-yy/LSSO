from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch
import torch.nn as nn

from lsso import GroupedLSSO, GroupedRRLSSO, LSSO, RRLSSO


@dataclass(frozen=True)
class Result:
    model: str
    relation_groups: int
    params_m: float
    uc_params_m: float
    forward_ms: float
    forward_backward_ms: float
    peak_memory_mib: float


def _median_cuda_ms(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(start.elapsed_time(end) for start, end in zip(starts, ends))


def _make_model(
    kind: str,
    *,
    dim: int,
    heads: int,
    rank: int,
    groups: int,
) -> nn.Module:
    if kind == "lsso":
        return LSSO(dim=dim, num_heads=heads, rank=rank)
    if kind == "rrlsso":
        return RRLSSO(dim=dim, num_heads=heads, rank=rank)
    if kind == "grouped-lsso":
        return GroupedLSSO(
            dim=dim,
            num_heads=heads,
            num_relation_groups=groups,
            rank=rank,
        )
    if kind == "grouped-rrlsso":
        return GroupedRRLSSO(
            dim=dim,
            num_heads=heads,
            num_relation_groups=groups,
            rank=rank,
        )
    raise ValueError(f"unknown model kind {kind!r}")


def benchmark_one(
    kind: str,
    *,
    batch: int,
    tokens: int,
    dim: int,
    heads: int,
    rank: int,
    groups: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> Result:
    device = torch.device("cuda")
    model = _make_model(kind, dim=dim, heads=heads, rank=rank, groups=groups).to(device)
    x = torch.randn(batch, tokens, dim, device=device)
    amp_enabled = dtype != torch.float32

    def forward() -> None:
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype, enabled=amp_enabled):
            model(x)

    forward_ms = _median_cuda_ms(forward, warmup=warmup, iterations=iterations)

    x_train = x.detach().requires_grad_(True)

    def forward_backward() -> None:
        model.zero_grad(set_to_none=True)
        x_train.grad = None
        with torch.autocast("cuda", dtype=dtype, enabled=amp_enabled):
            y = model(x_train)
            loss = y.float().square().mean()
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    forward_backward_ms = _median_cuda_ms(
        forward_backward,
        warmup=warmup,
        iterations=iterations,
    )
    peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

    relation_groups = groups if kind.startswith("grouped") else heads
    return Result(
        model=kind,
        relation_groups=relation_groups,
        params_m=sum(p.numel() for p in model.parameters()) / 1e6,
        uc_params_m=model.w_uc.weight.numel() / 1e6,
        forward_ms=forward_ms,
        forward_backward_ms=forward_backward_ms,
        peak_memory_mib=peak_memory_mib,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-head and grouped-relation LSSO/RRLSSO PyTorch paths."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    for groups in args.groups:
        if args.heads % groups != 0:
            raise ValueError(f"heads={args.heads} must be divisible by groups={groups}")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    specs = [("lsso", args.heads), ("rrlsso", args.heads)]
    for groups in args.groups:
        specs.extend((("grouped-lsso", groups), ("grouped-rrlsso", groups)))

    results = [
        benchmark_one(
            kind,
            batch=args.batch,
            tokens=args.tokens,
            dim=args.dim,
            heads=args.heads,
            rank=args.rank,
            groups=groups,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for kind, groups in specs
    ]

    print(
        "model\trelation_groups\tparams_M\tuc_params_M\tforward_ms\t"
        "forward_backward_ms\tpeak_memory_MiB"
    )
    for result in results:
        print(
            f"{result.model}\t{result.relation_groups}\t{result.params_m:.4f}\t"
            f"{result.uc_params_m:.4f}\t{result.forward_ms:.4f}\t"
            f"{result.forward_backward_ms:.4f}\t{result.peak_memory_mib:.1f}"
        )


if __name__ == "__main__":
    main()
