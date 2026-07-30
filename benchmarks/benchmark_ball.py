from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn as nn

from lsso import LSSO, LSSOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a complete LSSO or MHA mixer block on one CUDA device."
    )
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--operator", choices=("lsso", "mha"), default="lsso")
    parser.add_argument("--mode", choices=("forward", "train"), default="train")
    parser.add_argument(
        "--implementation",
        choices=("reference", "cuda"),
        default="cuda",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        default="float16",
    )
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
    }[name]


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("the benchmark requires a CUDA device")
    if args.steps <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative; steps and repeats must be positive")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = _dtype(args.dtype)
    torch.backends.cuda.matmul.fp32_precision = "tf32" if args.tf32 else "ieee"
    torch.backends.cudnn.fp32_precision = "tf32" if args.tf32 else "ieee"
    if args.operator == "lsso":
        layer: LSSO | nn.MultiheadAttention = LSSO(
            LSSOConfig(
                dim=args.dim,
                num_heads=args.heads,
                rank=args.rank,
            )
        ).to(device)
    else:
        layer = nn.MultiheadAttention(
            args.dim,
            args.heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        ).to(device)
    if args.mode == "forward":
        layer.eval()
    else:
        layer.train()
    input_dtype = dtype if args.operator == "lsso" else torch.float32
    x = torch.randn(
        args.batch,
        args.length,
        args.dim,
        device=device,
        dtype=input_dtype,
    )
    if args.operator == "lsso" and args.implementation == "cuda":
        from lsso.ball import cuda

        cuda.load(device=device)

    def run_layer() -> torch.Tensor:
        if args.operator == "lsso":
            return layer(x, implementation=args.implementation)
        output, _ = layer(x, x, x, need_weights=False)
        return output

    def step() -> None:
        if args.mode == "forward":
            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not torch.float32):
                    run_layer()
            return
        x.requires_grad_(True)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not torch.float32):
            run_layer().float().square().mean().backward()

    def clear_gradients() -> None:
        layer.zero_grad(set_to_none=True)
        x.grad = None

    for _ in range(args.warmup):
        clear_gradients()
        step()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(args.repeats):
        clear_gradients()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            clear_gradients()
            step()
        torch.cuda.synchronize()
        samples.append(1000.0 * (time.perf_counter() - start) / args.steps)

    properties = torch.cuda.get_device_properties(device)
    implementation = args.implementation if args.operator == "lsso" else "torch_mha"
    rank = str(args.rank) if args.operator == "lsso" else "na"
    position = "rank_rotary" if args.operator == "lsso" else "none"
    print(
        f"device={properties.name} sm={properties.major}.{properties.minor} "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"operator={args.operator} implementation={implementation} mode={args.mode} "
        f"dtype={args.dtype} tf32={args.tf32} batch={args.batch} length={args.length} "
        f"dim={args.dim} heads={args.heads} rank={rank} position={position} "
        f"median_ms={statistics.median(samples):.3f} "
        f"min_ms={min(samples):.3f} max_ms={max(samples):.3f}"
    )


if __name__ == "__main__":
    main()
