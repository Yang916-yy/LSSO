from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from lsso.modules import lsso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    return parser.parse_args()


def bench(name: str, fn, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / iters
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"{name}\t{ms:.3f} ms\tpeak={peak_mb:.1f} MB")
    return ms, peak_mb


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    B, H, N, r, dh = args.batch, args.heads, args.tokens, args.rank, args.head_dim
    torch.manual_seed(0)
    U0 = torch.randn(B, H, N, r, device=device, dtype=dtype)
    C0 = torch.randn(B, H, N, dh, device=device, dtype=dtype)
    mu0 = F.softplus(torch.zeros(1, H, 1, 1, device=device, dtype=dtype)) + 1e-5
    gamma0 = 0.1 * torch.sigmoid(torch.full((1, H, 1, 1), -4.0, device=device, dtype=dtype))
    probe = torch.randn(B, H, N, dh, device=device, dtype=dtype)

    def step(use_custom_backward: bool, use_triton_backward: bool = False) -> None:
        U = U0.detach().clone().requires_grad_(True)
        C = C0.detach().clone().requires_grad_(True)
        mu = mu0.detach().clone().requires_grad_(True)
        gamma = gamma0.detach().clone().requires_grad_(True)
        Y = lsso(
            U,
            C,
            mu,
            gamma,
            use_custom_backward=use_custom_backward,
            use_triton_backward=use_triton_backward,
        )
        (Y * probe).sum().backward()

    base_ms, _ = bench("autograd fwd+bwd", lambda: step(False), args.warmup, args.iters)
    custom_ms, _ = bench("custom pytorch fwd+bwd", lambda: step(True, False), args.warmup, args.iters)
    triton_ms, _ = bench("custom triton fwd+bwd", lambda: step(True, True), args.warmup, args.iters)
    print(f"custom_pytorch_speedup\t{base_ms / custom_ms:.3f}x")
    print(f"custom_triton_speedup\t{base_ms / triton_ms:.3f}x")


if __name__ == "__main__":
    main()
