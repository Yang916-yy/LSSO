from __future__ import annotations

import argparse
import time

import torch

from lsso import lsso, trace_normalize_basis


def timed(fn, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--sequence", type=int, default=197)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(7)
    shape = (args.batch, args.heads, args.sequence)
    U = torch.randn(*shape, args.rank, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    C = torch.randn(*shape, args.width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    mu = torch.full((1, args.heads, 1, 1), 0.9, device="cuda", requires_grad=True)
    gamma = torch.full((1, args.heads, 1, 1), 0.03, device="cuda", requires_grad=True)
    probe = torch.randn_like(C)

    def explicit() -> None:
        normalized = trace_normalize_basis(U, eps=1e-5, length_reference=1.0)
        output = lsso(normalized, C, mu, gamma, length_normalize=False)
        torch.autograd.grad((output * probe).sum(), (U, C, mu, gamma))

    def absorbed() -> None:
        output = lsso(
            U,
            C,
            mu,
            gamma,
            trace_normalize=True,
            normalization_eps=1e-5,
            length_reference=1.0,
        )
        torch.autograd.grad((output * probe).sum(), (U, C, mu, gamma))

    print(f"device={torch.cuda.get_device_name()} shape={shape} rank={args.rank} width={args.width}")
    for name, fn in (("explicit", explicit), ("absorbed", absorbed)):
        torch.cuda.reset_peak_memory_stats()
        milliseconds = timed(fn, warmup=args.warmup, iterations=args.iterations)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        print(f"{name:>8}: {milliseconds:.4f} ms, peak={peak_gib:.3f} GiB")


if __name__ == "__main__":
    main()
