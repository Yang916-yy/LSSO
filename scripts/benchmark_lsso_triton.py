from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from lsso.modules import lsso
from lsso.triton_kernels import (
    correction_apply_triton,
    fused_gram_system_utc_triton,
    fused_gram_utc_triton,
    triton_available,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def bench(name: str, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / iters
    print(f"{name}\t{ms:.3f} ms")
    return ms


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not triton_available():
        raise RuntimeError("Triton is not available")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    B, H, N, r, dh = args.batch, args.heads, args.tokens, args.rank, args.head_dim
    torch.manual_seed(0)
    U = torch.randn(B, H, N, r, device=device, dtype=dtype)
    C = torch.randn(B, H, N, dh, device=device, dtype=dtype)
    mu = F.softplus(torch.zeros(H, device=device)) + 1e-5
    gamma = 0.1 * torch.sigmoid(torch.full((H,), -4.0, device=device))

    U_bh = U.flatten(0, 1)
    C_bh = C.flatten(0, 1)
    eye = torch.eye(r, device=device).view(1, 1, r, r)

    with torch.inference_mode():
        UtU = torch.bmm(U_bh.transpose(1, 2).float(), U_bh.float()).view(B, H, r, r)
        UtC = torch.bmm(U_bh.transpose(1, 2).float(), C_bh.float()).view(B, H, r, dh)
        G = eye + (gamma / mu).view(1, H, 1, 1) * UtU
        L = torch.linalg.cholesky_ex(G.view(B * H, r, r), check_errors=False).L
        K = torch.cholesky_solve(UtC.view(B * H, r, dh), L).to(dtype).view(B, H, r, dh)

        fused = fused_gram_utc_triton(U, C)
        if fused is None:
            raise RuntimeError("fused_gram_utc_triton declined this shape")
        fused_system = fused_gram_system_utc_triton(U, C, (gamma / mu).view(1, H, 1, 1))
        if fused_system is None:
            raise RuntimeError("fused_gram_system_utc_triton declined this shape")
        Y_tri = correction_apply_triton(U, C, K, mu, gamma)
        if Y_tri is None:
            raise RuntimeError("correction_apply_triton declined this shape")

        bench(
            "torch UtU+UtC",
            lambda: (
                torch.bmm(U_bh.transpose(1, 2).float(), U_bh.float()),
                torch.bmm(U_bh.transpose(1, 2).float(), C_bh.float()),
            ),
            args.warmup,
            args.iters,
        )
        bench("triton fused UtU+UtC", lambda: fused_gram_utc_triton(U, C), args.warmup, args.iters)
        bench(
            "torch build G",
            lambda: eye + (gamma / mu).view(1, H, 1, 1) * UtU,
            args.warmup,
            args.iters,
        )
        bench(
            "triton fused G+UtC",
            lambda: fused_gram_system_utc_triton(U, C, (gamma / mu).view(1, H, 1, 1)),
            args.warmup,
            args.iters,
        )
        bench(
            "torch solve_ex",
            lambda: torch.linalg.solve_ex(
                G.view(B * H, r, r),
                UtC.view(B * H, r, dh),
                check_errors=False,
            ).result,
            args.warmup,
            args.iters,
        )
        bench(
            "torch cholesky solve",
            lambda: torch.cholesky_solve(
                UtC.view(B * H, r, dh),
                torch.linalg.cholesky_ex(G.view(B * H, r, r), check_errors=False).L,
            ),
            args.warmup,
            args.iters,
        )
        bench(
            "torch correction apply",
            lambda: C / mu.view(1, H, 1, 1)
            - (gamma / (mu * mu)).view(1, H, 1, 1) * torch.bmm(U_bh, K.flatten(0, 1)).view(B, H, N, dh),
            args.warmup,
            args.iters,
        )
        bench("triton correction apply", lambda: correction_apply_triton(U, C, K, mu, gamma), args.warmup, args.iters)
        bench("torch full lsso core", lambda: lsso(U, C, mu, gamma, use_triton=False), args.warmup, args.iters)
        bench("triton full lsso core", lambda: lsso(U, C, mu, gamma, use_triton=True), args.warmup, args.iters)


if __name__ == "__main__":
    main()
