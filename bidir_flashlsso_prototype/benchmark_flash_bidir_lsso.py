from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flash_bidir_lsso import flash_bidir_lsso
from lsso.modules import lsso


def _time_ms(fn, *, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def bench_case(B: int, H: int, N: int, R: int, DH: int, dtype: torch.dtype) -> None:
    torch.manual_seed(23000 + B + H + N + R + DH)
    U = torch.randn(B, H, N, R, device="cuda", dtype=dtype)
    C = torch.randn(B, H, N, DH, device="cuda", dtype=dtype)
    mu = torch.rand(H, device="cuda", dtype=torch.float32) + 0.5
    gamma = torch.rand(H, device="cuda", dtype=torch.float32) * 0.1

    def torch_fwd():
        return lsso(
            U,
            C,
            mu.view(1, H, 1, 1),
            gamma.view(1, H, 1, 1),
            causal=False,
            length_normalize=False,
        )

    def flash_fwd():
        return flash_bidir_lsso(U, C, mu, gamma)

    y_ref = torch_fwd()
    y = flash_fwd()
    max_err = (y.float() - y_ref.float()).abs().max().item()
    torch_ms = _time_ms(torch_fwd)
    flash_ms = _time_ms(flash_fwd)
    speedup = torch_ms / flash_ms if flash_ms > 0 else float("inf")
    print(
        f"B={B} H={H} N={N:5d} R={R:2d} DH={DH:3d} dtype={str(dtype).split('.')[-1]:>8} "
        f"torch={torch_ms:8.3f}ms flash={flash_ms:8.3f}ms speedup={speedup:5.2f}x err={max_err:.3e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    torch.set_float32_matmul_precision("high")
    print(torch.cuda.get_device_name())
    for dtype in (torch.float32, torch.bfloat16):
        for N in (196, 784, 1024, 3136, 4096, 8192):
            bench_case(B=2, H=8, N=N, R=32, DH=64, dtype=dtype)


if __name__ == "__main__":
    main()
