from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flash_bidir_lsso import _out_kernel, _stats_kernel, flash_bidir_lsso


def _time_ms(fn, *, warmup: int = 10, iters: int = 100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def profile_case(B: int, H: int, N: int, R: int, DH: int, dtype: torch.dtype) -> None:
    torch.manual_seed(25000 + B + H + N + R + DH)
    U = torch.randn(B, H, N, R, device="cuda", dtype=dtype).contiguous()
    C = torch.randn(B, H, N, DH, device="cuda", dtype=dtype).contiguous()
    mu = (torch.rand(H, device="cuda", dtype=torch.float32) + 0.5).contiguous()
    gamma = (torch.rand(H, device="cuda", dtype=torch.float32) * 0.1).contiguous()
    bh = B * H
    U_bh = U.reshape(bh, N, R)
    C_bh = C.reshape(bh, N, DH)
    S = torch.empty(bh, R, R, device="cuda", dtype=torch.float32)
    P = torch.empty(bh, R, DH, device="cuda", dtype=torch.float32)
    mu_bh = mu.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
    gamma_bh = gamma.reshape(1, H).expand(B, H).reshape(bh).float().contiguous()
    alpha = (gamma_bh / mu_bh).view(bh, 1, 1)
    eye = torch.eye(R, device="cuda", dtype=torch.float32).view(1, R, R)
    K = torch.empty(bh, R, DH, device="cuda", dtype=torch.float32)
    Y = torch.empty(bh, N, DH, device="cuda", dtype=dtype)

    def stats():
        grid = lambda meta: (triton.cdiv(DH, meta["BR"]) + triton.cdiv(R, meta["BR"]), triton.cdiv(R, meta["BR"]), bh)
        _stats_kernel[grid](
            U_bh,
            C_bh,
            S,
            P,
            U_bh.stride(0),
            U_bh.stride(1),
            U_bh.stride(2),
            C_bh.stride(0),
            C_bh.stride(1),
            C_bh.stride(2),
            S.stride(0),
            S.stride(1),
            S.stride(2),
            P.stride(0),
            P.stride(1),
            P.stride(2),
            N=N,
            R=R,
            DH=DH,
        )

    def solve():
        G = eye + alpha * S
        K.copy_(torch.linalg.solve(G, P))

    def solve_cholesky():
        G = eye + alpha * S
        chol = torch.linalg.cholesky(G)
        K.copy_(torch.cholesky_solve(P, chol))

    def solve_inverse():
        G = eye + alpha * S
        K.copy_(torch.bmm(torch.linalg.inv(G), P))

    def solve_ex():
        G = eye + alpha * S
        K.copy_(torch.linalg.solve_ex(G, P, check_errors=False)[0])

    def output():
        grid = lambda meta: (triton.cdiv(DH, meta["BD"]), triton.cdiv(N, meta["BL"]), bh)
        _out_kernel[grid](
            U_bh,
            C_bh,
            K,
            Y,
            mu_bh,
            gamma_bh,
            U_bh.stride(0),
            U_bh.stride(1),
            U_bh.stride(2),
            C_bh.stride(0),
            C_bh.stride(1),
            C_bh.stride(2),
            K.stride(0),
            K.stride(1),
            K.stride(2),
            Y.stride(0),
            Y.stride(1),
            Y.stride(2),
            N=N,
            R=R,
            DH=DH,
        )

    stats()
    solve()
    output()
    full_ms = _time_ms(lambda: flash_bidir_lsso(U, C, mu, gamma), iters=50)
    stats_ms = _time_ms(stats)
    solve_ms = _time_ms(solve)
    solve_ex_ms = _time_ms(solve_ex)
    chol_ms = _time_ms(solve_cholesky)
    inv_ms = _time_ms(solve_inverse)
    out_ms = _time_ms(output)
    print(
        f"N={N:5d} dtype={str(dtype).split('.')[-1]:>8} "
        f"full={full_ms:7.3f}ms stats={stats_ms:7.3f}ms solve={solve_ms:7.3f}ms solve_ex={solve_ex_ms:7.3f}ms "
        f"chol={chol_ms:7.3f}ms inv={inv_ms:7.3f}ms output={out_ms:7.3f}ms "
        f"sum={stats_ms + solve_ms + out_ms:7.3f}ms"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    print(torch.cuda.get_device_name())
    for dtype in (torch.bfloat16, torch.float32):
        for N in (196, 784, 3136, 8192):
            profile_case(B=2, H=8, N=N, R=32, DH=64, dtype=dtype)


if __name__ == "__main__":
    main()
