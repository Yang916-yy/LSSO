from __future__ import annotations

import argparse

import torch

from lsso.mathdx_backend import load_mathdx_backend, solve_spd, stats_solve_spd


def _time_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark_solve(
    systems: int,
    rank: int,
    rhs_width: int,
    warmup: int,
    iterations: int,
) -> None:
    a = torch.randn(systems, rank, rank, device="cuda")
    gram = a @ a.transpose(-1, -2) + torch.eye(rank, device="cuda")
    rhs = torch.randn(systems, rank, rhs_width, device="cuda")

    mathdx_ms = _time_ms(lambda: solve_spd(gram, rhs)[0], warmup, iterations)
    torch_ms = _time_ms(
        lambda: torch.linalg.solve_ex(gram, rhs, check_errors=False)[0],
        warmup,
        iterations,
    )
    print(
        f"solve systems={systems} rank={rank} rhs={rhs_width} "
        f"mathdx={mathdx_ms:.4f}ms torch={torch_ms:.4f}ms "
        f"speedup={torch_ms / mathdx_ms:.2f}x"
    )


def benchmark_fused(
    systems: int,
    sequence: int,
    rank: int,
    rhs_width: int,
    warmup: int,
    iterations: int,
) -> None:
    u = 0.2 * torch.randn(systems, sequence, rank, device="cuda")
    c = torch.randn(systems, sequence, rhs_width, device="cuda")
    alpha = torch.full((systems,), 0.02, device="cuda")
    eye = torch.eye(rank, device="cuda")

    def torch_path() -> torch.Tensor:
        gram = u.transpose(1, 2) @ u
        rhs = u.transpose(1, 2) @ c
        return torch.linalg.solve_ex(
            eye + alpha[:, None, None] * gram,
            rhs,
            check_errors=False,
        )[0]

    def hybrid_path() -> torch.Tensor:
        gram = u.transpose(1, 2) @ u
        rhs = u.transpose(1, 2) @ c
        return solve_spd(eye + alpha[:, None, None] * gram, rhs)[0]

    mathdx_ms = _time_ms(
        lambda: stats_solve_spd(u, c, alpha)[0], warmup, iterations
    )
    hybrid_ms = _time_ms(hybrid_path, warmup, iterations)
    torch_ms = _time_ms(torch_path, warmup, iterations)
    print(
        f"stats+solve systems={systems} N={sequence} rank={rank} rhs={rhs_width} "
        f"fused={mathdx_ms:.4f}ms hybrid={hybrid_ms:.4f}ms "
        f"torch={torch_ms:.4f}ms best={min(mathdx_ms, hybrid_ms):.4f}ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not load_mathdx_backend():
        raise SystemExit("Build the backend with scripts/build_mathdx_backend.sh")

    print(torch.cuda.get_device_name())
    for case in ((16, 32, 64), (64, 32, 64), (64, 32, 192), (128, 16, 192)):
        benchmark_solve(
            *case,
            warmup=args.warmup,
            iterations=args.iterations,
        )
    for case in (
        (64, 196, 32, 64),
        (64, 196, 32, 192),
        (64, 3136, 32, 64),
        (64, 3136, 32, 192),
    ):
        benchmark_fused(
            *case,
            warmup=args.warmup,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
