"""Benchmark the dimension-adaptive Trace-normalized solve.

The benchmark reports real end-to-end forward and first-order training
latency.  ``adaptive`` uses the public dispatcher; ``forced_rank`` bypasses
the smaller-side decision and is the old Woodbury baseline.  The latter is
only a performance comparator and is not expected to be robust for
rank-deficient, very-high-alpha systems.
"""

from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass

import torch

from lsso.modules import _TraceNormalizedLSSOAutograd, lsso_gain_alpha


@dataclass(frozen=True)
class Case:
    batch: int
    heads: int
    tokens: int
    rank: int
    width: int


def _measure(function, *, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations, (
        torch.cuda.max_memory_allocated() / 2**20
    )


def benchmark_case(
    case: Case, *, dtype: torch.dtype, warmup: int, iterations: int
) -> list[dict[str, object]]:
    B, H, N, rank, width = (
        case.batch, case.heads, case.tokens, case.rank, case.width
    )
    torch.manual_seed(1701 + N + rank)
    u = torch.randn(B, H, N, rank, device="cuda", dtype=dtype)
    c = torch.randn(B, H, N, width, device="cuda", dtype=dtype)
    gain = torch.ones(1, H, 1, 1, device="cuda")
    theta = torch.full((1, H, 1, 1), 0.18232156, device="cuda")

    def adaptive():
        return lsso_gain_alpha(
            u, c, gain, theta, trace_normalize=True,
            length_normalize=False, _log_alpha=True,
        )

    def forced_rank():
        return _TraceNormalizedLSSOAutograd.apply(
            u, c, gain, theta, None, None, 1e-5, False, 1.0, None
        )

    rows: list[dict[str, object]] = []
    for name, function in (("adaptive", adaptive), ("forced_rank", forced_rank)):
        with torch.no_grad():
            latency, peak = _measure(
                function, warmup=warmup, iterations=iterations
            )
        train_u = u.detach().requires_grad_()
        train_c = c.detach().requires_grad_()
        train_gain = gain.detach().requires_grad_()
        train_theta = theta.detach().requires_grad_()

        def train_step():
            if name == "adaptive":
                output = lsso_gain_alpha(
                    train_u, train_c, train_gain, train_theta,
                    trace_normalize=True, length_normalize=False,
                    _log_alpha=True,
                )
            else:
                output = _TraceNormalizedLSSOAutograd.apply(
                    train_u, train_c, train_gain, train_theta,
                    None, None, 1e-5, False, 1.0, None,
                )
            output.float().square().mean().backward()
            train_u.grad = train_c.grad = None
            train_gain.grad = train_theta.grad = None

        train_latency, train_peak = _measure(
            train_step, warmup=warmup, iterations=iterations
        )
        rows.append({
            "gpu": torch.cuda.get_device_name(),
            "capability": ".".join(
                str(value) for value in torch.cuda.get_device_capability()
            ),
            "dtype": str(dtype).removeprefix("torch."),
            "batch": B,
            "heads": H,
            "tokens": N,
            "rank": rank,
            "width": width,
            "path": name,
            "forward_ms": round(latency, 6),
            "train_ms": round(train_latency, 6),
            "forward_peak_mib": round(peak, 3),
            "train_peak_mib": round(train_peak, 3),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--comprehensive", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.comprehensive:
        cases = [
            Case(B, H, N, rank, width)
            for rank in (16, 32, 48, 64)
            for N in (max(1, rank // 2), rank - 1, rank, rank * 2)
            for B, H, width in ((1, 8, 32), (8, 8, 64))
        ]
    else:
        cases = [
            Case(8, 8, 16, 32, 32),
            Case(8, 8, 31, 32, 32),
            Case(8, 8, 32, 32, 32),
            Case(8, 8, 64, 32, 32),
            Case(16, 12, 145, 32, 64),
            Case(2, 8, 4096, 32, 32),
        ]

    rows: list[dict[str, object]] = []
    for case in cases:
        rows.extend(benchmark_case(
            case, dtype=torch.bfloat16,
            warmup=args.warmup, iterations=args.iterations,
        ))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    print(buffer.getvalue(), end="")


if __name__ == "__main__":
    main()
