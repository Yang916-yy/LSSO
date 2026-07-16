from __future__ import annotations

import argparse

import torch

from lsso.mathdx_backend import solve_spd, try_masked_stats_solve_spd


def timed(fn, warmup: int, iterations: int) -> float:
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


def peak_bytes(fn) -> int:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return peak - baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    B, H, N, r, dh = args.batch, args.heads, args.sequence, args.rank, args.width
    dtype = torch.bfloat16
    torch.manual_seed(11)
    u = (0.1 * torch.randn(B, H, N, r, device="cuda")).to(dtype)
    c = torch.randn(B, H, N, dh, device="cuda").to(dtype)
    alpha = (0.05 * torch.rand(B * H, device="cuda")).float()
    eye = torch.eye(r, device="cuda").expand(B * H, r, r)

    print(f"device={torch.cuda.get_device_name()} shape=({B},{H},{N},{r},{dh}) dtype={dtype}")
    print(
        "padding  old_ms  native_ms  hybrid_ms  speedup  "
        "old_peak_MiB  native_peak_MiB  hybrid_peak_MiB"
    )
    for ratio in (0.0, 0.25, 0.5, 0.75, 0.9):
        valid = max(1, round(N * (1.0 - ratio)))
        mask = torch.arange(N, device="cuda")[None] < valid
        mask = mask.expand(B, N).contiguous()
        scale = torch.full((B,), (N / valid) ** 0.5, device="cuda")

        def old_path():
            m = mask[:, None, :, None]
            us = u * m * scale[:, None, None, None].to(dtype)
            cs = c * m
            ubh = us.flatten(0, 1)
            cbh = cs.flatten(0, 1)
            ut = ubh.transpose(1, 2)
            gram = torch.bmm(ut, ubh, out_dtype=torch.float32)
            rhs = torch.bmm(ut, cbh, out_dtype=torch.float32)
            return solve_spd(eye + alpha[:, None, None] * gram, rhs)[0]

        def native_path():
            out = try_masked_stats_solve_spd(u, c, mask, scale, alpha)
            assert out is not None
            return out

        def hybrid_path():
            out = try_masked_stats_solve_spd(
                u, c, mask, scale, alpha, padding_ratio_hint=ratio
            )
            return old_path() if out is None else out

        old_ms = timed(old_path, 20, args.iterations)
        native_ms = timed(native_path, 20, args.iterations)
        hybrid_ms = timed(hybrid_path, 20, args.iterations)
        old_peak = peak_bytes(old_path) / 2**20
        native_peak = peak_bytes(native_path) / 2**20
        hybrid_peak = peak_bytes(hybrid_path) / 2**20
        print(
            f"{ratio:7.0%} {old_ms:7.3f} {native_ms:10.3f} "
            f"{hybrid_ms:10.3f} {old_ms/native_ms:8.2f}x "
            f"{old_peak:13.2f} {native_peak:16.2f} {hybrid_peak:16.2f}"
        )


if __name__ == "__main__":
    main()
