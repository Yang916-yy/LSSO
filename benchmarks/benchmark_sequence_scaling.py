from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.models.baselines import OfficialNystromAttention
from lsso import LSSO


class MHA(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, x, x, need_weights=False)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark bidirectional token mixers by sequence length.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_results/sequence_scaling"))
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nystrom-landmarks", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_mixer(name: str, dim: int, heads: int, landmarks: int) -> nn.Module:
    if name == "MHA":
        return MHA(dim, heads)
    if name == "Nystromformer":
        return OfficialNystromAttention(
            dim=dim,
            num_heads=heads,
            num_landmarks=landmarks,
            conv_kernel_size=65,
            dropout=0.0,
        )
    if name == "LSSO-r16":
        return LSSO(dim=dim, num_heads=heads, rank=16, gamma_max=0.3, theta_gamma_init=-4.0)
    if name == "LSSO-r32":
        return LSSO(dim=dim, num_heads=heads, rank=32, gamma_max=0.3, theta_gamma_init=-4.0)
    raise ValueError(name)


def mixer_macs(name: str, n: int, dim: int, heads: int, landmarks: int) -> int:
    if name == "MHA":
        return 4 * n * dim * dim + 2 * n * n * dim
    if name == "Nystromformer":
        m = min(n, landmarks)
        head_dim = dim // heads
        return (
            4 * n * dim * dim
            + 2 * n * m * dim
            + heads * m * m * head_dim
            + heads * m * m * m
            + 65 * n * dim
        )
    rank = 16 if name == "LSSO-r16" else 32
    return (
        n * dim * (heads * rank + dim)
        + n * dim * dim
        + heads * n * rank * rank
        + 2 * n * rank * dim
        + dim * rank * rank
        + heads * rank * rank * rank
    )


def event_time_ms(fn, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return times


def benchmark_one(
    name: str,
    seq_len: int,
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, object]:
    torch.cuda.empty_cache()
    gc.collect()

    device = torch.device(args.device)
    mixer = build_mixer(name, args.dim, args.heads, args.nystrom_landmarks)
    mixer = mixer.to(device=device).eval()
    x = torch.randn(args.batch_size, seq_len, args.dim, device=device)

    def forward() -> torch.Tensor:
        with torch.amp.autocast(
            device_type=device.type,
            enabled=dtype != torch.float32,
            dtype=dtype,
        ):
            return mixer(x)

    with torch.no_grad():
        for _ in range(args.warmup):
            forward()
    torch.cuda.synchronize()

    with torch.no_grad():
        forward_times = event_time_ms(forward, args.repeats)

    def train_step() -> None:
        mixer.zero_grad(set_to_none=True)
        train_x = x.detach().requires_grad_(True)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=dtype != torch.float32,
            dtype=dtype,
        ):
            loss = mixer(train_x).float().square().mean()
        loss.backward()

    for _ in range(args.warmup):
        train_step()
    torch.cuda.synchronize()
    fwd_bwd_times = event_time_ms(train_step, args.repeats)

    mixer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    train_step()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device)

    row = {
        "model": name,
        "seq_len": seq_len,
        "batch_size": args.batch_size,
        "dim": args.dim,
        "heads": args.heads,
        "dtype": args.dtype,
        "forward_ms": statistics.median(forward_times),
        "forward_ms_p25": statistics.quantiles(forward_times, n=4)[0],
        "forward_ms_p75": statistics.quantiles(forward_times, n=4)[2],
        "forward_backward_ms": statistics.median(fwd_bwd_times),
        "forward_backward_ms_p25": statistics.quantiles(fwd_bwd_times, n=4)[0],
        "forward_backward_ms_p75": statistics.quantiles(fwd_bwd_times, n=4)[2],
        "peak_memory_mb": peak_allocated / (1024**2),
        "incremental_peak_memory_mb": (peak_allocated - baseline_allocated) / (1024**2),
        "mixer_macs_g": mixer_macs(
            name, seq_len, args.dim, args.heads, args.nystrom_landmarks
        )
        / 1e9,
    }
    del mixer, x
    torch.cuda.empty_cache()
    gc.collect()
    return row


def save_table(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], path: Path) -> None:
    colors = {
        "MHA": "#1f77b4",
        "Nystromformer": "#d62728",
        "LSSO-r16": "#2ca02c",
        "LSSO-r32": "#9467bd",
    }
    metrics = [
        ("forward_ms", "Forward latency (ms)"),
        ("forward_backward_ms", "Forward + backward (ms)"),
        ("incremental_peak_memory_mb", "Incremental peak memory (MiB)"),
        ("mixer_macs_g", "Mixer MACs (G)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for model, color in colors.items():
            selected = sorted((r for r in rows if r["model"] == model), key=lambda r: r["seq_len"])
            ax.plot(
                [r["seq_len"] for r in selected],
                [r[metric] for r in selected],
                marker="o",
                linewidth=2,
                markersize=5,
                label=model,
                color=color,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024, 2048])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Sequence length")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle("Bidirectional token-mixer sequence scaling", fontsize=14)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for latency and peak-memory benchmarking")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    dtype = dtype_from_name(args.dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    models = ["MHA", "Nystromformer", "LSSO-r16", "LSSO-r32"]
    for seq_len in args.seq_lens:
        for model in models:
            started = time.time()
            try:
                row = benchmark_one(model, seq_len, args, dtype)
                row["status"] = "ok"
                row["error"] = ""
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row = {
                    "model": model,
                    "seq_len": seq_len,
                    "batch_size": args.batch_size,
                    "dim": args.dim,
                    "heads": args.heads,
                    "dtype": args.dtype,
                    "forward_ms": "",
                    "forward_ms_p25": "",
                    "forward_ms_p75": "",
                    "forward_backward_ms": "",
                    "forward_backward_ms_p25": "",
                    "forward_backward_ms_p75": "",
                    "peak_memory_mb": "",
                    "incremental_peak_memory_mb": "",
                    "mixer_macs_g": mixer_macs(
                        model, seq_len, args.dim, args.heads, args.nystrom_landmarks
                    )
                    / 1e9,
                    "status": "oom",
                    "error": str(exc).splitlines()[0],
                }
            row["wall_seconds"] = time.time() - started
            rows.append(row)
            print(json.dumps(row), flush=True)

    successful = [row for row in rows if row["status"] == "ok"]
    save_table(rows, args.output_dir / "sequence_scaling.csv")
    if successful:
        plot(successful, args.output_dir / "sequence_scaling.png")
    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "memory_definition": (
            "peak_memory_mb is torch.cuda.max_memory_allocated during one forward+backward "
            "step after warmup. incremental_peak_memory_mb subtracts the allocated baseline "
            "immediately before the step and is the value plotted. Both exclude unrelated "
            "processes and reserved-but-unused cache"
        ),
        "timing_definition": "median CUDA-event time after warmup",
        "mac_definition": (
            "mixer_macs_g is the theoretical MAC count for one sample and one mixer layer; "
            "timing and memory use the configured batch size"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
