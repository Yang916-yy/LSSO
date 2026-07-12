from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.models import SimpleFeaturePyramid, create_dense_vision_llama


def benchmark_variant(
    *,
    scale: str,
    mixer: str,
    rank: int,
    resolution: int,
    batch_size: int,
    window_size: int | None,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    model = create_dense_vision_llama(
        scale,
        mixer=mixer,
        rank=rank,
        image_size=224,
        window_size=window_size,
    ).cuda().bfloat16().train()
    neck = SimpleFeaturePyramid(model.dim, 256).cuda().bfloat16().train()
    image = torch.randn(
        batch_size,
        3,
        resolution,
        resolution,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def step() -> None:
        outputs = neck(model(image))
        loss = sum(output.float().square().mean() for output in outputs)
        loss.backward()
        model.zero_grad(set_to_none=True)
        neck.zero_grad(set_to_none=True)
        image.grad = None

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    return {
        "scale": scale,
        "mixer": mixer,
        "resolution": resolution,
        "batch_size": batch_size,
        "window_size": "global" if window_size is None else window_size,
        "median_step_ms": statistics.median(timings),
        "images_per_second": 1000.0 * batch_size / statistics.median(timings),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "base"), default="small")
    parser.add_argument("--mixer", choices=("mha", "lsso", "rrlsso"), default="rrlsso")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--windows", nargs="+", default=("14", "16", "global"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = []
    for value in args.windows:
        window_size = None if value == "global" else int(value)
        results.append(
            benchmark_variant(
                scale=args.scale,
                mixer=args.mixer,
                rank=args.rank,
                resolution=args.resolution,
                batch_size=args.batch_size,
                window_size=window_size,
                warmup=args.warmup,
                iterations=args.iterations,
            )
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
