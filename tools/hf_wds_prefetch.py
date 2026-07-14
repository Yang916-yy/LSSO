#!/usr/bin/env python3
"""Prefetch Hugging Face WebDataset shards with one shared Xet runtime."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import os
import random
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hf_wds_stream import stream_file


def shard_filenames(shard_limit: int = 0, seed: int = 0) -> list[str]:
    train_count = min(1024, shard_limit) if shard_limit else 1024
    val_count = min(64, shard_limit) if shard_limit else 64
    train = [f"imagenet1k-train-{index:04d}.tar" for index in range(train_count)]
    # Approximate WebDataset's seeded shard shuffle so the initial cache is
    # useful to the first epoch instead of always starting at shard zero.
    random.Random(seed).shuffle(train)
    validation = [
        f"imagenet1k-validation-{index:02d}.tar" for index in range(val_count)
    ]
    return train + validation


def prefetch(
    repo: str,
    cache_dir: Path,
    *,
    workers: int = 8,
    download_attempts: int = 0,
    shard_limit: int = 0,
    seed: int = 0,
) -> None:
    if workers < 1:
        raise ValueError("prefetch workers must be at least one")
    filenames = shard_filenames(shard_limit=shard_limit, seed=seed)

    def fetch(filename: str) -> str:
        stream_file(
            repo,
            filename,
            cache_dir,
            emit=False,
            max_downloads=workers,
            download_attempts=download_attempts,
        )
        return filename

    completed = 0
    print(
        f"Xet prefetch started: files={len(filenames)} workers={workers}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, filename) for filename in filenames]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 16 == 0 or completed == len(filenames):
                print(
                    f"Xet prefetch progress: {completed}/{len(filenames)} shards",
                    flush=True,
                )
    print("Xet prefetch complete", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download-attempts", type=int, default=0)
    parser.add_argument("--shard-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parent-pid", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.parent_pid:
        # Set this after exec instead of using Popen(preexec_fn), which is not
        # safe once PyTorch has created native worker threads.
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
        if os.getppid() != args.parent_pid:
            return
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for gated ImageNet shards")
    prefetch(
        args.repo,
        Path(args.cache_dir),
        workers=args.workers,
        download_attempts=args.download_attempts,
        shard_limit=args.shard_limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
