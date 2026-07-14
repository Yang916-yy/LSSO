#!/usr/bin/env python3
"""Stream one Hugging Face WebDataset shard while atomically caching it."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--cache-dir", required=True, type=Path)
    return parser.parse_args()


def _emit(path: Path) -> None:
    with path.open("rb") as source:
        try:
            shutil.copyfileobj(source, sys.stdout.buffer, length=1 << 20)
        except BrokenPipeError:
            pass


def stream_file(repo: str, filename: str, cache_dir: Path) -> None:
    """Use one request and one cache owner; fail fast on any transfer error."""

    destination = cache_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file():
            _emit(destination)
            return

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for gated ImageNet shards")
        request = urllib.request.Request(
            f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept-Encoding": "identity",
            },
        )
        partial.unlink(missing_ok=True)
        received = 0
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open(
                "wb", buffering=4 << 20
            ) as sink:
                expected = response.headers.get("Content-Length")
                while chunk := response.read(1 << 20):
                    sink.write(chunk)
                    sys.stdout.buffer.write(chunk)
                    received += len(chunk)
                sys.stdout.buffer.flush()
                sink.flush()
                os.fsync(sink.fileno())
            if expected is not None and received != int(expected):
                raise IOError(
                    f"truncated shard {filename}: expected {expected}, received {received}"
                )
            partial.replace(destination)
        except BrokenPipeError:
            partial.unlink(missing_ok=True)
            return
        except BaseException:
            partial.unlink(missing_ok=True)
            raise


def main() -> None:
    args = parse_args()
    stream_file(args.repo, args.filename, args.cache_dir)


if __name__ == "__main__":
    main()
