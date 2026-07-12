#!/usr/bin/env python3
"""Stream one gated Hugging Face shard to stdout while atomically caching it."""

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
    parser.add_argument("--cache-dir", required=True)
    return parser.parse_args()


def stream_file(repo: str, filename: str, cache_dir: Path) -> None:
    destination = cache_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file():
            with destination.open("rb") as source:
                shutil.copyfileobj(source, sys.stdout.buffer, length=1024 * 1024)
            return

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for gated ImageNet shards")
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as sink:
                while chunk := response.read(1024 * 1024):
                    sink.write(chunk)
                    sys.stdout.buffer.write(chunk)
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise


def main() -> None:
    args = parse_args()
    stream_file(args.repo, args.filename, Path(args.cache_dir))


if __name__ == "__main__":
    main()
