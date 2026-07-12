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
    parser.add_argument("--quiet", action="store_true", help="Cache without streaming to stdout")
    return parser.parse_args()


def stream_file(repo: str, filename: str, cache_dir: Path, *, emit: bool = True) -> None:
    destination = cache_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file():
            if emit:
                with destination.open("rb") as source:
                    try:
                        shutil.copyfileobj(source, sys.stdout.buffer, length=1024 * 1024)
                    except BrokenPipeError:
                        pass
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
                output_open = emit
                while chunk := response.read(1024 * 1024):
                    sink.write(chunk)
                    if output_open:
                        try:
                            sys.stdout.buffer.write(chunk)
                        except BrokenPipeError:
                            # A bounded smoke/eval iterator may close early. Finish
                            # the shard so the next run still gets a valid cache hit.
                            output_open = False
            partial.replace(destination)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise


def main() -> None:
    args = parse_args()
    stream_file(args.repo, args.filename, Path(args.cache_dir), emit=not args.quiet)


if __name__ == "__main__":
    main()
