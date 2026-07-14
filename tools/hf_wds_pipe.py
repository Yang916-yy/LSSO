#!/usr/bin/env python3
"""Reliably stream one gated Hugging Face shard to stdout.

Hugging Face's Xet bridge can occasionally reject a freshly redirected shard
request with HTTP 403.  curl does not retry 403 by default.  Retrying a curl
which already wrote a partial tar to stdout is unsafe, however, because it
would concatenate a second tar prefix onto WebDataset's cache file.  This
wrapper therefore retries only requests which failed before emitting bytes.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--attempts", type=int, default=30)
    return parser.parse_args()


def stream_shard(repo: str, filename: str, *, attempts: int = 30) -> None:
    if attempts < 1:
        raise ValueError("attempts must be at least one")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for gated ImageNet shards")
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"

    for attempt in range(1, attempts + 1):
        process = subprocess.Popen(
            [
                "curl",
                "--location",
                "--fail",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "30",
                "--speed-time",
                "120",
                "--speed-limit",
                "1024",
                "--config",
                "-",
                url,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        # Keep the token out of ps output and exception command strings.
        process.stdin.write(f'header = "Authorization: Bearer {token}"\n'.encode())
        process.stdin.close()

        emitted = 0
        try:
            while chunk := process.stdout.read(1024 * 1024):
                sys.stdout.buffer.write(chunk)
                emitted += len(chunk)
            sys.stdout.buffer.flush()
        except BrokenPipeError:
            process.terminate()
            process.wait(timeout=10)
            return

        stderr = process.stderr.read().decode(errors="replace").strip()
        returncode = process.wait()
        if returncode == 0:
            if emitted == 0:
                raise IOError(f"empty shard response for {filename}")
            return

        if emitted:
            raise IOError(
                f"shard transfer failed after emitting {emitted} bytes; refusing an "
                f"unsafe in-stream retry: {stderr}"
            )
        lowered = stderr.lower()
        if "error: 401" in lowered or "error: 404" in lowered:
            raise PermissionError(f"non-retryable shard request: {stderr}")
        if attempt == attempts:
            raise IOError(
                f"shard request failed before receiving bytes after {attempts} "
                f"attempts: {stderr}"
            )
        delay = min(2 ** min(attempt - 1, 5), 30) * random.uniform(0.75, 1.25)
        print(
            f"retrying {filename} after zero-byte failure "
            f"({attempt}/{attempts}): {stderr}",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)


def main() -> None:
    args = parse_args()
    stream_shard(args.repo, args.filename, attempts=args.attempts)


if __name__ == "__main__":
    main()
