#!/usr/bin/env python3
"""Download, validate, and serve one gated Hugging Face WebDataset shard."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--quiet", action="store_true", help="Cache without streaming to stdout")
    return parser.parse_args()


def validate_tar(path: Path, *, expected_bytes: int | None = None) -> None:
    actual_bytes = path.stat().st_size
    if expected_bytes is not None and actual_bytes != expected_bytes:
        raise IOError(
            f"truncated shard {path.name}: expected {expected_bytes} bytes, "
            f"received {actual_bytes}"
        )
    if actual_bytes == 0:
        raise IOError(f"empty shard: {path.name}")
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
    except tarfile.TarError as exc:
        raise IOError(f"invalid tar shard {path.name}: {exc}") from exc
    if not members:
        raise IOError(f"tar shard contains no members: {path.name}")


def _download_validated(
    url: str,
    token: str,
    partial: Path,
    *,
    attempts: int = 5,
) -> None:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        partial.unlink(missing_ok=True)
        request = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as sink:
                content_length = response.headers.get("Content-Length")
                expected_bytes = int(content_length) if content_length else None
                while chunk := response.read(1024 * 1024):
                    sink.write(chunk)
            validate_tar(partial, expected_bytes=expected_bytes)
            return
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise IOError(f"failed to download a valid shard after {attempts} attempts: {url}") from last_error


def _emit_file(path: Path) -> None:
    with path.open("rb") as source:
        try:
            shutil.copyfileobj(source, sys.stdout.buffer, length=1024 * 1024)
        except BrokenPipeError:
            pass


def stream_file(repo: str, filename: str, cache_dir: Path, *, emit: bool = True) -> None:
    destination = cache_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    verified_path = destination.with_suffix(destination.suffix + ".verified")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file():
            if not verified_path.is_file():
                try:
                    validate_tar(destination)
                except OSError:
                    destination.unlink(missing_ok=True)
                else:
                    verified_path.touch()
            if destination.is_file():
                if emit:
                    _emit_file(destination)
                return

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for gated ImageNet shards")
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            _download_validated(url, token, partial)
            partial.replace(destination)
            verified_path.touch()
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        if emit:
            _emit_file(destination)


def main() -> None:
    args = parse_args()
    stream_file(args.repo, args.filename, Path(args.cache_dir), emit=not args.quiet)


if __name__ == "__main__":
    main()
