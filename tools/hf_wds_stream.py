#!/usr/bin/env python3
"""Stream one gated Hugging Face WebDataset shard while caching it locally."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import os
import random
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import BinaryIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--quiet", action="store_true", help="Cache without streaming to stdout")
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=int(os.environ.get("LSSO_HF_MAX_DOWNLOADS", "8")),
        help="Maximum concurrent streaming shard requests across loader workers.",
    )
    parser.add_argument(
        "--download-attempts",
        type=int,
        default=int(os.environ.get("LSSO_HF_DOWNLOAD_ATTEMPTS", "0")),
        help="Attempts per shard; 0 retries transient network errors indefinitely.",
    )
    return parser.parse_args()


@contextmanager
def _download_slot(cache_dir: Path, slots: int):
    """Bound HTTP concurrency across independent WebDataset workers."""

    if slots < 1:
        raise ValueError("max downloads must be at least one")
    slot_dir = cache_dir / ".download-slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    first = os.getpid() % slots
    while True:
        for offset in range(slots):
            index = (first + offset) % slots
            handle = (slot_dir / f"slot-{index}.lock").open("a+b")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(random.uniform(0.10, 0.35))


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


def _copy_to_output(source: BinaryIO, output: BinaryIO) -> int:
    copied = 0
    while chunk := source.read(1024 * 1024):
        output.write(chunk)
        output.flush()
        copied += len(chunk)
    return copied


def _stream_download(
    url: str,
    token: str,
    partial: Path,
    *,
    output: BinaryIO | None,
    attempts: int = 0,
) -> None:
    """Forward bytes immediately while persisting a resumable local copy."""

    content_range_pattern = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)")
    emitted_bytes = 0
    expected_bytes: int | None = None
    last_error: BaseException | None = None
    attempt = 0

    while True:
        attempt += 1
        existing_bytes = partial.stat().st_size if partial.is_file() else 0
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept-Encoding": "identity",
        }
        if existing_bytes:
            headers["Range"] = f"bytes={existing_bytes}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                content_length = response.headers.get("Content-Length")
                content_range = response.headers.get("Content-Range")
                append = False
                if existing_bytes and status == 206 and content_range:
                    match = content_range_pattern.fullmatch(content_range.strip())
                    if not match or int(match.group(1)) != existing_bytes:
                        raise IOError(
                            f"invalid resume response for {partial.name}: {content_range}"
                        )
                    append = True
                    if match.group(3) != "*":
                        expected_bytes = int(match.group(3))
                    # A new WebDataset reader needs the cached prefix first.
                    if output is not None and emitted_bytes == 0:
                        with partial.open("rb") as prefix:
                            emitted_bytes += _copy_to_output(prefix, output)
                elif existing_bytes:
                    # The server ignored Range. This is recoverable only before
                    # any prefix has entered the current tar stream.
                    if emitted_bytes:
                        raise IOError("server ignored Range after streaming began")
                    partial.unlink(missing_ok=True)
                    existing_bytes = 0
                if expected_bytes is None and content_length:
                    expected_bytes = existing_bytes + int(content_length)

                with partial.open("ab" if append else "wb") as sink:
                    while chunk := response.read(1024 * 1024):
                        sink.write(chunk)
                        if output is not None:
                            output.write(chunk)
                            output.flush()
                            emitted_bytes += len(chunk)

            validate_tar(partial, expected_bytes=expected_bytes)
            return
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 404}:
                raise IOError(
                    f"non-retryable HTTP {exc.code} while downloading {url}"
                ) from exc
            if exc.code == 416 and partial.is_file():
                validate_tar(partial, expected_bytes=expected_bytes)
                if output is not None and emitted_bytes == 0:
                    with partial.open("rb") as prefix:
                        _copy_to_output(prefix, output)
                return
            last_error = exc
        except BrokenPipeError:
            # The training consumer exited. Keep the partial for the next run.
            raise
        except (OSError, urllib.error.URLError) as exc:
            if isinstance(exc, OSError) and exc.errno in {
                errno.EACCES,
                errno.ENOSPC,
                errno.EROFS,
            }:
                raise
            last_error = exc

        if attempts > 0 and attempt >= attempts:
            raise IOError(
                f"failed to stream a valid shard after {attempts} attempts: {url}; "
                f"last error: {type(last_error).__name__}: {last_error}"
            ) from last_error
        if attempt == 1 or attempt % 5 == 0:
            limit = "unbounded" if attempts <= 0 else str(attempts)
            print(
                f"resuming {partial.name}: attempt {attempt + 1}/{limit}; "
                f"cached={partial.stat().st_size if partial.is_file() else 0} bytes; "
                f"last error: {type(last_error).__name__}: {last_error}",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(min(2 ** min(attempt - 1, 5), 30) * random.uniform(0.75, 1.25))


def _emit_file(path: Path, output: BinaryIO) -> None:
    with path.open("rb") as source:
        _copy_to_output(source, output)


def stream_file(
    repo: str,
    filename: str,
    cache_dir: Path,
    *,
    emit: bool = True,
    max_downloads: int = 8,
    download_attempts: int = 0,
    output: BinaryIO | None = None,
) -> None:
    destination = cache_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = sys.stdout.buffer if emit and output is None else output
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
                if output is not None:
                    _emit_file(destination, output)
                return

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN is required for gated ImageNet shards")
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
        partial = destination.with_suffix(destination.suffix + ".partial")
        with _download_slot(cache_dir, max_downloads):
            _stream_download(
                url,
                token,
                partial,
                output=output,
                attempts=download_attempts,
            )
        partial.replace(destination)
        verified_path.touch()


def main() -> None:
    args = parse_args()
    stream_file(
        args.repo,
        args.filename,
        Path(args.cache_dir),
        emit=not args.quiet,
        max_downloads=args.max_downloads,
        download_attempts=args.download_attempts,
    )


if __name__ == "__main__":
    main()
