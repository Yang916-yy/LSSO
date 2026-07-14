#!/usr/bin/env python3
"""Download, validate, and serve one gated Hugging Face WebDataset shard."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import os
import random
import shutil
import sys
import tarfile
import time
from pathlib import Path


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
        help="Maximum concurrent HTTP shard downloads across loader workers.",
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
    """Bound HTTP concurrency across independent WebDataset worker processes."""

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


def _download_validated(
    repo: str,
    filename: str,
    token: str,
    cache_dir: Path,
    *,
    attempts: int = 0,
) -> Path:
    """Download through the official Hub/Xet client and validate the tar.

    ``local_dir`` is intentional: unlike the versioned Hub cache, it places the
    shard directly in our WebDataset cache and keeps only lightweight download
    metadata below ``.cache/huggingface``.  Modern huggingface_hub releases use
    hf_xet automatically, including its parallel range reconstruction and
    resumable incomplete-file handling.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub>=0.32 with hf_xet is required; install the "
            "project's experiments dependencies"
        ) from exc

    last_error: BaseException | None = None
    attempt = 0
    while True:
        attempt += 1
        downloaded: Path | None = None
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=filename,
                    repo_type="dataset",
                    token=token,
                    local_dir=cache_dir,
                )
            )
            validate_tar(downloaded)
            return downloaded
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status in {401, 404}:
                raise IOError(
                    f"non-retryable HTTP {status} while downloading "
                    f"{repo}/{filename}"
                ) from exc
            if isinstance(exc, OSError) and exc.errno in {
                errno.EACCES,
                errno.ENOSPC,
                errno.EROFS,
            }:
                raise
            last_error = exc
            # If Hub metadata pointed at a complete but corrupt local file,
            # remove it so the next attempt performs a fresh reconstruction.
            # In-progress Xet state lives elsewhere and remains resumable.
            if downloaded is not None:
                downloaded.unlink(missing_ok=True)
        if attempts > 0 and attempt >= attempts:
            raise IOError(
                f"failed to download a valid shard after {attempts} attempts: "
                f"{repo}/{filename}; "
                f"last error: {type(last_error).__name__}: {last_error}"
            ) from last_error
        if attempt == 1 or attempt % 5 == 0:
            limit = "unbounded" if attempts <= 0 else str(attempts)
            print(
                f"retrying {filename}: attempt {attempt + 1}/{limit}; "
                f"last error: {type(last_error).__name__}: {last_error}",
                file=sys.stderr,
                flush=True,
            )
        # Jitter prevents all WebDataset workers from retrying a rate-limit
        # response at the same instant.
        time.sleep(min(2 ** min(attempt - 1, 5), 30) * random.uniform(0.75, 1.25))


def _emit_file(path: Path) -> None:
    with path.open("rb") as source:
        try:
            shutil.copyfileobj(source, sys.stdout.buffer, length=1024 * 1024)
        except BrokenPipeError:
            pass


def stream_file(
    repo: str,
    filename: str,
    cache_dir: Path,
    *,
    emit: bool = True,
    max_downloads: int = 8,
    download_attempts: int = 0,
) -> None:
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
        with _download_slot(cache_dir, max_downloads):
            downloaded = _download_validated(
                repo,
                filename,
                token,
                cache_dir,
                attempts=download_attempts,
            )
        if downloaded.resolve() != destination.resolve():
            raise RuntimeError(
                f"Hub local_dir returned {downloaded}, expected {destination}"
            )
        # Custom urllib versions used this sibling partial.  Xet maintains its
        # own resumable state in local_dir/.cache/huggingface, so it is obsolete.
        destination.with_suffix(destination.suffix + ".partial").unlink(
            missing_ok=True
        )
        verified_path.touch()
        if emit:
            _emit_file(destination)


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
