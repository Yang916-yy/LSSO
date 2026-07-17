#!/usr/bin/env python3
"""Fail when generated artifacts or unexpectedly large files enter Git."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
FORBIDDEN_PREFIXES = (
    ".venv/",
    "artifacts/",
    "build/",
    "data/",
    "dist/",
    "runs/",
    "runs_archive/",
    "tmp/",
)
FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".log",
    ".o",
    ".partial",
    ".pid",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".so",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode(errors="surrogateescape")
    return [ROOT / name for name in output.split("\0") if name]


def main() -> int:
    failures: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(FORBIDDEN_PREFIXES):
            failures.append(f"generated path is tracked: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"generated file type is tracked: {relative}")
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            size_mib = path.stat().st_size / 2**20
            failures.append(f"tracked file is {size_mib:.1f} MiB: {relative}")

    if failures:
        print("Repository hygiene check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Repository hygiene check passed ({len(tracked_files())} tracked files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
