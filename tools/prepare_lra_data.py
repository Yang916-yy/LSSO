#!/usr/bin/env python3
"""Download and audit the data sources used by the four-task LRA protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.sequence_benchmarks.lra_data import (  # noqa: E402
    KAGGLE_LRA_HANDLE,
    PathfinderDataset,
    download_kaggle_lra,
    iter_listops,
    resolve_listops_files,
    resolve_pathfinder_directory,
)


EXPECTED_LISTOPS = {"train": 96_000, "val": 2_000, "test": 2_000}
EXPECTED_PATHFINDER32_HARD = 200_000


def sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_listops(data_root: Path) -> dict:
    files = resolve_listops_files(data_root)
    result = {}
    for split, path in files.items():
        labels = Counter()
        count = 0
        for _, label in iter_listops(path):
            labels[label] += 1
            count += 1
        if count != EXPECTED_LISTOPS[split]:
            raise RuntimeError(
                f"ListOps {split} has {count:,} rows; expected {EXPECTED_LISTOPS[split]:,}"
            )
        if not set(labels).issubset(set(range(10))):
            raise RuntimeError(f"ListOps {split} contains invalid labels: {sorted(labels)}")
        result[split] = {
            "path": str(path.relative_to(data_root)),
            "rows": count,
            "labels": dict(sorted(labels.items())),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


def audit_pathfinder(data_root: Path, check_images: bool) -> dict:
    source = resolve_pathfinder_directory(data_root, 32)
    dataset = PathfinderDataset(source)
    if len(dataset) != EXPECTED_PATHFINDER32_HARD:
        raise RuntimeError(
            f"Pathfinder32-hard has {len(dataset):,} usable rows; "
            f"expected {EXPECTED_PATHFINDER32_HARD:,}"
        )
    labels = Counter(dataset.labels)
    if set(labels) != {0, 1}:
        raise RuntimeError(f"Pathfinder contains invalid labels: {sorted(labels)}")
    missing = []
    if check_images:
        for relative, _ in dataset.samples:
            if not (source / relative).is_file():
                missing.append(str(relative))
                if len(missing) == 20:
                    break
    if missing:
        raise RuntimeError(f"Pathfinder images referenced by metadata are missing: {missing}")
    metadata = sorted((source / "curv_contour_length_14" / "metadata").glob("*.npy"))
    combined = hashlib.sha256()
    for path in metadata:
        combined.update(path.name.encode("utf-8"))
        combined.update(bytes.fromhex(sha256(path)))
    return {
        "path": str(source.relative_to(data_root)),
        "usable_rows": len(dataset),
        "labels": dict(sorted(labels.items())),
        "metadata_files": len(metadata),
        "metadata_sha256": combined.hexdigest(),
        "all_referenced_images_checked": check_images,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/lra")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    if not args.no_download:
        download_kaggle_lra(data_root, force=args.force_download)
    report = {
        "schema": 1,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "community_mirror": KAGGLE_LRA_HANDLE,
        "upstream_definition": "google-research/long-range-arena@cd31e5c6",
        "listops": audit_listops(data_root),
        "pathfinder32_hard": audit_pathfinder(data_root, not args.skip_image_check),
    }
    output = data_root / "source-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
