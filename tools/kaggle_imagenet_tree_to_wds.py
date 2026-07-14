"""Convert one deterministic part of Kaggle-mounted ImageNet into WebDataset shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kaggle_imagenet_to_wds import AtomicShardWriter, TRAIN_SAMPLES, VAL_SAMPLES


SYNSET_PATTERN = re.compile(r"n\d{8}")


def _flat_synsets(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.iterdir() if path.is_dir() and SYNSET_PATTERN.fullmatch(path.name)
    )


def locate_imagenet(input_root: Path) -> tuple[Path, Path | None]:
    """Locate either CLS-LOC or ``ILSVRC2012/<synset>`` without a broad scan."""

    input_root = input_root.resolve()
    bases = [input_root]
    bases.extend(path for path in input_root.iterdir() if path.is_dir())
    for base in bases:
        data = base / "ILSVRC" / "Data" / "CLS-LOC"
        mapping = base / "LOC_synset_mapping.txt"
        validation = base / "LOC_val_solution.csv"
        if data.is_dir() and mapping.is_file() and validation.is_file():
            return data, base
    flat_candidates = bases + [base / "ILSVRC2012" for base in bases]
    for candidate in flat_candidates:
        if len(_flat_synsets(candidate)) == 1000:
            return candidate, None
    raise FileNotFoundError(
        "could not find CLS-LOC or a flat ILSVRC2012 directory containing "
        f"1,000 synset folders under {input_root}"
    )


def load_local_labels(
    metadata_root: Path | None, data_root: Path | None = None
) -> tuple[list[str], dict[str, int]]:
    if metadata_root is None:
        if data_root is None:
            raise ValueError("data_root is required for a flat ImageNet tree")
        synsets = [path.name for path in _flat_synsets(data_root)]
        if len(synsets) != 1000:
            raise RuntimeError(f"expected 1000 unique synsets, found {len(synsets)}")
        return synsets, {}

    mapping = (metadata_root / "LOC_synset_mapping.txt").read_text(
        encoding="utf-8-sig"
    )
    synsets = [line.split(maxsplit=1)[0] for line in mapping.splitlines() if line.strip()]
    if len(synsets) != 1000 or len(set(synsets)) != 1000:
        raise RuntimeError(f"expected 1000 unique synsets, found {len(synsets)}")
    class_to_index = {synset: index for index, synset in enumerate(synsets)}

    validation: dict[str, int] = {}
    solution = (metadata_root / "LOC_val_solution.csv").read_text(
        encoding="utf-8-sig"
    )
    for row in csv.DictReader(io.StringIO(solution)):
        tokens = row["PredictionString"].strip().split()
        if not tokens:
            raise RuntimeError(f"empty validation label for {row['ImageId']}")
        validation[row["ImageId"]] = class_to_index[tokens[0]]
    if len(validation) != VAL_SAMPLES:
        raise RuntimeError(f"expected {VAL_SAMPLES} validation labels, found {len(validation)}")
    return synsets, validation


def class_bounds(part_id: int, num_train_parts: int, num_classes: int) -> tuple[int, int]:
    if num_train_parts <= 0:
        raise ValueError("num_train_parts must be positive")
    if not 0 <= part_id < num_train_parts:
        raise ValueError(f"train part must be in [0, {num_train_parts})")
    return (
        part_id * num_classes // num_train_parts,
        (part_id + 1) * num_classes // num_train_parts,
    )


def _jpeg_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg"}
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=4 << 20) as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(output: Path, values: dict) -> Path:
    manifest = output / "manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    return manifest


def build_part(
    data_root: Path,
    output: Path,
    *,
    part_id: int,
    num_train_parts: int,
    synsets: list[str],
    validation_labels: dict[str, int],
    maxcount: int,
) -> dict:
    validation_part = part_id == num_train_parts
    if not 0 <= part_id <= num_train_parts:
        raise ValueError(f"part_id must be in [0, {num_train_parts}]")
    split = "validation" if validation_part else "train"
    name_tag = "val" if validation_part else f"p{part_id:02d}"
    output.mkdir(parents=True, exist_ok=True)
    existing_manifest = output / "manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("part_id") != part_id:
            raise RuntimeError(f"output contains a different part: {existing_manifest}")
        return existing

    writer = AtomicShardWriter(
        output, split, maxcount=maxcount, name_tag=name_tag
    )
    seen = 0
    if validation_part:
        if not validation_labels or not (data_root / "val").is_dir():
            raise RuntimeError("this ImageNet mount does not contain a labeled validation split")
        selected_classes = [0, len(synsets)]
        files = _jpeg_files(data_root / "val")
        expected = len(validation_labels)
        if len(files) != expected:
            raise RuntimeError(f"expected {expected} validation JPEGs, found {len(files)}")
        for path in files:
            seen += 1
            if seen <= writer.completed:
                continue
            writer.write(path.stem, path.read_bytes(), validation_labels[path.stem])
            if seen % 10_000 == 0:
                print(f"packed validation {seen}/{expected}", flush=True)
    else:
        start, end = class_bounds(part_id, num_train_parts, len(synsets))
        selected_classes = [start, end]
        train_root = data_root / "train" if (data_root / "train").is_dir() else data_root
        expected = sum(len(_jpeg_files(train_root / synsets[i])) for i in range(start, end))
        for class_index in range(start, end):
            synset = synsets[class_index]
            for path in _jpeg_files(train_root / synset):
                seen += 1
                if seen <= writer.completed:
                    continue
                writer.write(path.stem, path.read_bytes(), class_index)
                if seen % 10_000 == 0:
                    print(f"packed train part {part_id}: {seen}/{expected}", flush=True)
    writer.commit()
    if seen != expected or writer.completed != expected:
        raise RuntimeError(
            f"part count mismatch: seen={seen}, committed={writer.completed}, expected={expected}"
        )

    shards = sorted(output.glob(f"imagenet1k-{split}-{name_tag}-*.tar"))
    shard_records = []
    for index, shard in enumerate(shards, 1):
        print(f"hashing {index}/{len(shards)}: {shard.name}", flush=True)
        shard_records.append(
            {"name": shard.name, "bytes": shard.stat().st_size, "sha256": _sha256(shard)}
        )
    manifest = {
        "format": "webdataset",
        "source": "kaggle:imagenet-object-localization-challenge",
        "part_id": part_id,
        "num_train_parts": num_train_parts,
        "split": split,
        "class_index_range": selected_classes,
        "samples": expected,
        "shards": shard_records,
    }
    _write_manifest(output, manifest)
    print(
        f"part ready: split={split} samples={expected} "
        f"size_gib={sum(x['bytes'] for x in shard_records) / 2**30:.2f}",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part-id", type=int, required=True)
    parser.add_argument("--num-train-parts", type=int, default=9)
    parser.add_argument("--maxcount", type=int, default=8192)
    args = parser.parse_args()
    data_root, metadata_root = locate_imagenet(args.input)
    synsets, validation_labels = load_local_labels(metadata_root, data_root)
    manifest = build_part(
        data_root,
        args.output,
        part_id=args.part_id,
        num_train_parts=args.num_train_parts,
        synsets=synsets,
        validation_labels=validation_labels,
        maxcount=args.maxcount,
    )
    if args.part_id < args.num_train_parts:
        # Every train class belongs to exactly one contiguous part; the global total
        # is checked after all part manifests are downloaded together.
        assert manifest["samples"] <= TRAIN_SAMPLES


if __name__ == "__main__":
    main()
