from __future__ import annotations

import json
import hashlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CharacterVocabulary:
    PAD = "<pad>"
    UNK = "<unk>"
    EOS = "<eos>"

    def __init__(self, tokens: Sequence[str]) -> None:
        ordered = [self.PAD, self.UNK, self.EOS]
        ordered.extend(token for token in tokens if token not in ordered)
        self.tokens = ordered
        self.lookup = {token: index for index, token in enumerate(ordered)}
        self.pad_token_id = self.lookup[self.PAD]
        self.unk_token_id = self.lookup[self.UNK]
        self.eos_token_id = self.lookup[self.EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_chars(self, text: str, max_length: int) -> list[int]:
        room = max(0, max_length - 1)
        values = [self.lookup.get(token, self.unk_token_id) for token in text[:room]]
        values.append(self.eos_token_id)
        return values

    def encode_listops(self, text: str, max_length: int) -> list[int]:
        normalized = text.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None})
        room = max(0, max_length - 1)
        values = [self.lookup.get(token, self.unk_token_id) for token in normalized.split()[:room]]
        values.append(self.eos_token_id)
        return values

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.tokens, ensure_ascii=False), encoding="utf-8")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.tokens, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "CharacterVocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8"))[3:])

    @classmethod
    def from_texts(cls, texts: Iterable[str], min_frequency: int = 1) -> "CharacterVocabulary":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(text)
        tokens = sorted(token for token, count in counts.items() if count >= min_frequency)
        return cls(tokens)

    @classmethod
    def from_listops(cls, texts: Iterable[str]) -> "CharacterVocabulary":
        tokens = set()
        for text in texts:
            normalized = text.translate({ord("]"): ord("X"), ord("("): None, ord(")"): None})
            tokens.update(normalized.split())
        return cls(sorted(tokens))


def iter_listops(path: Path) -> Iterator[tuple[str, int]]:
    with path.open("r", encoding="utf-8") as stream:
        header = stream.readline().rstrip("\n").split("\t")
        try:
            source_index, target_index = header.index("Source"), header.index("Target")
        except ValueError as error:
            raise ValueError(f"invalid ListOps header in {path}: {header}") from error
        for line in stream:
            columns = line.rstrip("\n").split("\t")
            if len(columns) <= max(source_index, target_index):
                continue
            yield columns[source_index], int(columns[target_index])


def iter_aan(path: Path) -> Iterator[tuple[str, str, int]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            columns = line.rstrip("\n").split("\t", 4)
            if len(columns) != 5:
                continue
            label, _, _, first, second = columns
            try:
                yield first, second, int(label)
            except ValueError:
                continue


class PackedTokenDataset(Dataset):
    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.offsets = np.load(prefix.with_suffix(".offsets.npy"), mmap_mode="r")
        self.labels_array = np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r")
        self.tokens = np.memmap(prefix.with_suffix(".tokens.bin"), dtype=np.uint16, mode="r")
        self.lengths = np.diff(self.offsets).astype(np.int64).tolist()
        self.labels = self.labels_array.astype(np.int64).tolist()

    def __len__(self) -> int:
        return len(self.labels_array)

    def __getitem__(self, index: int):
        start, end = int(self.offsets[index]), int(self.offsets[index + 1])
        tokens = torch.from_numpy(np.asarray(self.tokens[start:end]).copy()).long()
        return tokens, int(self.labels_array[index])


class PackedPairTokenDataset(Dataset):
    def __init__(self, prefix: Path) -> None:
        self.prefix = prefix
        self.first_offsets = np.load(prefix.with_suffix(".first.offsets.npy"), mmap_mode="r")
        self.second_offsets = np.load(prefix.with_suffix(".second.offsets.npy"), mmap_mode="r")
        self.labels_array = np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r")
        self.first = np.memmap(prefix.with_suffix(".first.tokens.bin"), dtype=np.uint16, mode="r")
        self.second = np.memmap(prefix.with_suffix(".second.tokens.bin"), dtype=np.uint16, mode="r")
        self.lengths = np.maximum(
            np.diff(self.first_offsets), np.diff(self.second_offsets)
        ).astype(np.int64).tolist()
        self.labels = self.labels_array.astype(np.int64).tolist()

    def __len__(self) -> int:
        return len(self.labels_array)

    def __getitem__(self, index: int):
        first_start, first_end = int(self.first_offsets[index]), int(self.first_offsets[index + 1])
        second_start, second_end = int(self.second_offsets[index]), int(self.second_offsets[index + 1])
        first = torch.from_numpy(np.asarray(self.first[first_start:first_end]).copy()).long()
        second = torch.from_numpy(np.asarray(self.second[second_start:second_end]).copy()).long()
        return first, second, int(self.labels_array[index])


def build_packed_tokens(
    prefix: Path,
    rows: Iterable[tuple[str, int]],
    encode: Callable[[str], Sequence[int]],
    manifest: dict | None = None,
) -> PackedTokenDataset:
    required = [
        prefix.with_suffix(".tokens.bin"),
        prefix.with_suffix(".offsets.npy"),
        prefix.with_suffix(".labels.npy"),
    ]
    manifest_path = prefix.with_suffix(".manifest.json")
    cache_matches = manifest is None or (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    )
    if all(path.exists() for path in required) and cache_matches:
        return PackedTokenDataset(prefix)
    for path in [*required, manifest_path]:
        path.unlink(missing_ok=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    temporary = prefix.with_suffix(".tokens.bin.tmp")
    offsets, labels = [0], []
    with temporary.open("wb") as stream:
        for text, label in rows:
            values = np.asarray(encode(text), dtype=np.uint16)
            values.tofile(stream)
            offsets.append(offsets[-1] + len(values))
            labels.append(int(label))
    temporary.replace(prefix.with_suffix(".tokens.bin"))
    np.save(prefix.with_suffix(".offsets.npy"), np.asarray(offsets, dtype=np.int64))
    np.save(prefix.with_suffix(".labels.npy"), np.asarray(labels, dtype=np.int64))
    if manifest is not None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    return PackedTokenDataset(prefix)


def build_packed_pairs(
    prefix: Path,
    rows: Iterable[tuple[str, str, int]],
    encode: Callable[[str], Sequence[int]],
    manifest: dict | None = None,
) -> PackedPairTokenDataset:
    required = [
        prefix.with_suffix(".first.tokens.bin"),
        prefix.with_suffix(".second.tokens.bin"),
        prefix.with_suffix(".first.offsets.npy"),
        prefix.with_suffix(".second.offsets.npy"),
        prefix.with_suffix(".labels.npy"),
    ]
    manifest_path = prefix.with_suffix(".manifest.json")
    cache_matches = manifest is None or (
        manifest_path.exists()
        and json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    )
    if all(path.exists() for path in required) and cache_matches:
        return PackedPairTokenDataset(prefix)
    for path in [*required, manifest_path]:
        path.unlink(missing_ok=True)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    first_tmp = prefix.with_suffix(".first.tokens.bin.tmp")
    second_tmp = prefix.with_suffix(".second.tokens.bin.tmp")
    first_offsets, second_offsets, labels = [0], [0], []
    with first_tmp.open("wb") as first_stream, second_tmp.open("wb") as second_stream:
        for first, second, label in rows:
            first_values = np.asarray(encode(first), dtype=np.uint16)
            second_values = np.asarray(encode(second), dtype=np.uint16)
            first_values.tofile(first_stream)
            second_values.tofile(second_stream)
            first_offsets.append(first_offsets[-1] + len(first_values))
            second_offsets.append(second_offsets[-1] + len(second_values))
            labels.append(int(label))
    first_tmp.replace(prefix.with_suffix(".first.tokens.bin"))
    second_tmp.replace(prefix.with_suffix(".second.tokens.bin"))
    np.save(prefix.with_suffix(".first.offsets.npy"), np.asarray(first_offsets, dtype=np.int64))
    np.save(prefix.with_suffix(".second.offsets.npy"), np.asarray(second_offsets, dtype=np.int64))
    np.save(prefix.with_suffix(".labels.npy"), np.asarray(labels, dtype=np.int64))
    if manifest is not None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    return PackedPairTokenDataset(prefix)


def source_signature(path: Path) -> dict[str, int | str]:
    """Fast source identity; avoids re-hashing multi-gigabyte AAN files each launch."""
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


class PathfinderDataset(Dataset):
    blacklist = {"pathfinder32/curv_baseline/imgs/0/sample_172.png"}

    def __init__(self, data_dir: Path) -> None:
        from PIL import Image  # noqa: F401 - validate the optional dependency early

        self.data_dir = data_dir
        difficulty = data_dir / "curv_contour_length_14"
        metadata_paths = sorted(
            (difficulty / "metadata").glob("*.npy"), key=lambda path: int(path.stem)
        )
        if not metadata_paths:
            raise FileNotFoundError(f"no Pathfinder metadata found under {difficulty}")
        self.samples: list[tuple[Path, int]] = []
        for metadata_path in metadata_paths:
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                columns = line.split()
                relative = Path("curv_contour_length_14") / columns[0] / columns[1]
                named = str(Path(data_dir.name) / relative).replace("\\", "/")
                if named not in self.blacklist:
                    self.samples.append((relative, int(columns[3])))
        self.labels = [label for _, label in self.samples]
        self.lengths = [self.resolution**2] * len(self.samples)

    @property
    def resolution(self) -> int:
        return int(self.data_dir.name.removeprefix("pathfinder"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

        path, label = self.samples[index]
        with Image.open(self.data_dir / path) as image:
            values = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        return torch.from_numpy(values.reshape(-1, 1)), label
