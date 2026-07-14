"""Stream Kaggle ImageNet into local WebDataset shards without storing the source archive."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tarfile
import time
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


COMPETITION = "imagenet-object-localization-challenge"
SOURCE_FILE = "imagenet_object_localization_patched2019.tar.gz"
TRAIN_SAMPLES = 1_281_167
VAL_SAMPLES = 50_000
_TRAIN_RE = re.compile(r"(?:^|/)ILSVRC/Data/CLS-LOC/train/([^/]+)/([^/]+\.(?:JPEG|jpg|jpeg))$")
_VAL_RE = re.compile(r"(?:^|/)ILSVRC/Data/CLS-LOC/val/([^/]+\.(?:JPEG|jpg|jpeg))$")


def _credentials() -> tuple[dict[str, str], tuple[str, str] | None]:
    token = os.environ.get("KAGGLE_API_TOKEN")
    token_path = Path.home() / ".kaggle" / "access_token"
    if not token and token_path.is_file():
        token = token_path.read_text(encoding="utf-8").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}, None

    legacy_path = Path.home() / ".kaggle" / "kaggle.json"
    if legacy_path.is_file():
        values = json.loads(legacy_path.read_text(encoding="utf-8"))
        if values.get("username") and values.get("key"):
            return {}, (values["username"], values["key"])
    raise RuntimeError(
        "set KAGGLE_API_TOKEN or create ~/.kaggle/access_token after accepting "
        "the ImageNet competition rules"
    )


def _session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=8,
        connect=8,
        read=8,
        backoff_factor=1.0,
        status_forcelist=(408, 429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _file_url(competition: str, filename: str) -> str:
    return (
        "https://www.kaggle.com/api/v1/competitions/data/download/"
        f"{quote(competition, safe='')}/{quote(filename, safe='')}"
    )


def _get(session: requests.Session, filename: str, *, stream: bool) -> requests.Response:
    headers, auth = _credentials()
    response = session.get(
        _file_url(COMPETITION, filename),
        headers=headers,
        auth=auth,
        allow_redirects=True,
        stream=stream,
        timeout=(30, 120),
    )
    response.raise_for_status()
    return response


def _small_file(session: requests.Session, filename: str) -> bytes:
    response = _get(session, filename, stream=False)
    payload = response.content
    response.close()
    if payload[:4] == b"PK\x03\x04":
        import zipfile

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            candidates = [name for name in archive.namelist() if not name.endswith("/")]
            if len(candidates) != 1:
                raise RuntimeError(f"unexpected Kaggle wrapper for {filename}: {candidates}")
            payload = archive.read(candidates[0])
    return payload


def load_labels(session: requests.Session) -> tuple[dict[str, int], dict[str, int]]:
    mapping = _small_file(session, "LOC_synset_mapping.txt").decode("utf-8-sig")
    synsets = [line.split(maxsplit=1)[0] for line in mapping.splitlines() if line.strip()]
    if len(synsets) != 1000 or len(set(synsets)) != 1000:
        raise RuntimeError(f"expected 1000 unique synsets, found {len(synsets)}")
    class_to_index = {synset: index for index, synset in enumerate(synsets)}

    solution = _small_file(session, "LOC_val_solution.csv").decode("utf-8-sig")
    validation: dict[str, int] = {}
    for row in csv.DictReader(io.StringIO(solution)):
        tokens = row["PredictionString"].strip().split()
        if not tokens:
            raise RuntimeError(f"empty validation label for {row['ImageId']}")
        validation[row["ImageId"]] = class_to_index[tokens[0]]
    if len(validation) != VAL_SAMPLES:
        raise RuntimeError(f"expected {VAL_SAMPLES} validation labels, found {len(validation)}")
    return class_to_index, validation


class AtomicShardWriter:
    def __init__(
        self, root: Path, split: str, maxcount: int, *, name_tag: str = ""
    ) -> None:
        self.root = root
        self.split = split
        self.maxcount = maxcount
        self.name_tag = name_tag
        self.file_split = f"{split}-{name_tag}" if name_tag else split
        self.root.mkdir(parents=True, exist_ok=True)
        for partial in self.root.glob(f"imagenet1k-{self.file_split}-*.tar.partial"):
            partial.unlink()
        self.shard_index, self.completed = self._resume_state()
        self.current_count = 0
        self.stream: BinaryIO | None = None
        self.archive: tarfile.TarFile | None = None
        self.partial: Path | None = None

    def _resume_state(self) -> tuple[int, int]:
        shard_index = completed = 0
        while True:
            stem = f"imagenet1k-{self.file_split}-{shard_index:05d}"
            shard = self.root / f"{stem}.tar"
            marker = self.root / f"{stem}.complete.json"
            if not shard.exists() and not marker.exists():
                break
            if not shard.is_file() or not marker.is_file():
                shard.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                break
            state = json.loads(marker.read_text(encoding="utf-8"))
            if state.get("split") != self.split or state.get("shard") != shard_index:
                raise RuntimeError(f"invalid completion marker: {marker}")
            completed += int(state["samples"])
            shard_index += 1
        return shard_index, completed

    def _open(self) -> None:
        stem = f"imagenet1k-{self.file_split}-{self.shard_index:05d}"
        self.partial = self.root / f"{stem}.tar.partial"
        self.stream = self.partial.open("wb", buffering=4 << 20)
        self.archive = tarfile.open(fileobj=self.stream, mode="w|")
        self.current_count = 0

    def write(self, key: str, jpeg: bytes, label: int) -> None:
        if self.archive is None:
            self._open()
        assert self.archive is not None
        self._add(f"{key}.jpg", jpeg)
        self._add(f"{key}.cls", str(label).encode("ascii"))
        self.current_count += 1
        if self.current_count >= self.maxcount:
            self.commit()

    def _add(self, name: str, payload: bytes) -> None:
        assert self.archive is not None
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        info.mtime = 0
        info.mode = 0o644
        self.archive.addfile(info, io.BytesIO(payload))

    def commit(self) -> None:
        if self.archive is None or self.current_count == 0:
            return
        assert self.stream is not None and self.partial is not None
        count = self.current_count
        self.archive.close()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        final = self.partial.with_suffix("")
        self.partial.replace(final)
        marker = final.with_suffix(".complete.json")
        temporary = marker.with_suffix(marker.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"split": self.split, "shard": self.shard_index, "samples": count}),
            encoding="utf-8",
        )
        temporary.replace(marker)
        print(f"committed {final.name}: {count} samples", flush=True)
        self.completed += count
        self.shard_index += 1
        self.current_count = 0
        self.archive = None
        self.stream = None
        self.partial = None

    def abort_partial(self) -> None:
        if self.archive is not None:
            self.archive.close()
        if self.stream is not None and not self.stream.closed:
            self.stream.close()
        if self.partial is not None:
            self.partial.unlink(missing_ok=True)
        self.archive = None
        self.stream = None
        self.partial = None
        self.current_count = 0


def convert_tar_stream(
    stream: BinaryIO,
    output: Path,
    class_to_index: dict[str, int],
    validation_labels: dict[str, int],
    *,
    maxcount: int = 10_000,
) -> tuple[int, int]:
    train_writer = AtomicShardWriter(output, "train", maxcount)
    val_writer = AtomicShardWriter(output, "validation", maxcount)
    seen = {"train": 0, "validation": 0}
    started = time.monotonic()
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as source:
            for member in source:
                if not member.isfile():
                    continue
                train_match = _TRAIN_RE.search(member.name)
                val_match = _VAL_RE.search(member.name)
                if train_match:
                    split = "train"
                    key = Path(train_match.group(2)).stem
                    label = class_to_index[train_match.group(1)]
                    writer = train_writer
                elif val_match:
                    split = "validation"
                    key = Path(val_match.group(1)).stem
                    label = validation_labels[key]
                    writer = val_writer
                else:
                    continue
                seen[split] += 1
                if seen[split] <= writer.completed:
                    continue
                extracted = source.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"could not read {member.name}")
                writer.write(key, extracted.read(), label)
                total = seen["train"] + seen["validation"]
                if total % 10_000 == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"streamed train={seen['train']} val={seen['validation']} "
                        f"images_per_second={total / elapsed:.1f}",
                        flush=True,
                    )
        train_writer.commit()
        val_writer.commit()
    except BaseException:
        train_writer.abort_partial()
        val_writer.abort_partial()
        raise
    return seen["train"], seen["validation"]


def _write_manifest(output: Path, train: int, validation: int) -> None:
    manifest = output / "dataset.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "source": f"kaggle:{COMPETITION}/{SOURCE_FILE}",
                "train_samples": train,
                "validation_samples": validation,
                "format": "webdataset",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maxcount", type=int, default=10_000)
    parser.add_argument("--attempts", type=int, default=0, help="0 retries forever")
    args = parser.parse_args()
    if args.maxcount <= 0:
        parser.error("--maxcount must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    session = _session()
    class_to_index, validation_labels = load_labels(session)

    attempt = 0
    while True:
        attempt += 1
        response = None
        try:
            print(f"opening Kaggle ImageNet stream (attempt {attempt})", flush=True)
            response = _get(session, SOURCE_FILE, stream=True)
            response.raw.decode_content = False
            prefix = response.raw.read(4)
            if prefix != b"\x1f\x8b\x08\x00" and prefix[:2] != b"\x1f\x8b":
                raise RuntimeError(
                    f"expected a direct gzip stream from Kaggle, got magic {prefix.hex()}"
                )
            stream = io.BufferedReader(_PrefixedReader(prefix, response.raw), buffer_size=4 << 20)
            train, validation = convert_tar_stream(
                stream,
                args.output,
                class_to_index,
                validation_labels,
                maxcount=args.maxcount,
            )
            if train != TRAIN_SAMPLES or validation != VAL_SAMPLES:
                raise RuntimeError(
                    f"sample count mismatch: train={train}/{TRAIN_SAMPLES}, "
                    f"validation={validation}/{VAL_SAMPLES}"
                )
            _write_manifest(args.output, train, validation)
            print(f"ImageNet WebDataset ready: {args.output}", flush=True)
            return
        except KeyboardInterrupt:
            raise
        except Exception as error:
            print(f"stream attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
            if args.attempts and attempt >= args.attempts:
                raise
            time.sleep(min(60, 5 * attempt))
        finally:
            if response is not None:
                response.close()


class _PrefixedReader(io.RawIOBase):
    def __init__(self, prefix: bytes, source: BinaryIO) -> None:
        self.prefix = memoryview(prefix)
        self.source = source

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        target = memoryview(buffer)
        written = 0
        if self.prefix:
            count = min(len(target), len(self.prefix))
            target[:count] = self.prefix[:count]
            self.prefix = self.prefix[count:]
            target = target[count:]
            written += count
        if target:
            count = self.source.readinto(target)
            if count:
                written += count
        return written


if __name__ == "__main__":
    main()
