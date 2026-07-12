from __future__ import annotations

import io
import tarfile

import pytest

from tools.hf_wds_stream import validate_tar


def test_validate_tar_accepts_complete_archive(tmp_path) -> None:
    path = tmp_path / "shard.tar"
    payload = b"image-bytes"
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo("000001.jpg")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    validate_tar(path, expected_bytes=path.stat().st_size)


def test_validate_tar_rejects_truncated_archive(tmp_path) -> None:
    path = tmp_path / "shard.tar"
    path.write_bytes(b"not-a-complete-tar")
    with pytest.raises(OSError):
        validate_tar(path)


def test_validate_tar_checks_content_length(tmp_path) -> None:
    path = tmp_path / "shard.tar"
    path.write_bytes(b"short")
    with pytest.raises(OSError, match="expected 100 bytes"):
        validate_tar(path, expected_bytes=100)
