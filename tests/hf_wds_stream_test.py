from __future__ import annotations

import io
import sys
import tarfile
from types import SimpleNamespace

import pytest

from tools.hf_wds_stream import _download_slot, _download_validated, validate_tar


def test_download_slot_rejects_nonpositive_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        with _download_slot(tmp_path, 0):
            pass


def test_download_slot_creates_reusable_lock(tmp_path) -> None:
    with _download_slot(tmp_path, 1):
        assert (tmp_path / ".download-slots" / "slot-0.lock").is_file()
    with _download_slot(tmp_path, 1):
        pass


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


def _archive_bytes() -> bytes:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        payload = b"image-bytes" * 100
        info = tarfile.TarInfo("000001.jpg")
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
    return archive.getvalue()


def test_download_uses_official_hub_local_dir(tmp_path, monkeypatch) -> None:
    requests = []

    def download(**kwargs):
        requests.append(kwargs)
        destination = kwargs["local_dir"] / kwargs["filename"]
        destination.write_bytes(_archive_bytes())
        return str(destination)

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    result = _download_validated("owner/dataset", "shard.tar", "token", tmp_path)
    assert result == tmp_path / "shard.tar"
    assert requests == [{
        "repo_id": "owner/dataset",
        "filename": "shard.tar",
        "repo_type": "dataset",
        "token": "token",
        "local_dir": tmp_path,
    }]


def test_download_honors_finite_attempt_limit(tmp_path, monkeypatch) -> None:
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("temporary outage")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError, match="after 2 attempts"):
        _download_validated(
            "owner/dataset",
            "shard.tar",
            "token",
            tmp_path,
            attempts=2,
        )
    assert len(calls) == 2


def test_download_does_not_retry_authentication_error(tmp_path, monkeypatch) -> None:
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        error = RuntimeError("unauthorized")
        error.response = SimpleNamespace(status_code=401)
        raise error

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    with pytest.raises(OSError, match="non-retryable HTTP 401"):
        _download_validated(
            "owner/dataset",
            "shard.tar",
            "token",
            tmp_path,
        )
    assert len(calls) == 1


def test_download_retries_transient_forbidden_response(tmp_path, monkeypatch) -> None:
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        error = RuntimeError("temporary CDN rejection")
        error.response = SimpleNamespace(status_code=403)
        raise error

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download)
    )
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError, match="after 2 attempts"):
        _download_validated(
            "owner/dataset",
            "shard.tar",
            "token",
            tmp_path,
            attempts=2,
        )
    assert len(calls) == 2
