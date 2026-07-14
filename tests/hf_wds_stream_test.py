from __future__ import annotations

import io
import tarfile
from email.message import Message

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


def test_download_resumes_a_truncated_partial_with_http_range(tmp_path, monkeypatch) -> None:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        payload = b"image-bytes" * 100
        info = tarfile.TarInfo("000001.jpg")
        info.size = len(payload)
        stream.addfile(info, io.BytesIO(payload))
    complete = archive.getvalue()
    cutoff = len(complete) // 2
    requests = []

    class Response(io.BytesIO):
        def __init__(self, content: bytes, status: int, headers: dict[str, str]):
            super().__init__(content)
            self.status = status
            self.headers = Message()
            for key, value in headers.items():
                self.headers[key] = value

        def getcode(self) -> int:
            return self.status

    def urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return Response(
                complete[:cutoff], 200, {"Content-Length": str(len(complete))}
            )
        assert request.get_header("Range") == f"bytes={cutoff}-"
        return Response(
            complete[cutoff:],
            206,
            {
                "Content-Length": str(len(complete) - cutoff),
                "Content-Range": f"bytes {cutoff}-{len(complete) - 1}/{len(complete)}",
            },
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    partial = tmp_path / "shard.tar.partial"
    _download_validated("https://example.test/shard.tar", "token", partial)
    assert partial.read_bytes() == complete
    assert len(requests) == 2
