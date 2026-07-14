from __future__ import annotations

import io
import tarfile
import urllib.error
from email.message import Message

import pytest

from tools.hf_wds_stream import _download_slot, _stream_download, validate_tar


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


class Response(io.BytesIO):
    def __init__(self, content: bytes, status: int, headers: dict[str, str]):
        super().__init__(content)
        self.status = status
        self.headers = Message()
        for key, value in headers.items():
            self.headers[key] = value

    def getcode(self) -> int:
        return self.status


def test_download_forwards_bytes_while_writing_partial(tmp_path, monkeypatch) -> None:
    complete = _archive_bytes()

    def urlopen(request, timeout):
        return Response(complete, 200, {"Content-Length": str(len(complete))})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    partial = tmp_path / "shard.tar.partial"
    output = io.BytesIO()
    _stream_download("https://example.test/shard.tar", "token", partial, output=output)
    assert output.getvalue() == complete
    assert partial.read_bytes() == complete


def test_download_emits_cached_prefix_then_resumes_range(tmp_path, monkeypatch) -> None:
    complete = _archive_bytes()
    cutoff = len(complete) // 2
    partial = tmp_path / "shard.tar.partial"
    partial.write_bytes(complete[:cutoff])
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
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
    output = io.BytesIO()
    _stream_download("https://example.test/shard.tar", "token", partial, output=output)
    assert len(requests) == 1
    assert output.getvalue() == complete
    assert partial.read_bytes() == complete


def test_network_retry_continues_one_tar_stream_without_duplicate_bytes(
    tmp_path, monkeypatch
) -> None:
    complete = _archive_bytes()
    cutoff = len(complete) // 2
    requests = []

    class FlakyResponse(Response):
        def __init__(self):
            super().__init__(complete, 200, {"Content-Length": str(len(complete))})
            self.reads = 0

        def read(self, size=-1):
            self.reads += 1
            if self.reads == 1:
                return super().read(cutoff)
            raise urllib.error.URLError("connection reset")

    def urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return FlakyResponse()
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
    output = io.BytesIO()
    _stream_download("https://example.test/shard.tar", "token", partial, output=output)
    assert len(requests) == 2
    assert output.getvalue() == complete
    assert partial.read_bytes() == complete


def test_download_honors_finite_attempt_limit(tmp_path, monkeypatch) -> None:
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        raise urllib.error.URLError("temporary outage")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError, match="after 2 attempts"):
        _stream_download(
            "https://example.test/shard.tar",
            "token",
            tmp_path / "shard.tar.partial",
            output=io.BytesIO(),
            attempts=2,
        )
    assert len(calls) == 2


def test_download_does_not_retry_authentication_error(tmp_path, monkeypatch) -> None:
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(request.full_url, 401, "unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(OSError, match="non-retryable HTTP 401"):
        _stream_download(
            "https://example.test/shard.tar",
            "token",
            tmp_path / "shard.tar.partial",
            output=io.BytesIO(),
        )
    assert len(calls) == 1


def test_download_retries_transient_forbidden_response(tmp_path, monkeypatch) -> None:
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        raise urllib.error.HTTPError(
            request.full_url, 403, "temporary CDN rejection", {}, None
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError, match="after 2 attempts"):
        _stream_download(
            "https://example.test/shard.tar",
            "token",
            tmp_path / "shard.tar.partial",
            output=io.BytesIO(),
            attempts=2,
        )
    assert len(calls) == 2
