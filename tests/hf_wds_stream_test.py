from __future__ import annotations

import io
import subprocess
import sys
import tarfile

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


class FakeProcess:
    def __init__(self, content: bytes, returncode: int = 0, error: str = ""):
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(content)
        self.stderr = io.BytesIO(error.encode())
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


def test_download_forwards_bytes_while_writing_partial(tmp_path, monkeypatch) -> None:
    complete = _archive_bytes()

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: FakeProcess(complete))
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
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        return FakeProcess(complete[cutoff:])

    monkeypatch.setattr("subprocess.Popen", popen)
    output = io.BytesIO()
    _stream_download("https://example.test/shard.tar", "token", partial, output=output)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--continue-at") + 1] == str(cutoff)
    assert output.getvalue() == complete
    assert partial.read_bytes() == complete


def test_network_retry_continues_one_tar_stream_without_duplicate_bytes(
    tmp_path, monkeypatch
) -> None:
    complete = _archive_bytes()
    cutoff = len(complete) // 2
    commands = []
    processes = [
        FakeProcess(complete[:cutoff], 56, "connection reset"),
        FakeProcess(complete[cutoff:]),
    ]

    def popen(command, **kwargs):
        commands.append(command)
        return processes.pop(0)

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    partial = tmp_path / "shard.tar.partial"
    output = io.BytesIO()
    _stream_download("https://example.test/shard.tar", "token", partial, output=output)
    assert len(commands) == 2
    assert commands[1][commands[1].index("--continue-at") + 1] == str(cutoff)
    assert output.getvalue() == complete
    assert partial.read_bytes() == complete


def test_download_honors_finite_attempt_limit(tmp_path, monkeypatch) -> None:
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        return FakeProcess(b"", 56, "temporary outage")

    monkeypatch.setattr("subprocess.Popen", popen)
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

    def popen(command, **kwargs):
        calls.append(command)
        return FakeProcess(b"", 22, "The requested URL returned error: 401")

    monkeypatch.setattr("subprocess.Popen", popen)
    with pytest.raises(OSError, match="non-retryable download error"):
        _stream_download(
            "https://example.test/shard.tar",
            "token",
            tmp_path / "shard.tar.partial",
            output=io.BytesIO(),
        )
    assert len(calls) == 1


def test_download_retries_transient_forbidden_response(tmp_path, monkeypatch) -> None:
    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        return FakeProcess(b"", 22, "The requested URL returned error: 403")

    monkeypatch.setattr("subprocess.Popen", popen)
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


def test_download_restarts_a_connection_that_delivers_no_bytes(tmp_path, monkeypatch) -> None:
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        return real_popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdin=kwargs.get("stdin"),
            stdout=kwargs.get("stdout"),
            stderr=kwargs.get("stderr"),
        )

    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    with pytest.raises(OSError, match="after 1 attempts"):
        _stream_download(
            "https://example.test/shard.tar",
            "token",
            tmp_path / "shard.tar.partial",
            output=io.BytesIO(),
            attempts=1,
            idle_timeout=0.05,
        )
