from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from tools.hf_wds_pipe import stream_shard


class FakeProcess:
    def __init__(self, body: bytes, returncode: int, stderr: bytes = b"") -> None:
        self.stdin = io.BytesIO()
        self.stdout = io.BytesIO(body)
        self.stderr = io.BytesIO(stderr)
        self.returncode = returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def terminate(self) -> None:
        return None


class Output:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


def test_retries_zero_byte_403_without_exposing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    processes = [
        FakeProcess(b"", 22, b"curl: (22) error: 403"),
        FakeProcess(b"tar bytes", 0),
    ]
    output = Output()
    with (
        patch("tools.hf_wds_pipe.subprocess.Popen", side_effect=processes) as popen,
        patch("tools.hf_wds_pipe.time.sleep"),
        patch("tools.hf_wds_pipe.sys.stdout", output),
    ):
        stream_shard("owner/repo", "shard.tar", attempts=2)
    assert output.buffer.getvalue() == b"tar bytes"
    assert popen.call_count == 2
    assert "hf_secret" not in " ".join(popen.call_args.args[0])


def test_refuses_to_retry_after_partial_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    output = Output()
    with (
        patch(
            "tools.hf_wds_pipe.subprocess.Popen",
            return_value=FakeProcess(b"partial", 18, b"transfer closed"),
        ),
        patch("tools.hf_wds_pipe.sys.stdout", output),
    ):
        with pytest.raises(OSError, match="unsafe in-stream retry"):
            stream_shard("owner/repo", "shard.tar", attempts=3)
    assert output.buffer.getvalue() == b"partial"


def test_missing_token_fails_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        stream_shard("owner/repo", "shard.tar")
