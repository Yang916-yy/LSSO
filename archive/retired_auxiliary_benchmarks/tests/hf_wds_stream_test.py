from __future__ import annotations

import io
from unittest.mock import patch

import pytest

from archive.retired_auxiliary_benchmarks.tools.hf_wds_stream import stream_file


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, expected: int | None = None) -> None:
        super().__init__(body)
        self.headers = {
            "Content-Length": str(len(body) if expected is None else expected)
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Output:
    def __init__(self) -> None:
        self.buffer = io.BytesIO()


def test_download_streams_and_reuses_atomic_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    first_output = Output()
    with (
        patch(
            "archive.retired_auxiliary_benchmarks.tools.hf_wds_stream.urllib.request.urlopen",
            return_value=FakeResponse(b"tar bytes"),
        ) as request,
        patch("archive.retired_auxiliary_benchmarks.tools.hf_wds_stream.sys.stdout", first_output),
    ):
        stream_file("owner/repo", "shard.tar", tmp_path)
    assert first_output.buffer.getvalue() == b"tar bytes"
    assert (tmp_path / "shard.tar").read_bytes() == b"tar bytes"
    assert not (tmp_path / "shard.tar.partial").exists()
    assert request.call_count == 1

    cached_output = Output()
    with (
        patch(
            "archive.retired_auxiliary_benchmarks.tools.hf_wds_stream.urllib.request.urlopen",
            side_effect=AssertionError("cache should avoid the network"),
        ),
        patch("archive.retired_auxiliary_benchmarks.tools.hf_wds_stream.sys.stdout", cached_output),
    ):
        stream_file("owner/repo", "shard.tar", tmp_path)
    assert cached_output.buffer.getvalue() == b"tar bytes"


def test_truncated_response_fails_fast_and_removes_partial(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    with (
        patch(
            "archive.retired_auxiliary_benchmarks.tools.hf_wds_stream.urllib.request.urlopen",
            return_value=FakeResponse(b"partial", expected=100),
        ),
        patch("archive.retired_auxiliary_benchmarks.tools.hf_wds_stream.sys.stdout", Output()),
    ):
        with pytest.raises(OSError, match="truncated shard"):
            stream_file("owner/repo", "shard.tar", tmp_path)
    assert not (tmp_path / "shard.tar").exists()
    assert not (tmp_path / "shard.tar.partial").exists()


def test_missing_token_fails_before_request(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        stream_file("owner/repo", "shard.tar", tmp_path)
