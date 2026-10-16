import hashlib
import urllib.request

import pytest

from offline_voice.download import (
    LLM_SHA256,
    PIPER_JSON_SHA256,
    PIPER_ONNX_SHA256,
    download_file,
    main,
    plan_text,
)
from offline_voice.errors import PipelineError


def test_plan_lists_known_checksums_and_does_not_download(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid(*args, **kwargs):
        raise AssertionError("plan mode must not open the network")

    monkeypatch.setattr(urllib.request, "urlopen", forbid)
    assert main([]) == 0
    output = capsys.readouterr().out
    assert PIPER_ONNX_SHA256 in output
    assert PIPER_JSON_SHA256 in output
    assert LLM_SHA256 in output
    assert "No files were downloaded" in output
    assert PIPER_ONNX_SHA256 in plan_text()


def test_download_file_checks_sha256(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"not-a-real-model"
    digest = hashlib.sha256(payload).hexdigest()

    class _Body:
        def __init__(self) -> None:
            self._offset = 0

        def read(self, count: int) -> bytes:
            chunk = payload[self._offset : self._offset + count]
            self._offset += count
            return chunk

        def __enter__(self) -> "_Body":
            return self

        def __exit__(self, *args) -> bool:
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _Body())
    destination = tmp_path / "voice.onnx"
    download_file("https://example.invalid/voice.onnx", destination, digest)
    assert destination.read_bytes() == payload

    with pytest.raises(PipelineError, match="checksum mismatch"):
        download_file("https://example.invalid/voice.onnx", tmp_path / "other.onnx", "0" * 64)
    assert not (tmp_path / "other.onnx").exists()
    assert not (tmp_path / "other.onnx.part").exists()
