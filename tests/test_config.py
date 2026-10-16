from pathlib import Path

import pytest

from offline_voice.config import load_config
from offline_voice.factory import build_pipeline


def test_demo_and_device_configs_load() -> None:
    demo = load_config("config/demo.yaml")
    assert demo.mode == "turn"
    assert demo.stt.backend == "fake"
    assert demo.llm.backend == "fake"
    assert demo.tts.backend == "fake"

    default = load_config("config/default.yaml")
    assert default.stt.backend == "faster-whisper"
    assert default.llm.backend == "llama-cpp"
    assert default.tts.backend == "piper"

    pi = load_config("config/pi.yaml")
    assert pi.vad.backend == "webrtcvad"
    assert pi.stt.device == "cpu"

    jetson = load_config("config/jetson.yaml")
    assert jetson.stt.device == "cuda"
    assert jetson.llm.backend == "ollama"


def test_real_configs_construct_without_loading_weights() -> None:
    for name in ("config/default.yaml", "config/pi.yaml", "config/jetson.yaml"):
        pipeline = build_pipeline(load_config(name))
        assert pipeline.config.stt.backend != "fake"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("mode: turn\nextra: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config fields"):
        load_config(path)


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("stt:\n  backend: cloud\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stt.backend"):
        load_config(path)


def test_missing_file() -> None:
    with pytest.raises(ValueError, match="config not found"):
        load_config("config/missing.yaml")
