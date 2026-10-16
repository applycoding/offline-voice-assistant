import wave
from pathlib import Path

import pytest

from offline_voice.audio import concat, read_wav, resample_linear, silence, tone, write_wav


def test_resample_keeps_duration() -> None:
    source = tone(0.5, 8000, 220.0, amplitude=0.4)
    converted = resample_linear(source, 16000)
    assert converted.sample_rate == 16000
    assert converted.duration_s == pytest.approx(0.5, abs=0.02)


def test_resample_empty() -> None:
    converted = resample_linear(silence(0.0, 8000), 16000)
    assert converted.n_samples == 0
    assert converted.sample_rate == 16000


def test_stereo_wav_downmix(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    left = tone(0.1, 16000, 440.0, amplitude=0.4)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        mono = left.pcm16
        stereo = bytearray()
        for index in range(0, len(mono), 2):
            stereo.extend(mono[index : index + 2])
            stereo.extend(mono[index : index + 2])
        handle.writeframes(bytes(stereo))
    loaded = read_wav(path)
    assert loaded.sample_rate == 16000
    assert loaded.n_samples == left.n_samples


def test_rejects_8bit_wav(tmp_path: Path) -> None:
    path = tmp_path / "tiny.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(16000)
        handle.writeframes(b"\x80" * 160)
    with pytest.raises(ValueError, match="16-bit"):
        read_wav(path)


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    original = concat([silence(0.05, 16000), tone(0.05, 16000, 880.0)])
    path = tmp_path / "clip.wav"
    write_wav(path, original)
    loaded = read_wav(path)
    assert loaded.pcm16 == original.pcm16
    assert loaded.sample_rate == 16000
