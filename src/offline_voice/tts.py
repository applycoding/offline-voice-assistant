"""Text-to-speech backends.

`fake` synthesizes a short tone so the demo can write a WAV file.
`piper` prefers the Piper CLI and falls back to the `piper` Python package.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Protocol

from offline_voice.audio import Audio, read_wav, silence, tone
from offline_voice.errors import MissingModelError, PipelineError
from offline_voice.optional import optional_module


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> Audio:
        """Render one utterance to mono PCM."""


class FakeTTS:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> Audio:
        words = max(1, len(text.split()))
        duration = min(1.2, 0.08 + 0.03 * words)
        return tone(duration, self.sample_rate, 440.0, amplitude=0.15)


class PiperTTS:
    def __init__(self, model_path: str, piper_bin: str = "piper", length_scale: float = 1.0) -> None:
        self.model_path = Path(model_path)
        self.piper_bin = piper_bin
        self.length_scale = length_scale
        self._voice: object | None = None

    def synthesize(self, text: str) -> Audio:
        cleaned = text.strip()
        if not cleaned:
            return silence(0.0, 22050)
        if not self.model_path.is_file():
            raise MissingModelError(
                f"Piper voice not found at {self.model_path}. "
                "Run: python scripts/download_models.py --piper"
            )
        if shutil.which(self.piper_bin):
            return self._via_cli(cleaned)
        optional_module(
            "piper",
            "Piper is not available. Install the `piper` binary on PATH "
            "or pip install 'offline-voice-assistant[tts]'.",
        )
        return self._via_python(cleaned)

    def _via_cli(self, text: str) -> Audio:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "speech.wav"
            command = [
                self.piper_bin,
                "--model",
                str(self.model_path),
                "--output_file",
                str(output),
                "--length-scale",
                str(self.length_scale),
            ]
            completed = subprocess.run(command, input=text.encode("utf-8"), capture_output=True, check=False)
            if completed.returncode != 0 or not output.is_file():
                detail = completed.stderr.decode("utf-8", errors="replace")[:400]
                raise PipelineError(f"piper exited {completed.returncode}: {detail}")
            return read_wav(output)

    def _via_python(self, text: str) -> Audio:
        module = optional_module("piper", "piper-tts is not installed.")
        if self._voice is None:
            self._voice = module.PiperVoice.load(str(self.model_path))
        voice = self._voice
        try:
            produced = voice.synthesize(text)
        except TypeError:
            return _legacy_synthesize(voice, text)
        if hasattr(produced, "audio_int16_bytes"):
            return Audio(pcm16=produced.audio_int16_bytes, sample_rate=int(produced.sample_rate))
        pcm = bytearray()
        rate = 22050
        for chunk in produced:
            pcm.extend(chunk.audio_int16_bytes)
            rate = int(chunk.sample_rate)
        if not pcm:
            raise PipelineError("Piper returned no audio")
        return Audio(pcm16=bytes(pcm), sample_rate=rate)


def _legacy_synthesize(voice: object, text: str) -> Audio:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        voice.synthesize(text, handle)  # type: ignore[attr-defined]
    buffer.seek(0)
    with wave.open(buffer, "rb") as handle:
        return Audio(pcm16=handle.readframes(handle.getnframes()), sample_rate=handle.getframerate())
