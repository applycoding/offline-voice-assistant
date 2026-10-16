"""Speech-to-text backends.

`fake` returns the transcript from config so tests never download Whisper.
`faster-whisper` loads a CTranslate2 model on the first call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from offline_voice.audio import Audio
from offline_voice.errors import MissingModelError
from offline_voice.optional import optional_module


class SpeechToText(Protocol):
    def transcribe(self, audio: Audio) -> str:
        """Return the transcript for one utterance."""


class FakeSTT:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, audio: Audio) -> str:
        if audio.n_samples == 0:
            return ""
        return self.transcript


class FasterWhisperSTT:
    def __init__(
        self,
        model: str,
        model_path: str,
        device: str,
        compute_type: str,
        language: str,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        self.language = language or None
        self._model: object | None = None

    def transcribe(self, audio: Audio) -> str:
        if audio.n_samples == 0:
            return ""
        numpy = optional_module(
            "numpy",
            "faster-whisper needs NumPy, which it normally installs with itself.",
        )
        samples = numpy.frombuffer(audio.pcm16, dtype=numpy.int16).astype(numpy.float32) / 32768.0
        segments, _info = self._load().transcribe(samples, language=self.language)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        module = optional_module(
            "faster_whisper",
            "faster-whisper is not installed. Install the stt extra: "
            "pip install 'offline-voice-assistant[stt]'",
        )
        target = self.model
        if self.model_path:
            path = Path(self.model_path)
            if not path.exists():
                raise MissingModelError(
                    f"faster-whisper model path not found: {path}. "
                    "Leave stt.model_path empty to let faster-whisper download a named size, "
                    "or point it at a local CTranslate2 directory."
                )
            target = str(path)
        self._model = module.WhisperModel(target, device=self.device, compute_type=self.compute_type)
        return self._model
