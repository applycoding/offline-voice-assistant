"""Voice activity detectors.

`energy` is a pure-Python RMS endpoint detector and the default, so the demo
runs with no extra packages. `webrtcvad` and `silero` are optional adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from offline_voice.audio import Audio, frame_byte_count, iter_frames, resample_linear, rms
from offline_voice.errors import MissingModelError
from offline_voice.optional import optional_module

_SILERO_HINT = (
    "Silero VAD needs PyTorch. Install the silero extra: pip install 'offline-voice-assistant[silero]'"
)


class VoiceActivity(Protocol):
    def trim(self, audio: Audio) -> Audio | None:
        """Return the speech portion of an utterance, or None when no speech is present."""
        ...


@dataclass(frozen=True)
class _Span:
    start_frame: int
    end_frame: int


class EnergyVAD:
    def __init__(
        self,
        threshold: float,
        silence_ms: int,
        min_speech_ms: int,
        pad_ms: int,
        frame_ms: int,
    ) -> None:
        self.threshold = threshold
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.pad_ms = pad_ms
        self.frame_ms = frame_ms

    def trim(self, audio: Audio) -> Audio | None:
        frames = iter_frames(audio, self.frame_ms)
        if not frames:
            return None
        voiced = [rms(frame) >= self.threshold for frame in frames]
        return _audio_from_mask(
            audio,
            voiced,
            self.frame_ms,
            self.min_speech_ms,
            self.silence_ms,
            self.pad_ms,
        )


class WebRtcVAD:
    def __init__(
        self,
        aggressiveness: int,
        silence_ms: int,
        min_speech_ms: int,
        pad_ms: int,
        frame_ms: int,
    ) -> None:
        if frame_ms not in {10, 20, 30}:
            raise ValueError("webrtcvad requires 10, 20, or 30 ms frames")
        self.aggressiveness = aggressiveness
        self.silence_ms = silence_ms
        self.min_speech_ms = min_speech_ms
        self.pad_ms = pad_ms
        self.frame_ms = frame_ms
        self._vad: object | None = None

    def trim(self, audio: Audio) -> Audio | None:
        if audio.sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("webrtcvad supports 8000, 16000, 32000, or 48000 Hz")
        frames = iter_frames(audio, self.frame_ms)
        if not frames:
            return None
        detector = self._load()
        voiced = [bool(detector.is_speech(frame, audio.sample_rate)) for frame in frames]
        return _audio_from_mask(
            audio,
            voiced,
            self.frame_ms,
            self.min_speech_ms,
            self.silence_ms,
            self.pad_ms,
        )

    def _load(self) -> object:
        if self._vad is None:
            module = optional_module(
                "webrtcvad",
                "webrtcvad is not installed. Install the vad extra: "
                "pip install 'offline-voice-assistant[vad]'",
            )
            self._vad = module.Vad(self.aggressiveness)
        return self._vad


class SileroVAD:
    def __init__(self, threshold: float, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._model: object | None = None
        self._timestamps: object | None = None

    def trim(self, audio: Audio) -> Audio | None:
        if audio.sample_rate == self.sample_rate:
            prepared = audio
        else:
            prepared = resample_linear(audio, self.sample_rate)
        if prepared.n_samples == 0:
            return None
        model, get_speech_timestamps = self._load()
        torch = optional_module("torch", _SILERO_HINT)
        numpy = optional_module("numpy", "Silero VAD needs NumPy. Install the silero extra.")
        floats = numpy.frombuffer(prepared.pcm16, dtype=numpy.int16).astype(numpy.float32) / 32768.0
        tensor = torch.from_numpy(floats)
        stamps = get_speech_timestamps(
            tensor,
            model,
            sampling_rate=self.sample_rate,
            threshold=self.threshold,
        )
        if not stamps:
            return None
        start = int(stamps[0]["start"])
        end = int(stamps[-1]["end"])
        start = max(0, start)
        end = min(prepared.n_samples, max(start, end))
        pcm = prepared.pcm16[start * 2 : end * 2]
        if not pcm:
            return None
        return Audio(pcm16=pcm, sample_rate=prepared.sample_rate)

    def _load(self) -> tuple[object, object]:
        if self._model is not None and self._timestamps is not None:
            return self._model, self._timestamps
        torch = optional_module("torch", _SILERO_HINT)
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
        except Exception as exc:
            raise MissingModelError(
                "Silero VAD weights could not be loaded. The first load uses torch.hub "
                "and needs a network path to github.com, or a cached hub checkout."
            ) from exc
        get_speech_timestamps = utils[0]
        self._model = model
        self._timestamps = get_speech_timestamps
        return model, get_speech_timestamps


def _audio_from_mask(
    audio: Audio,
    voiced: list[bool],
    frame_ms: int,
    min_speech_ms: int,
    silence_ms: int,
    pad_ms: int,
) -> Audio | None:
    span = _endpoint(voiced, frame_ms, min_speech_ms, silence_ms)
    if span is None:
        return None
    frame_bytes = frame_byte_count(audio.sample_rate, frame_ms)
    pad_frames = int(pad_ms / frame_ms) if frame_ms else 0
    start = max(0, span.start_frame - pad_frames) * frame_bytes
    end = min(len(audio.pcm16), (span.end_frame + pad_frames) * frame_bytes)
    if end <= start:
        return None
    pcm = audio.pcm16[start:end]
    if len(pcm) < 2:
        return None
    return Audio(pcm16=pcm, sample_rate=audio.sample_rate)


def _endpoint(voiced: list[bool], frame_ms: int, min_speech_ms: int, silence_ms: int) -> _Span | None:
    min_frames = max(1, int(min_speech_ms / frame_ms)) if frame_ms else 1
    silence_frames = max(1, int(silence_ms / frame_ms)) if frame_ms else 1
    start: int | None = None
    last_voice: int | None = None
    run = 0
    silent_run = 0
    for index, is_speech in enumerate(voiced):
        if is_speech:
            run += 1
            silent_run = 0
            last_voice = index
            if start is None and run >= min_frames:
                start = index - run + 1
            continue
        run = 0
        if start is None:
            continue
        silent_run += 1
        if silent_run >= silence_frames and last_voice is not None:
            return _Span(start_frame=start, end_frame=last_voice + 1)
    if start is None or last_voice is None:
        return None
    return _Span(start_frame=start, end_frame=last_voice + 1)
