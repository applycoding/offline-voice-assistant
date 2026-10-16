"""Mono 16-bit little-endian PCM helpers.

The pipeline keeps audio in this form so the core path does not depend on
NumPy. Optional backends convert at the boundary when they need a float array.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Audio:
    pcm16: bytes
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if len(self.pcm16) % 2 != 0:
            raise ValueError("pcm16 length must be even")

    @property
    def n_samples(self) -> int:
        return len(self.pcm16) // 2

    @property
    def duration_s(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        return self.n_samples / self.sample_rate


def pcm_to_samples(pcm: bytes) -> list[int]:
    if len(pcm) % 2 != 0:
        raise ValueError("pcm16 length must be even")
    return [int.from_bytes(pcm[index : index + 2], "little", signed=True) for index in range(0, len(pcm), 2)]


def samples_to_pcm(samples: list[int]) -> bytes:
    return b"".join(_clamp_int16(sample).to_bytes(2, "little", signed=True) for sample in samples)


def _clamp_int16(value: int) -> int:
    return max(-32768, min(32767, int(value)))


def read_wav(path: Path) -> Audio:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        raise ValueError(f"only 16-bit WAV is supported, got sample width {width}")
    if channels == 1:
        pcm = frames
    elif channels == 2:
        pcm = _downmix_stereo(frames)
    else:
        raise ValueError(f"only mono or stereo WAV is supported, got {channels} channels")
    return Audio(pcm16=pcm, sample_rate=rate)


def write_wav(path: Path, audio: Audio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(audio.pcm16)


def _downmix_stereo(frames: bytes) -> bytes:
    samples = pcm_to_samples(frames)
    mono: list[int] = []
    for index in range(0, len(samples) - 1, 2):
        mono.append((samples[index] + samples[index + 1]) // 2)
    return samples_to_pcm(mono)


def resample_linear(audio: Audio, target_rate: int) -> Audio:
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    if audio.sample_rate == target_rate:
        return audio
    source = pcm_to_samples(audio.pcm16)
    if not source:
        return Audio(pcm16=b"", sample_rate=target_rate)
    n_dst = max(1, int(round(len(source) * target_rate / audio.sample_rate)))
    if len(source) == 1:
        return Audio(pcm16=audio.pcm16, sample_rate=target_rate)
    span = len(source) - 1
    output: list[int] = []
    for index in range(n_dst):
        position = (index * span) / (n_dst - 1) if n_dst > 1 else 0.0
        left = int(math.floor(position))
        right = min(left + 1, span)
        frac = position - left
        value = source[left] * (1.0 - frac) + source[right] * frac
        output.append(int(round(value)))
    return Audio(pcm16=samples_to_pcm(output), sample_rate=target_rate)


def frame_byte_count(sample_rate: int, frame_ms: int) -> int:
    samples = int(sample_rate * frame_ms / 1000)
    if samples <= 0:
        raise ValueError("frame is empty; check sample_rate and frame_ms")
    return samples * 2


def iter_frames(audio: Audio, frame_ms: int) -> list[bytes]:
    size = frame_byte_count(audio.sample_rate, frame_ms)
    return [audio.pcm16[index : index + size] for index in range(0, len(audio.pcm16) - size + 1, size)]


def rms(frame: bytes) -> float:
    samples = pcm_to_samples(frame)
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        total += float(sample) * float(sample)
    return math.sqrt(total / len(samples)) / 32768.0


def tone(duration_s: float, sample_rate: int, frequency: float, amplitude: float = 0.5) -> Audio:
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    count = int(duration_s * sample_rate)
    scale = max(0.0, min(1.0, amplitude))
    samples: list[int] = []
    for index in range(count):
        value = math.sin(2.0 * math.pi * frequency * index / sample_rate) * scale * 32767.0
        samples.append(int(round(value)))
    return Audio(pcm16=samples_to_pcm(samples), sample_rate=sample_rate)


def silence(duration_s: float, sample_rate: int) -> Audio:
    if duration_s < 0:
        raise ValueError("duration_s must be non-negative")
    count = int(duration_s * sample_rate)
    return Audio(pcm16=b"\x00\x00" * count, sample_rate=sample_rate)


def concat(parts: list[Audio]) -> Audio:
    if not parts:
        raise ValueError("no audio to concatenate")
    rate = parts[0].sample_rate
    if any(part.sample_rate != rate for part in parts):
        raise ValueError("sample rates differ")
    return Audio(pcm16=b"".join(part.pcm16 for part in parts), sample_rate=rate)
