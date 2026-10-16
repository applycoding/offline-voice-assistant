"""YAML configuration for the voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_SYSTEM_PROMPT = """You are the onboard voice assistant for a humanoid robot.
You run fully offline on this device.

When you need a tool, reply with exactly one block and no other text:
<tool>{"name":"time","arguments":{}}</tool>
<tool>{"name":"weather","arguments":{"location":"the lab"}}</tool>
<tool>{"name":"robot","arguments":{"command":"wave"}}</tool>

Tools:
- time: local clock on this device. It does not use the network.
- weather: offline stub. It cannot look up a forecast.
  Use it only when the person asks about weather.
- robot: command must be one of wave, nod, stop, look_left, look_right.
  This queues a gesture stub. It does not move hardware by itself.

After a tool result arrives, answer in one or two short spoken sentences.
Do not emit a tool block in that final answer.
Do not claim you used the internet.
"""

MODES = frozenset({"turn", "stream"})
VAD_BACKENDS = frozenset({"energy", "webrtcvad", "silero"})
STT_BACKENDS = frozenset({"fake", "faster-whisper"})
LLM_BACKENDS = frozenset({"fake", "llama-cpp", "ollama"})
TTS_BACKENDS = frozenset({"fake", "piper"})

_ROOT_KEYS = frozenset({"mode", "max_tool_rounds", "audio", "vad", "stt", "llm", "tts", "tools", "latency"})
_AUDIO_KEYS = frozenset({"sample_rate", "frame_ms"})
_VAD_KEYS = frozenset(
    {"backend", "threshold", "aggressiveness", "silence_ms", "min_speech_ms", "pad_ms", "silero_threshold"}
)
_STT_KEYS = frozenset(
    {"backend", "model", "model_path", "language", "device", "compute_type", "fake_transcript"}
)
_LLM_KEYS = frozenset(
    {
        "backend",
        "model_path",
        "n_ctx",
        "temperature",
        "max_tokens",
        "ollama_model",
        "ollama_url",
        "timeout_s",
        "system_prompt",
    }
)
_TTS_KEYS = frozenset({"backend", "model_path", "piper_bin", "length_scale"})
_TOOLS_KEYS = frozenset({"enabled", "weather_location"})
_LATENCY_KEYS = frozenset({"log_path"})


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 30


@dataclass(frozen=True)
class VadConfig:
    backend: str = "energy"
    threshold: float = 0.02
    aggressiveness: int = 2
    silence_ms: int = 600
    min_speech_ms: int = 200
    pad_ms: int = 120
    silero_threshold: float = 0.5


@dataclass(frozen=True)
class SttConfig:
    backend: str = "fake"
    model: str = "tiny"
    model_path: str = ""
    language: str = "en"
    device: str = "cpu"
    compute_type: str = "int8"
    fake_transcript: str = "what time is it"


@dataclass(frozen=True)
class LlmConfig:
    backend: str = "fake"
    model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    n_ctx: int = 2048
    temperature: float = 0.2
    max_tokens: int = 256
    ollama_model: str = "qwen2.5:0.5b"
    ollama_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 120.0
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class TtsConfig:
    backend: str = "fake"
    model_path: str = "models/en_US-lessac-medium.onnx"
    piper_bin: str = "piper"
    length_scale: float = 1.0


@dataclass(frozen=True)
class ToolsConfig:
    enabled: bool = True
    weather_location: str = "the lab"


@dataclass(frozen=True)
class LatencyConfig:
    log_path: str = "logs/latency.jsonl"


@dataclass(frozen=True)
class AppConfig:
    mode: str = "turn"
    max_tool_rounds: int = 3
    audio: AudioConfig = AudioConfig()
    vad: VadConfig = VadConfig()
    stt: SttConfig = SttConfig()
    llm: LlmConfig = LlmConfig()
    tts: TtsConfig = TtsConfig()
    tools: ToolsConfig = ToolsConfig()
    latency: LatencyConfig = LatencyConfig()


def load_config(path: str | Path) -> AppConfig:
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"config not found: {file_path}")
    loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    data = loaded if loaded is not None else {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    _reject_unknown("config", data, _ROOT_KEYS)
    mode = str(data.get("mode", "turn"))
    _require_choice("mode", mode, MODES)
    max_tool_rounds = int(data.get("max_tool_rounds", 3))
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be at least 1")
    audio = _audio(_section(data, "audio"))
    vad = _vad(_section(data, "vad"), audio.frame_ms)
    return AppConfig(
        mode=mode,
        max_tool_rounds=max_tool_rounds,
        audio=audio,
        vad=vad,
        stt=_stt(_section(data, "stt")),
        llm=_llm(_section(data, "llm")),
        tts=_tts(_section(data, "tts")),
        tools=_tools(_section(data, "tools")),
        latency=_latency(_section(data, "latency")),
    )


def _section(data: dict, key: str) -> dict:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _reject_unknown(label: str, data: dict, allowed: frozenset[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        names = ", ".join(unknown)
        raise ValueError(f"unknown {label} fields: {names}")


def _require_choice(label: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"unknown {label} '{value}'. Expected one of: {choices}")


def _audio(data: dict) -> AudioConfig:
    _reject_unknown("audio", data, _AUDIO_KEYS)
    sample_rate = int(data.get("sample_rate", 16000))
    frame_ms = int(data.get("frame_ms", 30))
    if sample_rate <= 0:
        raise ValueError("audio.sample_rate must be positive")
    if frame_ms <= 0:
        raise ValueError("audio.frame_ms must be positive")
    return AudioConfig(sample_rate=sample_rate, frame_ms=frame_ms)


def _vad(data: dict, frame_ms: int) -> VadConfig:
    _reject_unknown("vad", data, _VAD_KEYS)
    backend = str(data.get("backend", "energy"))
    _require_choice("vad.backend", backend, VAD_BACKENDS)
    aggressiveness = int(data.get("aggressiveness", 2))
    if aggressiveness not in {0, 1, 2, 3}:
        raise ValueError("vad.aggressiveness must be 0, 1, 2, or 3")
    if backend == "webrtcvad" and frame_ms not in {10, 20, 30}:
        raise ValueError("webrtcvad requires audio.frame_ms of 10, 20, or 30")
    threshold = float(data.get("threshold", 0.02))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("vad.threshold must be between 0 and 1")
    silence_ms = int(data.get("silence_ms", 600))
    min_speech_ms = int(data.get("min_speech_ms", 200))
    pad_ms = int(data.get("pad_ms", 120))
    if min(silence_ms, min_speech_ms, pad_ms) < 0:
        raise ValueError("vad timing fields must be non-negative")
    return VadConfig(
        backend=backend,
        threshold=threshold,
        aggressiveness=aggressiveness,
        silence_ms=silence_ms,
        min_speech_ms=min_speech_ms,
        pad_ms=pad_ms,
        silero_threshold=float(data.get("silero_threshold", 0.5)),
    )


def _stt(data: dict) -> SttConfig:
    _reject_unknown("stt", data, _STT_KEYS)
    backend = str(data.get("backend", "fake"))
    _require_choice("stt.backend", backend, STT_BACKENDS)
    return SttConfig(
        backend=backend,
        model=str(data.get("model", "tiny")),
        model_path=str(data.get("model_path", "")),
        language=str(data.get("language", "en")),
        device=str(data.get("device", "cpu")),
        compute_type=str(data.get("compute_type", "int8")),
        fake_transcript=str(data.get("fake_transcript", "what time is it")),
    )


def _llm(data: dict) -> LlmConfig:
    _reject_unknown("llm", data, _LLM_KEYS)
    backend = str(data.get("backend", "fake"))
    _require_choice("llm.backend", backend, LLM_BACKENDS)
    timeout_s = float(data.get("timeout_s", 120.0))
    if timeout_s <= 0:
        raise ValueError("llm.timeout_s must be positive")
    return LlmConfig(
        backend=backend,
        model_path=str(data.get("model_path", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf")),
        n_ctx=int(data.get("n_ctx", 2048)),
        temperature=float(data.get("temperature", 0.2)),
        max_tokens=int(data.get("max_tokens", 256)),
        ollama_model=str(data.get("ollama_model", "qwen2.5:0.5b")),
        ollama_url=str(data.get("ollama_url", "http://127.0.0.1:11434")).rstrip("/"),
        timeout_s=timeout_s,
        system_prompt=str(data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
    )


def _tts(data: dict) -> TtsConfig:
    _reject_unknown("tts", data, _TTS_KEYS)
    backend = str(data.get("backend", "fake"))
    _require_choice("tts.backend", backend, TTS_BACKENDS)
    return TtsConfig(
        backend=backend,
        model_path=str(data.get("model_path", "models/en_US-lessac-medium.onnx")),
        piper_bin=str(data.get("piper_bin", "piper")),
        length_scale=float(data.get("length_scale", 1.0)),
    )


def _tools(data: dict) -> ToolsConfig:
    _reject_unknown("tools", data, _TOOLS_KEYS)
    return ToolsConfig(
        enabled=bool(data.get("enabled", True)),
        weather_location=str(data.get("weather_location", "the lab")),
    )


def _latency(data: dict) -> LatencyConfig:
    _reject_unknown("latency", data, _LATENCY_KEYS)
    return LatencyConfig(log_path=str(data.get("log_path", "logs/latency.jsonl")))
