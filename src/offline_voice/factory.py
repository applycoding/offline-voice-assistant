"""Build a pipeline from YAML config."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from offline_voice.config import AppConfig
from offline_voice.latency import LatencyLog
from offline_voice.llm import FakeLLM, LanguageModel, LlamaCppLLM, OllamaLLM
from offline_voice.pipeline import VoicePipeline
from offline_voice.stt import FakeSTT, FasterWhisperSTT, SpeechToText
from offline_voice.tools import ToolRegistry
from offline_voice.tts import FakeTTS, PiperTTS, TextToSpeech
from offline_voice.vad import EnergyVAD, SileroVAD, WebRtcVAD


def build_pipeline(config: AppConfig, clock: Callable[[], datetime] | None = None) -> VoicePipeline:
    return VoicePipeline(
        vad=build_vad(config),
        stt=build_stt(config),
        llm=build_llm(config),
        tts=build_tts(config),
        tools=ToolRegistry(
            enabled=config.tools.enabled,
            clock=clock,
            weather_location=config.tools.weather_location,
        ),
        latency=LatencyLog(
            backend_for={
                "vad": config.vad.backend,
                "stt": config.stt.backend,
                "llm": config.llm.backend,
                "tts": config.tts.backend,
                "tools": "local",
            }
        ),
        config=config,
    )


def build_vad(config: AppConfig) -> EnergyVAD | WebRtcVAD | SileroVAD:
    vad = config.vad
    frame_ms = config.audio.frame_ms
    if vad.backend == "energy":
        return EnergyVAD(
            threshold=vad.threshold,
            silence_ms=vad.silence_ms,
            min_speech_ms=vad.min_speech_ms,
            pad_ms=vad.pad_ms,
            frame_ms=frame_ms,
        )
    if vad.backend == "webrtcvad":
        return WebRtcVAD(
            aggressiveness=vad.aggressiveness,
            silence_ms=vad.silence_ms,
            min_speech_ms=vad.min_speech_ms,
            pad_ms=vad.pad_ms,
            frame_ms=frame_ms,
        )
    if vad.backend == "silero":
        return SileroVAD(threshold=vad.silero_threshold, sample_rate=config.audio.sample_rate)
    raise ValueError(f"unknown vad backend '{vad.backend}'")


def build_stt(config: AppConfig) -> SpeechToText:
    stt = config.stt
    if stt.backend == "fake":
        return FakeSTT(stt.fake_transcript)
    if stt.backend == "faster-whisper":
        return FasterWhisperSTT(
            model=stt.model,
            model_path=stt.model_path,
            device=stt.device,
            compute_type=stt.compute_type,
            language=stt.language,
        )
    raise ValueError(f"unknown stt backend '{stt.backend}'")


def build_llm(config: AppConfig) -> LanguageModel:
    llm = config.llm
    if llm.backend == "fake":
        return FakeLLM()
    if llm.backend == "llama-cpp":
        return LlamaCppLLM(
            model_path=llm.model_path,
            n_ctx=llm.n_ctx,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
        )
    if llm.backend == "ollama":
        return OllamaLLM(
            base_url=llm.ollama_url,
            model=llm.ollama_model,
            temperature=llm.temperature,
            max_tokens=llm.max_tokens,
            timeout_s=llm.timeout_s,
        )
    raise ValueError(f"unknown llm backend '{llm.backend}'")


def build_tts(config: AppConfig) -> TextToSpeech:
    tts = config.tts
    if tts.backend == "fake":
        return FakeTTS(sample_rate=config.audio.sample_rate)
    if tts.backend == "piper":
        return PiperTTS(model_path=tts.model_path, piper_bin=tts.piper_bin, length_scale=tts.length_scale)
    raise ValueError(f"unknown tts backend '{tts.backend}'")
