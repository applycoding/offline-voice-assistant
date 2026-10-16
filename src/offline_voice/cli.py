"""Command line for the offline voice demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from offline_voice import __version__
from offline_voice.audio import Audio, concat, read_wav, resample_linear, write_wav
from offline_voice.config import AppConfig, load_config
from offline_voice.errors import PipelineError
from offline_voice.factory import build_pipeline
from offline_voice.optional import optional_module
from offline_voice.pipeline import StreamEvent, TurnResult, VoicePipeline
from offline_voice.tools import ToolResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="offline-voice",
        description=(
            "Demo of a fully offline voice assistant. "
            "The default config uses fake speech models so it runs without weights."
        ),
    )
    parser.add_argument("--version", action="version", version=f"offline-voice {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one turn from text, a WAV file, or the microphone")
    latency = sub.add_parser("latency", help="Run one turn and print timings measured by this process")
    for command in (run, latency):
        command.add_argument("--config", default="config/demo.yaml", help="YAML config path")
        command.add_argument("--wav", help="16-bit WAV input")
        command.add_argument("--mic", action="store_true", help="Record from the default microphone")
        command.add_argument("--seconds", type=float, default=5.0, help="Microphone capture length")
        command.add_argument("--text", help="Skip audio and start from this transcript")
        command.add_argument("--mode", choices=("turn", "stream"), help="Override config mode")
        command.add_argument("--out", help="Write synthesized speech to this WAV path")
        command.add_argument("--latency-log", help="Override the JSONL latency log path")
        command.add_argument("--json", action="store_true", help="Print a JSON object on stdout")

    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (PipelineError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    chosen = sum(bool(item) for item in (args.wav, args.mic, args.text))
    if chosen != 1:
        print("error: choose one of --wav, --mic, or --text", file=sys.stderr)
        return 2
    config = load_config(args.config)
    if args.mode:
        config = _with_mode(config, args.mode)
    pipeline = build_pipeline(config)
    return _run(pipeline, args)


def _with_mode(config: AppConfig, mode: str) -> AppConfig:
    return AppConfig(
        mode=mode,
        max_tool_rounds=config.max_tool_rounds,
        audio=config.audio,
        vad=config.vad,
        stt=config.stt,
        llm=config.llm,
        tts=config.tts,
        tools=config.tools,
        latency=config.latency,
    )


def _run(pipeline: VoicePipeline, args: argparse.Namespace) -> int:
    mode = pipeline.config.mode
    if args.text:
        if mode == "stream":
            events = list(pipeline.run_stream_text(args.text))
            _emit_stream(events, args, pipeline)
            return 0
        result = pipeline.run_turn_text(args.text)
        _emit_turn(result, args, pipeline)
        return 0
    audio = _load_audio(args, pipeline.config.audio.sample_rate)
    if mode == "stream":
        events = list(pipeline.run_stream_audio(audio))
        _emit_stream(events, args, pipeline)
        return 0
    result = pipeline.run_turn_audio(audio)
    _emit_turn(result, args, pipeline)
    return 0


def _load_audio(args: argparse.Namespace, sample_rate: int) -> Audio:
    if args.wav:
        return read_wav(Path(args.wav))
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")
    return record_microphone(args.seconds, sample_rate)


def record_microphone(seconds: float, sample_rate: int) -> Audio:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    sounddevice = optional_module(
        "sounddevice",
        "Microphone capture needs the mic extra: pip install 'offline-voice-assistant[mic]'",
    )
    frames = int(seconds * sample_rate)
    recorded = sounddevice.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sounddevice.wait()
    pcm = recorded.tobytes() if hasattr(recorded, "tobytes") else bytes(recorded)
    return Audio(pcm16=pcm, sample_rate=sample_rate)


def _emit_turn(result: TurnResult, args: argparse.Namespace, pipeline: VoicePipeline) -> None:
    audio_path = _write_audio(args.out, result.speech)
    _write_log(pipeline, args.latency_log)
    payload = {
        "transcript": result.transcript,
        "reply": result.reply,
        "no_speech": result.no_speech,
        "tool_results": [_tool_dict(item) for item in result.tool_results],
        "audio": audio_path,
        "latency": pipeline.latency.as_dicts(),
        "note": "durations were measured by this process for the configured backends",
    }
    if args.json:
        print(json.dumps(payload))
        return
    _print_human(result.transcript, result.reply, result.tool_results, result.no_speech, audio_path)
    print()
    print(pipeline.latency.markdown_table())


def _emit_stream(events: list[StreamEvent], args: argparse.Namespace, pipeline: VoicePipeline) -> None:
    reply = ""
    transcript = ""
    no_speech = True
    tools = []
    speech = None
    public_events = []
    for event in events:
        if event.kind == "transcript":
            transcript = event.text
            no_speech = event.text == "" and speech is None
        elif event.kind == "tool" and event.tool is not None:
            tools.append(event.tool)
        elif event.kind == "audio" and event.audio is not None:
            speech = event.audio if speech is None else _concat_audio(speech, event.audio)
            no_speech = False
        elif event.kind in {"reply", "done"}:
            reply = event.text or reply
            no_speech = False if reply or tools else no_speech
        public_events.append(
            {
                "kind": event.kind,
                "text": event.text,
                "duration_s": None if event.audio is None else round(event.audio.duration_s, 4),
                "tool": None if event.tool is None else _tool_dict(event.tool),
            }
        )
    if transcript:
        no_speech = False
    audio_path = _write_audio(args.out, speech)
    _write_log(pipeline, args.latency_log)
    if args.json:
        print(
            json.dumps(
                {
                    "transcript": transcript,
                    "reply": reply,
                    "no_speech": no_speech and not transcript,
                    "tool_results": [_tool_dict(item) for item in tools],
                    "audio": audio_path,
                    "events": public_events,
                    "latency": pipeline.latency.as_dicts(),
                    "note": "durations were measured by this process for the configured backends",
                }
            )
        )
        return
    _print_human(transcript, reply, tools, no_speech and not transcript, audio_path)
    print()
    print(pipeline.latency.markdown_table())


def _print_human(
    transcript: str,
    reply: str,
    tools: list[ToolResult],
    no_speech: bool,
    audio_path: str | None,
) -> None:
    if no_speech:
        print("no speech detected")
    else:
        print(f"transcript: {transcript}")
    for item in tools:
        print(f"tool {item.name}: {item.text}")
    if reply:
        print(f"reply: {reply}")
    if audio_path:
        print(f"audio: {audio_path}")
    elif reply:
        print("audio: not saved")


def _write_audio(path: str | None, audio: Audio | None) -> str | None:
    if path is None or audio is None or audio.n_samples == 0:
        return None
    destination = Path(path)
    write_wav(destination, audio)
    return str(destination)


def _write_log(pipeline: VoicePipeline, override: str | None) -> None:
    raw = override if override is not None else pipeline.config.latency.log_path
    if not raw:
        return
    pipeline.latency.write_jsonl(Path(raw))


def _tool_dict(item: ToolResult) -> dict[str, object]:
    return {"name": item.name, "ok": item.ok, "text": item.text}


def _concat_audio(left: Audio, right: Audio) -> Audio:
    if left.sample_rate != right.sample_rate:
        right = resample_linear(right, left.sample_rate)
    return concat([left, right])
