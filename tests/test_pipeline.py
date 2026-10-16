import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from offline_voice.audio import concat, silence, tone
from offline_voice.config import load_config
from offline_voice.factory import build_pipeline
from offline_voice.llm import FakeLLM


def _clock() -> datetime:
    return datetime(2026, 10, 16, 12, 0, 0)


def _utterance():
    return concat(
        [
            silence(0.3, 16000),
            tone(0.5, 16000, 440.0, amplitude=0.5),
            silence(0.4, 16000),
        ]
    )


def test_turn_audio_runs_each_stage_and_calls_time(tmp_path: Path) -> None:
    pipeline = build_pipeline(load_config("config/demo.yaml"), clock=_clock)
    result = pipeline.run_turn_audio(_utterance())
    assert result.transcript == "what time is it"
    assert result.tool_results[0].name == "time"
    assert result.tool_results[0].text == "2026-10-16T12:00:00"
    assert "2026-10-16T12:00:00" in result.reply
    assert result.speech is not None
    stages = [record.stage for record in pipeline.latency.records]
    assert stages[:3] == ["vad", "stt", "llm"]
    assert "tools" in stages
    assert "tts" in stages
    for record in pipeline.latency.records:
        assert record.measured_by == "script"
    log_path = tmp_path / "latency.jsonl"
    pipeline.latency.write_jsonl(log_path)
    row = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["measured_by"] == "script"
    assert "measured by script" in pipeline.latency.markdown_table()


def test_silence_skips_speech_stages() -> None:
    pipeline = build_pipeline(load_config("config/demo.yaml"), clock=_clock)
    result = pipeline.run_turn_audio(silence(0.8, 16000))
    assert result.no_speech
    assert [record.stage for record in pipeline.latency.records] == ["vad"]
    assert pipeline.latency.records[0].detail == "no-speech"


def test_text_tools_weather_and_robot() -> None:
    pipeline = build_pipeline(load_config("config/demo.yaml"), clock=_clock)
    weather = pipeline.run_turn_text("what is the weather outside")
    assert weather.tool_results[0].name == "weather"
    assert "No forecast was fetched" in weather.reply
    robot = build_pipeline(load_config("config/demo.yaml"), clock=_clock).run_turn_text("please look left")
    assert robot.tool_results[0].text == "Queued robot command: look_left."


def test_stream_mode_speaks_after_the_tool() -> None:
    config = replace(load_config("config/demo.yaml"), mode="stream")
    pipeline = build_pipeline(config, clock=_clock)
    events = list(pipeline.run_stream_text("what time is it"))
    kinds = [event.kind for event in events]
    assert "tool" in kinds
    assert "audio" in kinds
    assert kinds[-1] == "done"
    tool = next(event.tool for event in events if event.kind == "tool")
    assert tool is not None
    assert tool.text == "2026-10-16T12:00:00"
    reply = next(event.text for event in events if event.kind == "reply")
    assert "2026-10-16T12:00:00" in reply
    assert all(event.audio is None or event.audio.n_samples > 0 for event in events)


def test_tool_round_limit() -> None:
    config = replace(load_config("config/demo.yaml"), max_tool_rounds=2)
    pipeline = build_pipeline(config, clock=_clock)
    raw = '<tool>{"name":"time","arguments":{}}</tool>'
    pipeline.llm = FakeLLM(script=[raw, raw, raw])
    result = pipeline.run_turn_text("loop")
    assert len(result.tool_results) == 2
    assert result.reply == "Tool-call limit reached for this turn."
