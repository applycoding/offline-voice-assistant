import pytest

from offline_voice.errors import PipelineError
from offline_voice.llm import FakeLLM, Message, parse_tool_calls
from offline_voice.streaming import IncrementalToolParser, SentenceBuffer


def test_parse_tool_call_and_spoken_text() -> None:
    raw = 'Sure. <tool>{"name":"robot","arguments":{"command":"nod"}}</tool>'
    text, calls = parse_tool_calls(raw)
    assert text == "Sure."
    assert calls[0].name == "robot"
    assert calls[0].arguments == {"command": "nod"}


def test_parse_rejects_truncated_tool() -> None:
    with pytest.raises(PipelineError, match="unterminated"):
        parse_tool_calls("<tool>{\"name\":\"time\"}")


def test_incremental_parser_splits_a_tag_across_chunks() -> None:
    raw = '<tool>{"name":"time","arguments":{}}</tool>'
    parser = IncrementalToolParser()
    pieces = []
    for index in range(0, len(raw), 3):
        pieces.extend(parser.feed(raw[index : index + 3]))
    pieces.extend(parser.flush())
    tools = [piece for piece in pieces if piece.kind == "tool"]
    speech = "".join(piece.text for piece in pieces if piece.kind == "speech")
    assert len(tools) == 1
    assert tools[0].call is not None
    assert tools[0].call.name == "time"
    assert speech.strip() == ""


def test_fake_llm_routes_keywords() -> None:
    model = FakeLLM()
    weather = model.complete([Message("user", "what is the weather")])
    assert weather.tool_calls[0].name == "weather"
    robot = model.complete([Message("user", "please wave")])
    assert robot.tool_calls[0].arguments["command"] == "wave"
    clock = model.complete([Message("user", "what time is it")])
    assert clock.tool_calls[0].name == "time"
    follow = model.complete([Message("tool", "2026-10-16T12:00:00", name="time")])
    assert "2026-10-16T12:00:00" in follow.text
    assert follow.tool_calls == []


def test_sentence_buffer_waits_for_punctuation() -> None:
    buffer = SentenceBuffer()
    assert buffer.feed("Hello") == []
    assert buffer.feed(" there. Next") == ["Hello there."]
    assert buffer.flush() == "Next"
