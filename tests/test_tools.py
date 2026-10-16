from datetime import datetime

from offline_voice.llm import ToolCall
from offline_voice.tools import ToolRegistry


def test_time_uses_injected_clock() -> None:
    registry = ToolRegistry(clock=lambda: datetime(2026, 10, 16, 12, 0, 0))
    result = registry.execute(ToolCall("time", {}))
    assert result.ok
    assert result.text == "2026-10-16T12:00:00"


def test_weather_stub_does_not_claim_a_forecast() -> None:
    registry = ToolRegistry()
    result = registry.execute(ToolCall("weather", {"location": "hangar"}))
    assert result.ok
    assert "No forecast was fetched" in result.text
    assert "hangar" in result.text


def test_robot_allowlist() -> None:
    registry = ToolRegistry()
    ok = registry.execute(ToolCall("robot", {"command": "look left"}))
    assert ok.ok
    assert "look_left" in ok.text
    rejected = registry.execute(ToolCall("robot", {"command": "dance"}))
    assert not rejected.ok
    assert "Unknown robot command" in rejected.text


def test_unknown_tool_and_disabled() -> None:
    registry = ToolRegistry()
    unknown = registry.execute(ToolCall("search", {}))
    assert not unknown.ok
    disabled = ToolRegistry(enabled=False)
    blocked = disabled.execute(ToolCall("time", {}))
    assert not blocked.ok
