"""Small offline tools the assistant can call.

`time` reads the device clock. `weather` is a stub and does not fetch a
forecast. `robot` acknowledges a gesture name and does not drive hardware.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from offline_voice.llm import ToolCall

ROBOT_COMMANDS = ("wave", "nod", "stop", "look_left", "look_right")


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    text: str


class ToolRegistry:
    def __init__(
        self,
        enabled: bool = True,
        clock: Callable[[], datetime] | None = None,
        weather_location: str = "the lab",
    ) -> None:
        self.enabled = enabled
        self.clock = clock or datetime.now
        self.weather_location = weather_location

    def specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "time",
                "description": "Local time on this device. No network.",
                "arguments": {},
            },
            {
                "name": "weather",
                "description": "Offline stub. Does not look up weather.",
                "arguments": {"location": "string"},
            },
            {
                "name": "robot",
                "description": "Queue a humanoid gesture stub.",
                "arguments": {"command": list(ROBOT_COMMANDS)},
            },
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        if not self.enabled:
            return ToolResult(call.name, False, "Tools are disabled.")
        if call.name == "time":
            stamp = self.clock().isoformat(timespec="seconds")
            return ToolResult("time", True, stamp)
        if call.name == "weather":
            location = str(call.arguments.get("location") or self.weather_location)
            return ToolResult(
                "weather",
                True,
                f"Offline weather stub. No forecast was fetched. Location noted: {location}.",
            )
        if call.name == "robot":
            command = str(call.arguments.get("command", "")).strip().lower().replace(" ", "_")
            if command not in ROBOT_COMMANDS:
                allowed = ", ".join(ROBOT_COMMANDS)
                return ToolResult("robot", False, f"Unknown robot command '{command}'. Allowed: {allowed}.")
            return ToolResult("robot", True, f"Queued robot command: {command}.")
        return ToolResult(call.name, False, f"Unknown tool '{call.name}'.")
