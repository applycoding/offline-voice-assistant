"""Local language-model backends and the tool-call text protocol.

Real models are asked to emit a tool as `<tool>{...json...}</tool>`. The fake
backend follows the same protocol so tests cover the parser the device uses.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from offline_voice.errors import MissingModelError, PipelineError
from offline_voice.optional import optional_module

_TOOL_OPEN = "<tool>"
_TOOL_CLOSE = "</tool>"


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    name: str | None = None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    raw: str


class LanguageModel(Protocol):
    def complete(self, messages: list[Message]) -> LLMResponse:
        """Return one full reply, including any tool blocks."""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield raw text fragments, including any tool blocks."""


def tool_call_from_json(body: str) -> ToolCall:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"tool block is not valid JSON: {body.strip()[:180]}") from exc
    if not isinstance(payload, dict):
        raise PipelineError("tool block must be a JSON object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PipelineError("tool block is missing a name")
    arguments = payload.get("arguments", payload.get("args", {}))
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise PipelineError("tool arguments must be a JSON object")
    return ToolCall(name=name.strip(), arguments=arguments)


def parse_tool_calls(raw: str) -> tuple[str, list[ToolCall]]:
    calls: list[ToolCall] = []
    spoken: list[str] = []
    cursor = 0
    while True:
        start = raw.find(_TOOL_OPEN, cursor)
        if start < 0:
            spoken.append(raw[cursor:])
            break
        spoken.append(raw[cursor:start])
        end = raw.find(_TOOL_CLOSE, start + len(_TOOL_OPEN))
        if end < 0:
            raise PipelineError("unterminated <tool> block")
        body = raw[start + len(_TOOL_OPEN) : end]
        calls.append(tool_call_from_json(body))
        cursor = end + len(_TOOL_CLOSE)
    return "".join(spoken).strip(), calls


def chat_messages(messages: list[Message]) -> list[dict[str, str]]:
    """Map tool results onto the user role.

    llama.cpp and Ollama chat templates in this demo accept system, user, and
    assistant. Tool results are passed back as user text with a stable prefix.
    """

    converted: list[dict[str, str]] = []
    for message in messages:
        if message.role == "tool":
            label = message.name or "tool"
            converted.append({"role": "user", "content": f"[tool result:{label}] {message.content}"})
        else:
            converted.append({"role": message.role, "content": message.content})
    return converted


class FakeLLM:
    """Keyword router used by the demo config and the unit tests.

    A script of raw replies, when provided, is consumed in order. Otherwise the
    backend emits the same `<tool>` protocol a local model is prompted to use.
    """

    def __init__(self, script: list[str] | None = None, chunk_size: int = 7) -> None:
        self._script = list(script) if script is not None else None
        self.chunk_size = chunk_size
        self.calls: list[list[Message]] = []

    def complete(self, messages: list[Message]) -> LLMResponse:
        raw = self._next_raw(messages)
        text, calls = parse_tool_calls(raw)
        return LLMResponse(text=text, tool_calls=calls, raw=raw)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        raw = self._next_raw(messages)
        size = max(1, self.chunk_size)
        for index in range(0, len(raw), size):
            yield raw[index : index + size]

    def _next_raw(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        if self._script is not None:
            if not self._script:
                return "Okay."
            return self._script.pop(0)
        return _heuristic_raw(messages)


def _heuristic_raw(messages: list[Message]) -> str:
    if messages and messages[-1].role == "tool":
        name = messages[-1].name or "tool"
        return f"Done. {name} reported: {messages[-1].content}"
    user_text = _latest_user_text(messages)
    lowered = user_text.lower()
    if re.search(r"\bweather\b", lowered):
        return '<tool>{"name":"weather","arguments":{"location":"the lab"}}</tool>'
    command = _robot_command(lowered)
    if command is not None:
        payload = json.dumps({"name": "robot", "arguments": {"command": command}})
        return f"<tool>{payload}</tool>"
    if re.search(r"\b(time|clock)\b", lowered):
        return '<tool>{"name":"time","arguments":{}}</tool>'
    return (
        "I am the offline demo assistant. "
        "I can tell the time, stub the weather, or send a robot command."
    )


def _latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _robot_command(lowered: str) -> str | None:
    if re.search(r"\blook left\b", lowered) or "look_left" in lowered:
        return "look_left"
    if re.search(r"\blook right\b", lowered) or "look_right" in lowered:
        return "look_right"
    for name in ("wave", "nod", "stop"):
        if re.search(rf"\b{name}\b", lowered):
            return name
    return None


class LlamaCppLLM:
    def __init__(self, model_path: str, n_ctx: int, temperature: float, max_tokens: int) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm: Any = None

    def complete(self, messages: list[Message]) -> LLMResponse:
        result = self._load().create_chat_completion(
            messages=chat_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = result["choices"][0]["message"].get("content") or ""
        text, calls = parse_tool_calls(raw)
        return LLMResponse(text=text, tool_calls=calls, raw=raw)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        chunks = self._load().create_chat_completion(
            messages=chat_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in chunks:
            delta = chunk["choices"][0].get("delta") or {}
            token = delta.get("content") or ""
            if token:
                yield token

    def _load(self) -> Any:
        if self._llm is not None:
            return self._llm
        if not self.model_path.is_file():
            raise MissingModelError(
                f"GGUF model not found at {self.model_path}. "
                "Run: python scripts/download_models.py --llm"
            )
        module = optional_module(
            "llama_cpp",
            "llama-cpp-python is not installed. Install the llm extra: "
            "pip install 'offline-voice-assistant[llm]'",
        )
        self._llm = module.Llama(model_path=str(self.model_path), n_ctx=self.n_ctx, verbose=False)
        return self._llm


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_s: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

    def complete(self, messages: list[Message]) -> LLMResponse:
        data = self._post(self._payload(messages, stream=False))
        raw = data.get("message", {}).get("content") or ""
        text, calls = parse_tool_calls(raw)
        return LLMResponse(text=text, tool_calls=calls, raw=raw)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        request = self._request(self._payload(messages, stream=True))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for line in response:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    data = json.loads(stripped.decode("utf-8"))
                    content = data.get("message", {}).get("content") or ""
                    if content:
                        yield content
                    if data.get("done"):
                        break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise PipelineError(f"Ollama stream failed: {exc}") from exc

    def _payload(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": chat_messages(messages),
            "stream": stream,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }

    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        body = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "offline-voice-assistant"},
            method="POST",
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise PipelineError(
                f"Ollama request to {self.base_url} failed. "
                f"Is `ollama serve` running with model {self.model}? {exc}"
            ) from exc
