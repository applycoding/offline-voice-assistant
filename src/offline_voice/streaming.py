"""Incremental split of a token stream into speech and tool calls."""

from __future__ import annotations

from dataclasses import dataclass

from offline_voice.errors import PipelineError
from offline_voice.llm import ToolCall, tool_call_from_json

_TOOL_OPEN = "<tool>"
_TOOL_CLOSE = "</tool>"


@dataclass(frozen=True)
class Piece:
    kind: str
    text: str = ""
    call: ToolCall | None = None


class IncrementalToolParser:
    def __init__(self) -> None:
        self._buffer = ""
        self._inside = False
        self.tool_calls: list[ToolCall] = []
        self._raw: list[str] = []

    @property
    def raw(self) -> str:
        return "".join(self._raw)

    def feed(self, token: str) -> list[Piece]:
        if token:
            self._raw.append(token)
            self._buffer += token
        return self._drain(final=False)

    def flush(self) -> list[Piece]:
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[Piece]:
        pieces: list[Piece] = []
        while self._buffer:
            if self._inside:
                close_at = self._buffer.find(_TOOL_CLOSE)
                if close_at < 0:
                    if final:
                        raise PipelineError("unterminated <tool> block")
                    return pieces
                body = self._buffer[:close_at]
                self._buffer = self._buffer[close_at + len(_TOOL_CLOSE) :]
                self._inside = False
                call = tool_call_from_json(body)
                self.tool_calls.append(call)
                pieces.append(Piece(kind="tool", call=call))
                continue
            open_at = self._buffer.find(_TOOL_OPEN)
            if open_at < 0:
                if final:
                    pieces.append(Piece(kind="speech", text=self._buffer))
                    self._buffer = ""
                    return pieces
                held = held_tag_suffix(self._buffer, _TOOL_OPEN)
                emit_until = len(self._buffer) - len(held)
                emit = self._buffer[:emit_until]
                self._buffer = held
                if emit:
                    pieces.append(Piece(kind="speech", text=emit))
                return pieces
            if open_at > 0:
                pieces.append(Piece(kind="speech", text=self._buffer[:open_at]))
            self._buffer = self._buffer[open_at + len(_TOOL_OPEN) :]
            self._inside = True
        return pieces


class SentenceBuffer:
    """Hold tokens until a sentence boundary, then release the sentence."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        ready: list[str] = []
        while True:
            end = _sentence_end(self._buffer)
            if end is None:
                return ready
            sentence = self._buffer[:end].strip()
            self._buffer = self._buffer[end:].lstrip()
            if sentence:
                ready.append(sentence)

    def flush(self) -> str | None:
        text = self._buffer.strip()
        self._buffer = ""
        return text or None


def held_tag_suffix(text: str, tag: str) -> str:
    """Return a suffix of `text` that is a proper prefix of `tag`, if any."""

    limit = min(len(text), len(tag) - 1)
    for size in range(limit, 0, -1):
        if tag.startswith(text[-size:]):
            return text[-size:]
    return ""


def _sentence_end(text: str) -> int | None:
    for index, char in enumerate(text):
        if char in ".!?" and index + 1 < len(text) and text[index + 1].isspace():
            return index + 1
    return None
