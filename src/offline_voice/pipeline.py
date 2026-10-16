"""Turn-based and streaming voice turns.

Speech-to-text runs on a finished utterance in both modes. Streaming applies
to the language model and to sentence-sized speech synthesis. Classic Whisper
is not treated as a streaming recognizer here.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

from offline_voice.audio import Audio, resample_linear
from offline_voice.config import AppConfig
from offline_voice.latency import LatencyLog
from offline_voice.llm import LanguageModel, Message
from offline_voice.streaming import IncrementalToolParser, Piece, SentenceBuffer
from offline_voice.stt import SpeechToText
from offline_voice.tools import ToolRegistry, ToolResult
from offline_voice.tts import TextToSpeech
from offline_voice.vad import VoiceActivity


@dataclass
class TurnResult:
    transcript: str
    reply: str
    speech: Audio | None
    tool_results: list[ToolResult]
    no_speech: bool


@dataclass
class StreamEvent:
    kind: str
    text: str = ""
    audio: Audio | None = None
    tool: ToolResult | None = None


@dataclass
class VoicePipeline:
    vad: VoiceActivity
    stt: SpeechToText
    llm: LanguageModel
    tts: TextToSpeech
    tools: ToolRegistry
    latency: LatencyLog
    config: AppConfig

    def run_turn_audio(self, audio: Audio) -> TurnResult:
        prepared = resample_linear(audio, self.config.audio.sample_rate)
        with self.latency.measure("vad") as mark:
            speech = self.vad.trim(prepared)
            if speech is None:
                mark.detail = "no-speech"
        if speech is None:
            return self._empty_turn(no_speech=True)
        with self.latency.measure("stt"):
            transcript = self.stt.transcribe(speech).strip()
        if not transcript:
            return self._empty_turn(no_speech=False)
        return self._finish_turn(transcript)

    def run_turn_text(self, text: str) -> TurnResult:
        transcript = text.strip()
        if not transcript:
            return self._empty_turn(no_speech=False)
        return self._finish_turn(transcript)

    def run_stream_audio(self, audio: Audio) -> Iterator[StreamEvent]:
        prepared = resample_linear(audio, self.config.audio.sample_rate)
        with self.latency.measure("vad") as mark:
            speech = self.vad.trim(prepared)
            if speech is None:
                mark.detail = "no-speech"
        if speech is None:
            yield StreamEvent(kind="done", text="")
            return
        with self.latency.measure("stt"):
            transcript = self.stt.transcribe(speech).strip()
        yield StreamEvent(kind="transcript", text=transcript)
        if not transcript:
            yield StreamEvent(kind="done", text="")
            return
        yield from self._speak_with_tools(transcript)

    def run_stream_text(self, text: str) -> Iterator[StreamEvent]:
        transcript = text.strip()
        yield StreamEvent(kind="transcript", text=transcript)
        if not transcript:
            yield StreamEvent(kind="done", text="")
            return
        yield from self._speak_with_tools(transcript)

    def _finish_turn(self, transcript: str) -> TurnResult:
        reply, tool_results = self._complete_turn(transcript)
        spoken = self._synthesize(reply)
        return TurnResult(
            transcript=transcript,
            reply=reply,
            speech=spoken,
            tool_results=tool_results,
            no_speech=False,
        )

    def _empty_turn(self, no_speech: bool) -> TurnResult:
        return TurnResult(transcript="", reply="", speech=None, tool_results=[], no_speech=no_speech)

    def _complete_turn(self, user_text: str) -> tuple[str, list[ToolResult]]:
        messages = [Message("system", self.config.llm.system_prompt), Message("user", user_text)]
        results: list[ToolResult] = []
        for _ in range(self.config.max_tool_rounds):
            with self.latency.measure("llm"):
                response = self.llm.complete(messages)
            if not response.tool_calls:
                return response.text, results
            messages.append(Message("assistant", response.raw))
            with self.latency.measure("tools"):
                for call in response.tool_calls:
                    result = self.tools.execute(call)
                    results.append(result)
                    messages.append(Message("tool", result.text, name=result.name))
        return "Tool-call limit reached for this turn.", results

    def _synthesize(self, text: str) -> Audio | None:
        if not text.strip():
            return None
        with self.latency.measure("tts"):
            return self.tts.synthesize(text)

    def _speak_with_tools(self, user_text: str) -> Iterator[StreamEvent]:
        messages = [Message("system", self.config.llm.system_prompt), Message("user", user_text)]
        spoken: list[str] = []
        for _ in range(self.config.max_tool_rounds):
            parser = IncrementalToolParser()
            sentences = SentenceBuffer()
            for token in self._timed_tokens(messages):
                for piece in parser.feed(token):
                    yield from self._emit_piece(piece, sentences, spoken)
            for piece in parser.flush():
                yield from self._emit_piece(piece, sentences, spoken)
            tail = sentences.flush()
            if tail:
                yield from self._speak_sentence(tail, spoken)
            if not parser.tool_calls:
                reply = " ".join(spoken).strip()
                yield StreamEvent(kind="reply", text=reply)
                yield StreamEvent(kind="done", text=reply)
                return
            messages.append(Message("assistant", parser.raw))
            with self.latency.measure("tools"):
                for call in parser.tool_calls:
                    if call is None:
                        continue
                    result = self.tools.execute(call)
                    messages.append(Message("tool", result.text, name=result.name))
                    yield StreamEvent(kind="tool", text=result.text, tool=result)
        reply = "Tool-call limit reached for this turn."
        yield StreamEvent(kind="reply", text=reply)
        yield StreamEvent(kind="done", text=reply)

    def _timed_tokens(self, messages: list[Message]) -> Iterator[str]:
        spent = 0.0
        stream = iter(self.llm.stream(messages))
        try:
            while True:
                tick = time.perf_counter()
                try:
                    token = next(stream)
                except StopIteration:
                    spent += (time.perf_counter() - tick) * 1000.0
                    return
                spent += (time.perf_counter() - tick) * 1000.0
                yield token
        finally:
            self.latency.add("llm", spent)

    def _emit_piece(
        self,
        piece: Piece,
        sentences: SentenceBuffer,
        spoken: list[str],
    ) -> Iterator[StreamEvent]:
        if piece.kind == "speech":
            yield StreamEvent(kind="token", text=piece.text)
            for sentence in sentences.feed(piece.text):
                yield from self._speak_sentence(sentence, spoken)
            return
        if piece.kind == "tool":
            return
        raise ValueError(f"unknown stream piece '{piece.kind}'")

    def _speak_sentence(self, sentence: str, spoken: list[str]) -> Iterator[StreamEvent]:
        cleaned = sentence.strip()
        if not cleaned:
            return
        spoken.append(cleaned)
        with self.latency.measure("tts"):
            audio = self.tts.synthesize(cleaned)
        yield StreamEvent(kind="audio", text=cleaned, audio=audio)
