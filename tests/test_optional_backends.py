import urllib.error
import urllib.request

import pytest

from offline_voice.audio import silence
from offline_voice.errors import BackendNotInstalledError, MissingModelError, PipelineError
from offline_voice.llm import LlamaCppLLM, Message, OllamaLLM
from offline_voice.stt import FasterWhisperSTT
from offline_voice.tts import PiperTTS
from offline_voice.vad import SileroVAD, WebRtcVAD


def test_missing_weights_fail_before_any_download(tmp_path) -> None:
    missing_gguf = tmp_path / "missing.gguf"
    llm = LlamaCppLLM(str(missing_gguf), n_ctx=512, temperature=0.0, max_tokens=8)
    with pytest.raises(MissingModelError, match="download_models.py --llm"):
        llm.complete([Message("user", "hello")])

    piper = PiperTTS(str(tmp_path / "missing.onnx"))
    with pytest.raises(MissingModelError, match="download_models.py --piper"):
        piper.synthesize("hello")


def test_optional_imports_are_not_required(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, hint: str):
        raise BackendNotInstalledError(hint)

    monkeypatch.setattr("offline_voice.stt.optional_module", missing)
    monkeypatch.setattr("offline_voice.vad.optional_module", missing)
    stt = FasterWhisperSTT("tiny", "", "cpu", "int8", "en")
    with pytest.raises(BackendNotInstalledError):
        stt.transcribe(silence(0.1, 16000))
    webrtc = WebRtcVAD(2, 300, 200, 60, 30)
    with pytest.raises(BackendNotInstalledError):
        webrtc.trim(silence(0.2, 16000))
    silero = SileroVAD(0.5)
    with pytest.raises(BackendNotInstalledError):
        silero.trim(silence(0.2, 16000))


def test_ollama_error_stays_local(monkeypatch: pytest.MonkeyPatch) -> None:
    def offline(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    model = OllamaLLM("http://127.0.0.1:11434", "qwen2.5:0.5b", 0.0, 8, 1.0)
    with pytest.raises(PipelineError, match="Ollama"):
        model.complete([Message("user", "hello")])
