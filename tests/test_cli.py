import json
from pathlib import Path

import pytest

from offline_voice import cli
from offline_voice.audio import concat, silence, tone, write_wav
from offline_voice.errors import BackendNotInstalledError


def _wav(path: Path, audio) -> str:
    write_wav(path, audio)
    return str(path)


def test_cli_text_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "run",
            "--config",
            "config/demo.yaml",
            "--text",
            "what time is it",
            "--json",
            "--out",
            str(tmp_path / "reply.wav"),
            "--latency-log",
            str(tmp_path / "latency.jsonl"),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool_results"][0]["name"] == "time"
    assert payload["note"].startswith("durations were measured")
    assert payload["latency"][0]["measured_by"] == "script"
    assert Path(payload["audio"]).is_file()
    logged = (tmp_path / "latency.jsonl").read_text(encoding="utf-8")
    assert '"stage": "llm"' in logged


def test_cli_wav_and_silence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    utterance = concat(
        [silence(0.3, 16000), tone(0.5, 16000, 440.0, amplitude=0.5), silence(0.4, 16000)]
    )
    code = cli.main(
        [
            "run",
            "--config",
            "config/demo.yaml",
            "--wav",
            _wav(tmp_path / "in.wav", utterance),
            "--json",
            "--latency-log",
            "",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["transcript"] == "what time is it"
    assert payload["no_speech"] is False

    code = cli.main(
        [
            "run",
            "--config",
            "config/demo.yaml",
            "--wav",
            _wav(tmp_path / "quiet.wav", silence(0.6, 16000)),
            "--json",
            "--latency-log",
            "",
        ]
    )
    assert code == 0
    quiet = json.loads(capsys.readouterr().out)
    assert quiet["no_speech"] is True


def test_cli_stream_and_latency_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "latency",
            "--config",
            "config/demo.yaml",
            "--mode",
            "stream",
            "--text",
            "please nod",
            "--latency-log",
            str(tmp_path / "lat.jsonl"),
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "measured by script" in output
    assert "Queued robot command: nod." in output
    assert (tmp_path / "lat.jsonl").is_file()


def test_cli_requires_one_input(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["run", "--config", "config/demo.yaml", "--latency-log", ""])
    assert code == 2
    assert "choose one" in capsys.readouterr().err


def test_checked_in_sample_is_detected_as_speech(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "run",
            "--config",
            "config/demo.yaml",
            "--wav",
            "samples/utterance.wav",
            "--json",
            "--latency-log",
            "",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["no_speech"] is False
    assert any(row["stage"] == "stt" for row in payload["latency"])


def test_microphone_helper_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, hint: str):
        raise BackendNotInstalledError(hint)

    monkeypatch.setattr(cli, "optional_module", missing)
    with pytest.raises(BackendNotInstalledError, match="mic extra"):
        cli.record_microphone(0.2, 16000)
