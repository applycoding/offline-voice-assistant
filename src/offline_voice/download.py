"""Download the optional Piper voice and a small GGUF language model.

The script prints a plan and does not contact the network unless `--piper` or
`--llm` is set. Checksums below are the Hugging Face Git-LFS object ids for
those exact files.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from offline_voice.errors import PipelineError

PIPER_ONNX_URL = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
    "en/en_US/lessac/medium/en_US-lessac-medium.onnx"
)
PIPER_JSON_URL = PIPER_ONNX_URL + ".json"
LLM_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

PIPER_ONNX_SHA256 = "5efe09e69902187827af646e1a6e9d269dee769f9877d17b16b1b46eeaaf019f"
PIPER_JSON_SHA256 = "efe19c417bed055f2d69908248c6ba650fa135bc868b0e6abb3da181dab690a0"
LLM_SHA256 = "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db"


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    filename: str
    sha256: str
    size_bytes: int
    note: str


ARTIFACTS: dict[str, Artifact] = {
    "piper-onnx": Artifact(
        name="piper-onnx",
        url=PIPER_ONNX_URL,
        filename="en_US-lessac-medium.onnx",
        sha256=PIPER_ONNX_SHA256,
        size_bytes=63_201_294,
        note="Piper en_US lessac medium voice",
    ),
    "piper-json": Artifact(
        name="piper-json",
        url=PIPER_JSON_URL,
        filename="en_US-lessac-medium.onnx.json",
        sha256=PIPER_JSON_SHA256,
        size_bytes=4_885,
        note="Piper voice config, kept next to the .onnx file",
    ),
    "llm": Artifact(
        name="llm",
        url=LLM_URL,
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        sha256=LLM_SHA256,
        size_bytes=491_400_032,
        note="Qwen2.5 0.5B Instruct Q4_K_M, about 469 MiB",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download optional offline voice models. With no flags, print the plan only."
    )
    parser.add_argument("--dest", default="models", help="Directory for downloaded files")
    parser.add_argument("--piper", action="store_true", help="Download the Piper lessac medium voice")
    parser.add_argument("--llm", action="store_true", help="Download Qwen2.5-0.5B-Instruct Q4_K_M")
    parser.add_argument("--force", action="store_true", help="Download again even when the checksum matches")
    args = parser.parse_args(argv)
    selected = _selected(args.piper, args.llm)
    if not selected:
        print(plan_text())
        return 0
    destination = Path(args.dest)
    try:
        for artifact in selected:
            path = download_artifact(artifact, destination, force=args.force)
            print(f"ready: {path}")
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _selected(piper: bool, llm: bool) -> list[Artifact]:
    chosen: list[Artifact] = []
    if piper:
        chosen.append(ARTIFACTS["piper-onnx"])
        chosen.append(ARTIFACTS["piper-json"])
    if llm:
        chosen.append(ARTIFACTS["llm"])
    return chosen


def plan_text() -> str:
    lines = [
        "No files were downloaded. Pass --piper and/or --llm to fetch weights.",
        "",
        "Piper voice (en_US lessac medium):",
        f"  file: {ARTIFACTS['piper-onnx'].filename} ({ARTIFACTS['piper-onnx'].size_bytes} bytes)",
        f"  url:  {ARTIFACTS['piper-onnx'].url}",
        f"  sha256: {ARTIFACTS['piper-onnx'].sha256}",
        f"  config sha256: {ARTIFACTS['piper-json'].sha256}",
        "",
        "Local LLM (llama.cpp GGUF):",
        f"  file: {ARTIFACTS['llm'].filename} ({ARTIFACTS['llm'].size_bytes} bytes)",
        f"  url:  {ARTIFACTS['llm'].url}",
        f"  sha256: {ARTIFACTS['llm'].sha256}",
        "",
        "Speech-to-text is downloaded by faster-whisper on first use, not by this script.",
        "A small CPU default is the `tiny` model, int8:",
        "  python -c \"from faster_whisper import WhisperModel; "
        "WhisperModel('tiny', device='cpu', compute_type='int8')\"",
        "",
        "Ollama is the other LLM backend. It keeps its own model store:",
        "  ollama pull qwen2.5:0.5b",
    ]
    return "\n".join(lines)


def download_artifact(artifact: Artifact, destination: Path, force: bool = False) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / artifact.filename
    if target.is_file() and not force and _sha256(target) == artifact.sha256:
        return target
    download_file(artifact.url, target, artifact.sha256)
    return target


def download_file(url: str, dest: Path, sha256: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".part")
    hasher = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "offline-voice-assistant"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            while True:
                block = response.read(1024 * 256)
                if not block:
                    break
                hasher.update(block)
                handle.write(block)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise PipelineError(f"download failed for {url}: {exc}") from exc
    digest = hasher.hexdigest()
    if digest != sha256:
        temporary.unlink(missing_ok=True)
        raise PipelineError(f"checksum mismatch for {dest.name}: expected {sha256}, got {digest}")
    temporary.replace(dest)


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 256)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()
