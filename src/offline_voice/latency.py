"""Per-stage timings measured by this process.

Every record is labeled `measured by script`. The module does not ship sample
numbers, and it does not estimate hardware it did not run on.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Measurement:
    detail: str = ""


@dataclass(frozen=True)
class StageTiming:
    stage: str
    duration_ms: float
    backend: str
    measured_by: str = "script"
    detail: str = ""


class LatencyLog:
    def __init__(self, backend_for: dict[str, str] | None = None) -> None:
        self.backend_for = dict(backend_for or {})
        self.records: list[StageTiming] = []

    @contextmanager
    def measure(self, stage: str) -> Iterator[Measurement]:
        bucket = Measurement()
        started = time.perf_counter()
        try:
            yield bucket
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.add(stage, elapsed_ms, detail=bucket.detail)

    def add(self, stage: str, duration_ms: float, detail: str = "") -> None:
        self.records.append(
            StageTiming(
                stage=stage,
                duration_ms=duration_ms,
                backend=self.backend_for.get(stage, ""),
                measured_by="script",
                detail=detail,
            )
        )

    def totals_ms(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for record in self.records:
            totals[record.stage] = totals.get(record.stage, 0.0) + record.duration_ms
        return totals

    def markdown_table(self) -> str:
        lines = [
            "Timings were measured by this script for the backends named in each row.",
            "",
            "| Stage | Backend | Duration (ms) | Source |",
            "| --- | --- | ---: | --- |",
        ]
        for record in self.records:
            lines.append(
                f"| {record.stage} | {record.backend} | {record.duration_ms:.2f} | measured by script |"
            )
        total = sum(record.duration_ms for record in self.records)
        lines.append(f"| total |  | {total:.2f} | measured by script |")
        return "\n".join(lines)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record)) + "\n")

    def as_dicts(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.records]
