"""Retrieval observability for the TUI.

Every actual recall produces a ``RetrievalEvent`` with the real stage timings
from the engine (compile / embed / search / pack), the hits returned, and a
human-readable explanation of *why* the path was fast or slow. The TUI renders
these live; nothing here is fabricated or estimated.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime

from .engine.memory import RecallReport


@dataclass
class RetrievalEvent:
    """One real retrieval, measured by the engine."""

    query: str
    started_at: datetime = field(default_factory=datetime.now)
    report: RecallReport | None = None
    hits: int = 0
    used_fallback: bool = False
    exception: str | None = None

    @property
    def total_ms(self) -> float:
        if self.report is None:
            return 0.0
        return (
            self.report.compile_ms
            + self.report.embed_ms
            + self.report.search_ms
            + self.report.pack_ms
        )

    def explain(self) -> list[str]:
        """Why was this fast/slow? Based only on what really ran."""
        r = self.report
        if r is None:
            return ["No measurement available"]
        lines = []
        if r.compile_ms <= 0.1:
            lines.append("Query compiled instantly")
        else:
            lines.append(f"Query compiled in {r.compile_ms:.3f} ms")
        if r.embed_ms > 0:
            lines.append(f"Embedding generated in {r.embed_ms:.3f} ms")
        else:
            lines.append("Embedding generation skipped")
        lines.append(f"Search evaluated {self.hits} candidate(s) "
                     f"in {r.search_ms:.3f} ms")
        if r.pack and r.pack.used_fallback:
            lines.append("Fallback used — broad candidate pass")
        elif r.pack and r.pack.sufficient:
            lines.append("Evidence sufficient — no deep rerank needed")
        else:
            lines.append("Evidence insufficient — conservative answer")
        return lines


@dataclass
class RetrievalTracker:
    """Rolling history + performance statistics of real retrievals."""

    events: list[RetrievalEvent] = field(default_factory=list)
    _started: float = field(default_factory=time.monotonic)

    def record(self, event: RetrievalEvent) -> None:
        self.events.append(event)
        if len(self.events) > 1000:
            del self.events[: len(self.events) - 1000]

    def clear(self) -> None:
        self.events.clear()

    @property
    def count(self) -> int:
        return len(self.events)

    def latencies(self) -> list[float]:
        return [e.total_ms for e in self.events]

    def percentile(self, pct: float) -> float:
        vals = sorted(self.latencies())
        if not vals:
            return 0.0
        idx = min(len(vals) - 1, int(pct / 100 * len(vals)))
        return vals[idx]

    def avg(self) -> float:
        vals = self.latencies()
        return statistics.mean(vals) if vals else 0.0

    def fast_path_rate(self) -> float:
        """Fraction of retrievals that skipped deep/embedding work."""
        if not self.events:
            return 0.0
        fast = sum(1 for e in self.events if e.report and e.report.embed_ms == 0)
        return fast / len(self.events)

    def hit_rate(self) -> float:
        """Fraction of retrievals that found sufficient evidence."""
        if not self.events:
            return 0.0
        hits = sum(1 for e in self.events if e.report and e.report.sufficient)
        return hits / len(self.events)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "avg_ms": round(self.avg(), 4),
            "p50_ms": round(self.percentile(50), 4),
            "p95_ms": round(self.percentile(95), 4),
            "p99_ms": round(self.percentile(99), 4),
            "fast_path_rate": round(self.fast_path_rate(), 3),
            "hit_rate": round(self.hit_rate(), 3),
        }


__all__ = ["RetrievalEvent", "RetrievalTracker"]