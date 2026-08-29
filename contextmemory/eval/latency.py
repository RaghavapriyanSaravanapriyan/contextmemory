"""Deterministic latency benchmarking for memory systems.

Latency in the field is dominated by LLM/network time on the read path. To
measure the memory system's *own* deterministic cost we run the write path
(ingest) and read path (answer) against a ``NullReader`` that returns
instantly, so the numbers reflect only the system's in-process overhead.
This isolates the property the field does not measure and ContextMemory
targets: sub-200ms interactive reads with a deterministic (LLM-free) read
path.

Run with the same ``system_factory`` used elsewhere so every system is
measured on the same synthetic workload. Wall-clock times are inherently
machine-dependent; on one rig the cross-system comparison is the signal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, median
from time import perf_counter

from .protocol import MemorySystem, Session, Turn


class NullReader:
    """Returns instantly, isolating the system's own deterministic cost."""

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        return ""


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    n: int

    @classmethod
    def from_samples(cls, samples_s: list[float]) -> LatencyStats:
        if not samples_s:
            raise ValueError("cannot build LatencyStats from no samples")
        ordered = sorted(samples_s)
        p95 = ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]
        return cls(
            p50_ms=median(ordered) * 1000,
            p95_ms=p95 * 1000,
            mean_ms=mean(ordered) * 1000,
            min_ms=ordered[0] * 1000,
            max_ms=ordered[-1] * 1000,
            n=len(ordered),
        )


@dataclass
class LatencyReport:
    ingest: LatencyStats
    answer: LatencyStats
    n_sessions: int
    n_probes: int

    def summary(self) -> str:
        ingest = self.ingest
        answer = self.answer
        lines = [
            f"ingest  p50 {ingest.p50_ms:8.3f} ms  p95 {ingest.p95_ms:8.3f} ms"
            f"  mean {ingest.mean_ms:8.3f} ms  (n={ingest.n})",
            f"answer  p50 {answer.p50_ms:8.3f} ms  p95 {answer.p95_ms:8.3f} ms"
            f"  mean {answer.mean_ms:8.3f} ms  (n={answer.n})",
        ]
        return "\n".join(lines)


def synthetic_workload(
    n_sessions: int,
    probes: list[str],
    seed: int = 42,
) -> tuple[list[Session], list[tuple[str, datetime]]]:
    """Build a deterministic synthetic corpus of ``n_sessions`` sessions.

    Sessions are simple turn sequences with varying content so ingest and
    read paths have real work to do. Returns (sessions, [(probe, date)]).
    """
    topics = [
        "weather forecast",
        "favorite restaurant",
        "weekend plans",
        "work project status",
        "family update",
        "travel itinerary",
        "new hobby",
        "budget discussion",
    ]
    base = datetime(2024, 1, 1)
    sessions: list[Session] = []
    for i in range(n_sessions):
        topic = topics[i % len(topics)]
        turns = [
            Turn(role="user", content=f"Session {i}: talking about {topic}."),
            Turn(role="assistant", content=f"Understood, continuing on {topic}."),
            Turn(role="user", content=f"More details on {topic} for session {i}."),
        ]
        sessions.append(
            Session(session_id=f"s{i}", timestamp=base + timedelta(days=i), turns=turns)
        )
    date = base + timedelta(days=n_sessions + 1)
    return sessions, [(p, date) for p in probes]


def bench_latency(
    system_factory: Callable[[], MemorySystem],
    n_sessions: int = 200,
    probes: list[str] | None = None,
    seed: int = 42,
) -> LatencyReport:
    """Measure deterministic ingest and answer latency for one system.

    A single fresh system ingests the full synthetic corpus (timed per
    session), then answers each probe (timed per probe) against a
    ``NullReader``. p50/p95 are reported in milliseconds.
    """
    if probes is None:
        probes = [
            "What was discussed in the most recent session?",
            "What topic came up most often?",
            "Summarize the user's weekend plans.",
            "What was the user's budget discussion about?",
        ]
    sessions, probe_dates = synthetic_workload(n_sessions, probes, seed=seed)
    system = system_factory()
    ingest_samples: list[float] = []
    for session in sessions:
        t0 = perf_counter()
        system.ingest(session)
        ingest_samples.append(perf_counter() - t0)

    answer_samples: list[float] = []
    for probe, date in probe_dates:
        t0 = perf_counter()
        system.answer(probe, date)
        answer_samples.append(perf_counter() - t0)

    return LatencyReport(
        ingest=LatencyStats.from_samples(ingest_samples),
        answer=LatencyStats.from_samples(answer_samples),
        n_sessions=n_sessions,
        n_probes=len(probe_dates),
    )