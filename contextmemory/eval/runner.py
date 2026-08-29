"""Replay-and-answer runner for memory evaluation.

Each question instance is independent: a fresh system is built, its
haystack sessions are replayed in chronological order through the write
path, then the question is asked. Timing is recorded on both paths so
latency can be compared across systems on the same rig.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from time import perf_counter

from .data import QuestionInstance
from .protocol import MemorySystem


@dataclass
class Timing:
    ingest_s: float = 0.0
    answer_s: float = 0.0


@dataclass
class ReplayResult:
    question_id: str
    question_type: str
    question: str
    answer: str
    hypothesis: str
    is_abstention: bool
    timing: Timing
    judged: bool | None = field(default=None)


def replay_instance(
    instance: QuestionInstance,
    system: MemorySystem,
) -> ReplayResult:
    """Replay one instance's sessions chronologically, then ask the question."""
    ordered = sorted(instance.sessions, key=lambda s: s.timestamp)
    timing = Timing()
    for session in ordered:
        start = perf_counter()
        system.ingest(session)
        timing.ingest_s += perf_counter() - start
    start = perf_counter()
    hypothesis = system.answer(instance.question, instance.question_date)
    timing.answer_s = perf_counter() - start
    return ReplayResult(
        question_id=instance.question_id,
        question_type=instance.question_type,
        question=instance.question,
        answer=instance.answer,
        hypothesis=hypothesis,
        is_abstention=instance.is_abstention,
        timing=timing,
    )


def replay(
    instances: list[QuestionInstance],
    system_factory: Callable[[], MemorySystem],
    progress: bool = True,
) -> list[ReplayResult]:
    """Replay a set of instances, building a fresh system for each."""
    results: list[ReplayResult] = []
    total = len(instances)
    for i, instance in enumerate(instances, start=1):
        system = system_factory()
        results.append(replay_instance(instance, system))
        if progress and (i % 50 == 0 or i == total):
            print(f"  replayed {i}/{total}")
    return results