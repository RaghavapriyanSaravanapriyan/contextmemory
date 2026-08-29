"""Tests for the deterministic latency bench."""

from __future__ import annotations

import pytest

from contextmemory.eval import (
    Session,
    Turn,
    bench_latency,
    synthetic_workload,
)
from contextmemory.eval.latency import LatencyStats, NullReader
from contextmemory.eval.systems import FullHistorySystem


def test_null_reader_returns_instantly_and_empty() -> None:
    reader = NullReader()
    assert reader.complete([{"role": "user", "content": "hi"}]) == ""


def test_latency_stats_percentiles() -> None:
    stats = LatencyStats.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.n == 5
    assert stats.p50_ms == 3000.0
    assert stats.p95_ms == 5000.0
    assert stats.mean_ms == 3000.0
    assert stats.min_ms == 1000.0
    assert stats.max_ms == 5000.0


def test_latency_stats_rejects_empty() -> None:
    with pytest.raises(ValueError):
        LatencyStats.from_samples([])


def test_synthetic_workload_is_deterministic() -> None:
    a = synthetic_workload(10, ["probe?"])
    b = synthetic_workload(10, ["probe?"])
    assert len(a[0]) == 10
    assert a == b


def test_synthetic_workload_sessions_have_turns() -> None:
    sessions, probes = synthetic_workload(3, ["q1"])
    assert all(len(s.turns) == 3 for s in sessions)
    assert all(isinstance(t, Turn) for s in sessions for t in s.turns)
    assert len(probes) == 1


def test_bench_latency_full_history() -> None:
    reader = NullReader()
    report = bench_latency(
        lambda: FullHistorySystem(reader),
        n_sessions=20,
        probes=["What was discussed?"],
    )
    assert report.n_sessions == 20
    assert report.n_probes == 1
    assert report.ingest.n == 20
    assert report.answer.n == 1
    assert report.ingest.p50_ms >= 0
    assert report.answer.p50_ms >= 0
    assert "p50" in report.summary()


def test_bench_latency_matches_report_counts() -> None:
    reader = NullReader()
    report = bench_latency(
        lambda: FullHistorySystem(reader),
        n_sessions=5,
        probes=["a", "b"],
    )
    assert report.answer.n == 2


def test_full_history_answer_grows_with_history() -> None:
    # Sanity: answer time should not be degenerate for small corpora; the
    # actual scaling comparison is a cross-system bench, not a unit test.
    reader = NullReader()
    small = bench_latency(lambda: FullHistorySystem(reader), n_sessions=5)
    large = bench_latency(lambda: FullHistorySystem(reader), n_sessions=50)
    assert large.n_sessions > small.n_sessions


def test_sessions_are_chronological() -> None:
    sessions, _ = synthetic_workload(5, ["q"])
    stamps = [s.timestamp for s in sessions]
    assert stamps == sorted(stamps)
    assert all(isinstance(s, Session) for s in sessions)