"""Tests for the replay runner."""

from __future__ import annotations

from datetime import datetime

from contextmemory.eval import (
    QuestionInstance,
    Session,
    Turn,
    replay,
    replay_instance,
)
from contextmemory.eval.runner import Timing
from contextmemory.eval.systems import RecencyWindowSystem

from .conftest import FakeReader


def _session(sid: str, dt: datetime, text: str) -> Session:
    return Session(
        session_id=sid,
        timestamp=dt,
        turns=[Turn(role="user", content=text)],
    )


def _instance(sessions: list[Session]) -> QuestionInstance:
    return QuestionInstance(
        question_id="q1",
        question_type="single-session-user",
        question="question?",
        answer="answer",
        question_date=datetime(2023, 5, 1),
        sessions=sessions,
    )


class _RecordingSystem:
    """Records ingest order and answers with a canned response."""

    def __init__(self, reader: FakeReader) -> None:
        self._reader = reader
        self.ingested: list[str] = []

    def ingest(self, session: Session) -> None:
        self.ingested.append(session.session_id)

    def answer(self, question: str, question_date: datetime) -> str:
        return self._reader.complete([{"role": "user", "content": question}])


def test_replay_instance_is_chronological() -> None:
    reader = FakeReader()
    system = _RecordingSystem(reader)
    sessions = [
        _session("s3", datetime(2023, 1, 3), "c"),
        _session("s1", datetime(2023, 1, 1), "a"),
        _session("s2", datetime(2023, 1, 2), "b"),
    ]
    result = replay_instance(_instance(sessions), system)
    assert system.ingested == ["s1", "s2", "s3"]
    assert result.question_id == "q1"
    assert result.hypothesis == "42"
    assert result.is_abstention is False


def test_replay_records_timing() -> None:
    reader = FakeReader()
    system = _RecordingSystem(reader)
    result = replay_instance(
        _instance([_session("s1", datetime(2023, 1, 1), "a")]), system
    )
    assert isinstance(result.timing, Timing)
    assert result.timing.ingest_s >= 0
    assert result.timing.answer_s >= 0


def test_replay_builds_fresh_system_per_instance() -> None:
    reader = FakeReader()
    inst1 = _instance([_session("s1", datetime(2023, 1, 1), "a")])
    inst2 = _instance([_session("s2", datetime(2023, 1, 1), "b")])

    results = replay(
        [inst1, inst2],
        lambda: _RecordingSystem(reader),
        progress=False,
    )
    assert len(results) == 2
    assert results[0].question_id == "q1"
    assert results[1].question_id == "q1"


def test_recency_window_system_drops_old_sessions() -> None:
    reader = FakeReader(default="unknown")
    system = RecencyWindowSystem(reader, window=2)
    for i in range(4):
        system.ingest(_session(f"s{i}", datetime(2023, 1, i + 1), f"msg {i}"))
    system.answer("question?", datetime(2023, 2, 1))
    prompt = reader.calls[-1][0]["content"]
    assert "msg 3" in prompt
    assert "msg 2" in prompt
    assert "msg 0" not in prompt
    assert "msg 1" not in prompt