"""Tests for eval-harness baseline systems (incl. BEAM-scale recency)."""

from __future__ import annotations

from datetime import datetime

from contextmemory.eval.protocol import Session, Turn
from contextmemory.eval.systems import RecencyCharsSystem
from tests.conftest import FakeReader


def _session(sid: str, *lines: str) -> Session:
    return Session(
        session_id=sid,
        timestamp=datetime(2024, 1, 1),
        turns=[Turn(role="user", content=line) for line in lines],
    )


def test_recency_chars_keeps_only_tail() -> None:
    reader = FakeReader(default="ok")
    system = RecencyCharsSystem(reader, max_chars=200)
    system.ingest(_session("s1", "OLD-MARKER-" + "x" * 300))
    system.ingest(_session("s2", "NEW-MARKER-" + "y" * 10))
    system.answer("what?", datetime(2024, 2, 1))
    prompt = reader.calls[-1][-1]["content"]
    assert "NEW-MARKER" in prompt
    assert "OLD-MARKER" not in prompt


def test_recency_chars_single_short_session() -> None:
    reader = FakeReader(default="ok")
    system = RecencyCharsSystem(reader, max_chars=8000)
    system.ingest(_session("s1", "hello world"))
    assert system.answer("hi?", datetime(2024, 2, 1)) == "ok"
