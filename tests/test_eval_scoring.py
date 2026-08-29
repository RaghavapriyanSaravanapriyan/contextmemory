"""Tests for scoring."""

from __future__ import annotations

from datetime import datetime

from contextmemory.eval import (
    ScoreReport,
    build_anscheck_prompt,
    deterministic_match,
    judge_results,
    score_deterministic,
)
from contextmemory.eval.protocol import ReaderClient, Session, Turn
from contextmemory.eval.runner import ReplayResult, Timing

from .conftest import FakeReader


def _result(
    qtype: str,
    answer: str,
    hypothesis: str,
    abstention: bool = False,
    qid: str = "q1",
) -> ReplayResult:
    return ReplayResult(
        question_id=qid + ("_abs" if abstention else ""),
        question_type=qtype,
        question="question?",
        answer=answer,
        hypothesis=hypothesis,
        is_abstention=abstention,
        timing=Timing(),
    )


def test_deterministic_match_exact() -> None:
    assert deterministic_match("San Francisco", "San Francisco")
    assert deterministic_match("I live in san francisco", "San Francisco")


def test_deterministic_match_negatives() -> None:
    assert not deterministic_match("New York", "San Francisco")
    assert not deterministic_match("", "San Francisco")


def test_deterministic_match_content_words() -> None:
    assert deterministic_match(
        "The GPS system was not functioning correctly",
        "GPS system not functioning correctly",
    )
    assert not deterministic_match(
        "The GPS system is malfunctioning",
        "GPS system not functioning correctly",
    )


def test_score_deterministic_aggregation() -> None:
    results = [
        _result("single-session-user", "A", "A"),
        _result("single-session-user", "B", "A"),
        _result("temporal-reasoning", "A", "A"),
    ]
    report = score_deterministic(results)
    assert isinstance(report, ScoreReport)
    assert report.n == 3
    assert abs(report.overall - 2 / 3) < 1e-9
    assert report.per_type["single-session-user"] == 0.5
    assert report.per_type["temporal-reasoning"] == 1.0


def test_build_anscheck_prompt_abstention() -> None:
    prompt = build_anscheck_prompt(
        "multi-session", "q?", "expl", "resp", abstention=True
    )
    assert "unanswerable" in prompt
    assert "Does the model correctly identify the question as unanswerable?" in prompt


def test_build_anscheck_prompt_knowledge_update() -> None:
    prompt = build_anscheck_prompt(
        "knowledge-update", "q?", "ans", "resp", abstention=False
    )
    assert "updated answer" in prompt


def test_build_anscheck_prompt_preference() -> None:
    prompt = build_anscheck_prompt(
        "single-session-preference", "q?", "rubric", "resp", abstention=False
    )
    assert "Rubric" in prompt


def test_judge_results_sets_labels() -> None:
    reader = FakeReader(behaviors={"Question: question?": "yes"})
    results = [_result("single-session-user", "ans", "hyp")]
    report, labeled = judge_results(results, reader)
    assert labeled[0].judged is True
    assert report.overall == 1.0


def test_judge_yes_detection_is_case_insensitive() -> None:
    class UpperReader:
        def complete(self, messages, temperature=0.0) -> str:
            return "YES"

    reader = UpperReader()
    report, _ = judge_results([_result("multi-session", "a", "h")], reader)
    assert report.overall == 1.0


def test_reader_client_is_used_for_ingest_and_answer() -> None:
    reader = FakeReader()
    sys = _DummySystem(reader)
    session = Session(
        session_id="s1",
        timestamp=datetime(2023, 1, 1),
        turns=[Turn(role="user", content="hi")],
    )
    sys.ingest(session)
    assert sys.answer("question?", datetime(2023, 1, 2)) == "42"


class _DummySystem:
    def __init__(self, reader: ReaderClient) -> None:
        self._reader = reader

    def ingest(self, session: Session) -> None:
        pass

    def answer(self, question: str, question_date: datetime) -> str:
        return self._reader.complete([{"role": "user", "content": question}])