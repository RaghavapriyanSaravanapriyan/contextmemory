"""Tests for LongMemEval data loading."""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from contextmemory.eval import (
    QuestionInstance,
    load_longmemeval,
    parse_longmemeval_date,
)

SAMPLE = [
    {
        "question_id": "q1",
        "question_type": "single-session-user",
        "question": "What city do I live in?",
        "answer": "San Francisco",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/04/10 (Mon) 14:47"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "I live in San Francisco.",
                    "has_answer": True,
                },
                {"role": "assistant", "content": "Great!"},
            ],
            [{"role": "user", "content": "Raining today."}],
        ],
        "answer_session_ids": ["s1"],
    },
    {
        "question_id": "q2_abs",
        "question_type": "multi-session",
        "question": "Did we discuss X?",
        "answer": "No discussion",
        "question_date": "2023/04/11 (Mon) 09:00",
        "haystack_session_ids": ["s3"],
        "haystack_dates": ["2023/04/10 (Mon) 10:00"],
        "haystack_sessions": [[{"role": "user", "content": "hello"}]],
        "answer_session_ids": [],
    },
]


def _write_sample(tmp_path, data) -> str:
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_parse_longmemeval_date() -> None:
    assert parse_longmemeval_date("2023/04/10 (Mon) 17:50") == datetime(
        2023, 4, 10, 17, 50
    )
    assert parse_longmemeval_date("2023/04/10 (Mon)") == datetime(2023, 4, 10)
    with pytest.raises(ValueError):
        parse_longmemeval_date("not a date")


def test_load_longmemeval(tmp_path) -> None:
    instances = load_longmemeval(_write_sample(tmp_path, SAMPLE))
    assert len(instances) == 2
    inst = instances[0]
    assert isinstance(inst, QuestionInstance)
    assert inst.question_id == "q1"
    assert inst.question_type == "single-session-user"
    assert inst.answer == "San Francisco"
    assert inst.question_date == datetime(2023, 4, 10, 23, 7)
    assert not inst.is_abstention
    assert len(inst.sessions) == 2
    assert inst.answer_session_ids == ["s1"]
    first = inst.sessions[0]
    assert first.session_id == "s1"
    assert first.timestamp == datetime(2023, 4, 10, 17, 50)
    assert first.turns[0].role == "user"
    assert first.turns[0].has_answer is True


def test_abstention_flag(tmp_path) -> None:
    instances = load_longmemeval(_write_sample(tmp_path, SAMPLE))
    assert instances[1].is_abstention


def test_load_rejects_mismatched_haystack(tmp_path) -> None:
    bad = [dict(SAMPLE[0], haystack_session_ids=["s1"])]
    with pytest.raises(ValueError):
        load_longmemeval(_write_sample(tmp_path, bad))