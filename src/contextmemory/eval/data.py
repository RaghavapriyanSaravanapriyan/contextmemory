"""LongMemEval dataset loading.

Mirrors the released format of ``longmemeval-cleaned`` on HuggingFace
(oracle, S, M variants). See reports/research for benchmark context.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .protocol import Session, Turn

_DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def parse_longmemeval_date(value: str) -> datetime:
    """Parse a LongMemEval timestamp like ``2023/04/10 (Mon) 17:50``."""
    value = value.strip()
    m = _DATE_RE.search(value)
    if not m:
        raise ValueError(f"cannot parse date: {value!r}")
    year, month, day = (int(g) for g in m.groups())
    time_part = value[m.end():].strip()
    hour = minute = 0
    tm = re.search(r"(\d{1,2}):(\d{2})", time_part)
    if tm:
        hour, minute = int(tm.group(1)), int(tm.group(2))
    return datetime(year, month, day, hour, minute)


@dataclass(frozen=True)
class QuestionInstance:
    """One LongMemEval evaluation instance."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: datetime
    sessions: list[Session] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")


def load_longmemeval(path: str | Path) -> list[QuestionInstance]:
    """Load a LongMemEval JSON file into evaluation instances.

    Sessions are emitted in the file's order; the runner is responsible
    for chronological replay when the file is not pre-sorted (the oracle
    variant is explicitly unsorted).
    """
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"expected a JSON list, got {type(raw).__name__}")
    instances: list[QuestionInstance] = []
    for item in raw:
        session_ids = item["haystack_session_ids"]
        dates = item["haystack_dates"]
        if len(session_ids) != len(dates) or len(session_ids) != len(
            item["haystack_sessions"]
        ):
            raise ValueError(f"{item['question_id']}: mismatched haystack lengths")
        sessions: list[Session] = []
        for sid, sdate, turns in zip(
            session_ids, dates, item["haystack_sessions"], strict=True
        ):
            session_turns = [
                Turn(
                    role=t["role"],
                    content=t["content"],
                    has_answer=t.get("has_answer"),
                )
                for t in turns
            ]
            sessions.append(
                Session(
                    session_id=sid,
                    timestamp=parse_longmemeval_date(sdate),
                    turns=session_turns,
                )
            )
        instances.append(
            QuestionInstance(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=item["question"],
                answer=item["answer"],
                question_date=parse_longmemeval_date(item["question_date"]),
                sessions=sessions,
                answer_session_ids=item.get("answer_session_ids", []),
            )
        )
    return instances