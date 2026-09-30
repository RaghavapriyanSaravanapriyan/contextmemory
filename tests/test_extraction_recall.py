"""Regression tests for the write path's extraction recall.

These lock the defect measured on 2026-09-29: a single extraction pass
over a long session dropped the single most salient fact in it (a
verbatim "personal best time 27:12" out of a 12-turn LongMemEval
session), which then made the read path abstain. The fix is deterministic
windowing plus a recall-oriented prompt, and these tests pin both
properties: nothing is lost at a window boundary, and repeated facts
across windows are merged instead of duplicated.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

from contextmemory.engine.extractor import LLMExtractor
from contextmemory.eval.protocol import Session, Turn

TS = datetime(2023, 5, 12, 15, 0)


class _EchoExtractor:
    """Records prompts; emits one cell per number/date it can see."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, temperature=0.0, max_tokens=None,
                 json_mode=False) -> str:
        text = messages[0]["content"]
        self.prompts.append(text)
        found = re.findall(
            r"\b\d{1,2}:\d{2}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d+\s+(?:kg|miles|km)\b",
            text,
        )
        return json.dumps({"cells": [{"text": f"fact {f}"} for f in found]})


def _long_session(needle: str, *, filler_turns: int = 10,
                  filler_chars: int = 900) -> Session:
    turns = [Turn(role="user", content=f"message {i} " + "x" * filler_chars)
             for i in range(filler_turns)]
    turns.insert(len(turns) // 2, Turn(role="user", content=needle))
    return Session(session_id="s", timestamp=TS, turns=turns)


def test_long_session_is_windowed_not_truncated() -> None:
    ex = _EchoExtractor()
    extractor = LLMExtractor(ex, window_chars=8000)
    session = _long_session("my personal best time was 27:12")

    cells = extractor.extract(session)

    # More than one window: a single pass is what lost the fact.
    assert len(ex.prompts) > 1
    # The needle survives the split.
    assert any("27:12" in c.text for c in cells)
    # Lossless: every turn appears in exactly one window.
    joined = "\n".join(ex.prompts)
    assert all(f"message {i} " in joined for i in range(10))
    assert joined.count("27:12") == 1


def test_short_session_is_a_single_call() -> None:
    ex = _EchoExtractor()
    extractor = LLMExtractor(ex, window_chars=8000)
    extractor.extract(Session(
        session_id="s", timestamp=TS,
        turns=[Turn(role="user", content="I moved to Seattle.")],
    ))
    assert len(ex.prompts) == 1


def test_oversized_single_turn_still_extracted() -> None:
    """A pasted log must not vanish just because it exceeds the budget."""
    ex = _EchoExtractor()
    extractor = LLMExtractor(ex, window_chars=1000)
    extractor.extract(Session(
        session_id="s", timestamp=TS,
        turns=[Turn(role="user", content="kg " * 5000)],
    ))
    assert len(ex.prompts) == 1  # its own window, never skipped


class _Duplicating:
    def complete(self, messages, temperature=0.0, max_tokens=None,
                 json_mode=False) -> str:
        return json.dumps({"cells": [
            {"text": "The user lives in Seattle."},
            {"text": "the  user lives in  seattle!"},
            {"text": "The user runs 5k every morning."},
        ]})


def test_repeated_facts_across_windows_are_deduped() -> None:
    cells = LLMExtractor(_Duplicating()).extract(_long_session("noise"))
    texts = [c.text for c in cells]
    assert texts.count("The user lives in Seattle.") == 1
    assert len(texts) == 2
