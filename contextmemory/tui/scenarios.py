"""Deterministic hackathon demo scenarios for the TUI.

The scripted demo runs offline with no network and no API key: it drives the
C++ ETMC engine directly through structured cells, so the current-vs-historical
truth, provenance, abstention, and token-economy story are all visible and
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..core import PREFERENCE, WORLD, CellInput
from ..eval.protocol import Session, Turn

BASE = datetime(2024, 1, 1)


def ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


@dataclass
class DemoStep:
    """One scripted event: a fact update or a question."""

    label: str
    when: datetime
    session_id: str = ""
    turns: list[Turn] = field(default_factory=list)
    question: str = ""
    expected: str = ""

    @property
    def is_question(self) -> bool:
        return bool(self.question)


def build_scenario() -> list[DemoStep]:
    """The three-minute story: memory forms -> contradiction -> token economics
    -> abstention."""
    return [
        DemoStep(
            label="seed profile",
            when=BASE,
            session_id="s1",
            turns=[
                Turn(role="user",
                     content="I live in New York and work at Acme Corp."),
                Turn(role="user",
                     content="I usually use Vim and prefer concise answers."),
                Turn(role="user",
                     content="I am planning a hiking trip in Colorado."),
            ],
        ),
        DemoStep(
            label="current state",
            when=BASE + timedelta(days=7),
            question="Where do I live now?",
            expected="New York",
        ),
        DemoStep(
            label="profile",
            when=BASE + timedelta(days=7),
            question="What is my profession?",
            expected="Acme Corp",
        ),
        DemoStep(
            label="contradiction / update",
            when=BASE + timedelta(days=30),
            session_id="s2",
            turns=[
                Turn(role="user",
                     content="I moved to Seattle and joined Globex this month."),
            ],
        ),
        DemoStep(
            label="current truth after move",
            when=BASE + timedelta(days=37),
            question="Where do I live now?",
            expected="Seattle",
        ),
        DemoStep(
            label="historical truth",
            when=BASE + timedelta(days=37),
            question="Where did I live before moving?",
            expected="New York",
        ),
        DemoStep(
            label="stale fact is not current",
            when=BASE + timedelta(days=37),
            question="Who is my employer now?",
            expected="Globex",
        ),
        DemoStep(
            label="abstention",
            when=BASE + timedelta(days=37),
            question="What is my passport number?",
            expected=None,
        ),
    ]


def seed_cells() -> list[CellInput]:
    """Structured cells the scripted scenario ingests (no LLM needed)."""
    return [
        CellInput(
            text="User lives in New York",
            subject="user",
            predicate="location",
            object="New York",
            kind=WORLD,
            observed_at=ms(BASE),
            valid_from=ms(BASE),
            salience=0.9,
            entities=["user"],
            tags=["location", "home"],
        ),
        CellInput(
            text="User works at Acme Corp",
            subject="user",
            predicate="employer",
            object="Acme Corp",
            kind=WORLD,
            observed_at=ms(BASE),
            valid_from=ms(BASE),
            salience=0.9,
            entities=["user", "Acme Corp"],
            tags=["work", "employer"],
        ),
        CellInput(
            text="User prefers Vim and concise answers",
            subject="user",
            predicate="preference",
            object="Vim, concise",
            kind=PREFERENCE,
            observed_at=ms(BASE),
            valid_from=ms(BASE),
            salience=0.7,
            entities=["user"],
            tags=["tooling", "style"],
        ),
        CellInput(
            text="User is planning a hiking trip in Colorado",
            subject="user",
            predicate="plan",
            object="hiking trip in Colorado",
            kind=WORLD,
            observed_at=ms(BASE),
            valid_from=ms(BASE),
            salience=0.6,
            entities=["user", "Colorado"],
            tags=["travel", "hiking"],
        ),
    ]


def update_cells() -> list[CellInput]:
    """The contradiction: same subject/predicate, new object -> versioning."""
    when = BASE + timedelta(days=30)
    return [
        CellInput(
            text="User moved to Seattle and joined Globex",
            subject="user",
            predicate="location",
            object="Seattle",
            kind=WORLD,
            observed_at=ms(when),
            valid_from=ms(when),
            salience=0.95,
            entities=["user", "Seattle"],
            tags=["location", "move"],
        ),
        CellInput(
            text="User works at Globex",
            subject="user",
            predicate="employer",
            object="Globex",
            kind=WORLD,
            observed_at=ms(when),
            valid_from=ms(when),
            salience=0.95,
            entities=["user", "Globex"],
            tags=["work", "employer"],
        ),
    ]


def demo_session(step: DemoStep) -> Session | None:
    if not step.turns:
        return None
    return Session(
        session_id=step.session_id,
        timestamp=step.when,
        turns=step.turns,
    )


DEMO_TITLE = "ContextMemory — ETMC Demo"