"""Custom-dimension scenarios the public benchmarks do not measure.

LongMemEval and friends measure end-to-end answer accuracy on retrieval. The
dimensions below are the ones ContextMemory exists to win on, and which the
field is widely reported to fail:

* ``write-precision`` -- does the write path store what was actually said,
  and does the read path abstain rather than fabricate when asked about
  something never discussed?
* ``evolution`` -- do facts correctly track updates and contradictions over
  time (current value vs historical value, no staleness)?
* ``forgetting`` -- do stable core facts survive consolidation, while
  superseded facts stop contaminating current answers?

Each dimension is a synthetic, fully-controlled timeline with known ground
truth, so scores are reproducible without a benchmark dataset. Probes are
scored deterministically (``deterministic_match`` for recall-style probes,
``is_abstention`` for abstention probes); this is a development proxy, not an
LLM judge.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from time import perf_counter

from .protocol import MemorySystem, Session, Turn
from .scoring import deterministic_match

_ABSTENTION_MARKERS = (
    "not mention",
    "not discuss",
    "not provide",
    "not say",
    "not state",
    "no mention",
    "no information",
    "no record",
    "not enough",
    "dont have enough",
    "do not know",
    "does not know",
    "did not know",
    "dont know",
    "doesnt know",
    "cant answer",
    "cannot answer",
    "unable",
    "unknown",
    "not available",
    "insufficient",
    "not in the transcript",
    "not in the history",
    "not covered",
    "never mentioned",
    "no such",
    "there is no",
    "there are no",
    "doesnt have",
    "does not have",
    "dont have",
    "i have no",
    "not included",
    "not given",
    "not stated",
    "does not say",
    "doesnt say",
    "not say",
    "not mentioned anywhere",
)


def is_abstention(hypothesis: str) -> bool:
    """True if a response declines to answer rather than asserting a fact.

    A curated marker list applied to normalized text (lowercased, whitespace
    collapsed, apostrophes stripped). This is a deterministic proxy for the
    official LongMemEval abstention prompt; it is intentionally conservative
    (a fabricated answer contains none of these markers).
    """
    hyp = hypothesis.lower()
    hyp = hyp.replace("'", "")
    hyp = re.sub(r"\s+", " ", hyp).strip()
    if not hyp:
        return False
    return any(marker in hyp for marker in _ABSTENTION_MARKERS)


@dataclass(frozen=True)
class Probe:
    """A question asked against a scenario at a specific date."""

    question: str
    question_date: datetime
    expected: str | None = None
    scoring: str = "match"  # "match" (recall) | "abstain" (no fabrication)

    @property
    def is_abstention_probe(self) -> bool:
        return self.scoring == "abstain"


@dataclass(frozen=True)
class Scenario:
    """A synthetic timeline plus the probes that probe it."""

    name: str
    dimension: str
    sessions: list[Session]
    probes: list[Probe] = field(default_factory=list)


@dataclass
class ProbeResult:
    probe: Probe
    hypothesis: str
    correct: bool
    answer_s: float = 0.0


@dataclass
class ScenarioResult:
    scenario: Scenario
    probe_results: list[ProbeResult]
    ingest_s: float = 0.0

    @property
    def overall(self) -> float:
        if not self.probe_results:
            return 0.0
        return sum(r.correct for r in self.probe_results) / len(self.probe_results)

    @property
    def n_probes(self) -> int:
        return len(self.probe_results)


@dataclass
class DimensionReport:
    dimension: str
    scenario_results: list[ScenarioResult]
    n_probes: int = 0

    @property
    def overall(self) -> float:
        if not self.n_probes:
            return 0.0
        total = sum(
            sum(r.correct for r in s.probe_results) for s in self.scenario_results
        )
        return total / self.n_probes


def _session(sid: str, dt: datetime, *lines: str) -> Session:
    turns = [
        Turn(role=role, content=content)
        for role, content in (("user", line) for line in lines)
    ]
    return Session(session_id=sid, timestamp=dt, turns=turns)


def write_precision_scenario() -> Scenario:
    """Stated facts must be stored; unstated facts must not be fabricated."""
    sessions = [
        _session(
            "w1",
            datetime(2024, 1, 5),
            "I'm moving to Austin next month. My new office is downtown.",
        ),
        _session(
            "w2",
            datetime(2024, 2, 1),
            "I arrived in Austin today. The weather here is great.",
        ),
        _session(
            "w3",
            datetime(2024, 2, 10),
            "My favorite hobby is trail running.",
        ),
    ]
    probes = [
        Probe(
            "What city did the user move to?",
            datetime(2024, 3, 1),
            expected="Austin",
        ),
        Probe(
            "What is the user's favorite hobby?",
            datetime(2024, 3, 1),
            expected="trail running",
        ),
        Probe(
            "What is the user's pet's name?",
            datetime(2024, 3, 1),
            scoring="abstain",
        ),
        Probe(
            "What year did the user graduate college?",
            datetime(2024, 3, 1),
            scoring="abstain",
        ),
        Probe(
            "What is the user's favorite food?",
            datetime(2024, 3, 1),
            scoring="abstain",
        ),
    ]
    return Scenario(
        name="write-precision",
        dimension="write-precision",
        sessions=sessions,
        probes=probes,
    )


def evolution_scenario() -> Scenario:
    """Facts must evolve: the current value updates, history is preserved."""
    sessions = [
        _session(
            "e1",
            datetime(2024, 1, 10),
            "I work at Acme Corp as a software engineer.",
        ),
        _session(
            "e2",
            datetime(2024, 4, 20),
            "I got promoted to senior software engineer at Acme.",
        ),
        _session(
            "e3",
            datetime(2024, 7, 1),
            "I quit Acme. I now work at Globex as a product manager.",
        ),
        _session(
            "e4",
            datetime(2024, 8, 15),
            "Actually, my title at Globex is senior product manager.",
        ),
    ]
    probes = [
        Probe(
            "Where does the user currently work?",
            datetime(2024, 9, 1),
            expected="Globex",
        ),
        Probe(
            "What is the user's current job title?",
            datetime(2024, 9, 1),
            expected="senior product manager",
        ),
        Probe(
            "Where did the user work before joining Globex?",
            datetime(2024, 9, 1),
            expected="Acme Corp",
        ),
        Probe(
            "What was the user's job title at Acme?",
            datetime(2024, 9, 1),
            expected="software engineer",
        ),
        Probe(
            "Where does the user work?",
            datetime(2024, 6, 1),
            expected="Acme Corp",
        ),
    ]
    return Scenario(
        name="evolution",
        dimension="evolution",
        sessions=sessions,
        probes=probes,
    )


def forgetting_scenario() -> Scenario:
    """Core facts survive; superseded facts stop contaminating current state."""
    noise = [
        "The forecast calls for sun tomorrow.",
        "I finished reading a great novel about the sea.",
        "My neighbor got a new car, a blue sedan.",
        "I tried a new coffee shop downtown.",
        "The construction on Main Street finished last week.",
        "I signed up for a ceramics class on weekends.",
        "My laptop battery barely lasts an hour now.",
    ]
    base = datetime(2024, 1, 1)
    sessions = [
        _session(
            "f1",
            base,
            "My name is Sarah. I live in Boston.",
        ),
    ]
    for i, line in enumerate(noise, start=2):
        sessions.append(
            _session(f"f{i}", base + timedelta(days=7 * i), line)
        )
    sessions.extend(
        [
            _session(
                "f_mid",
                base + timedelta(days=90),
                "I moved to Chicago last week. Loving the lake.",
            ),
            _session(
                "f_noise2",
                base + timedelta(days=95),
                "The office moved to a new building.",
            ),
            _session(
                "f_noise3",
                base + timedelta(days=100),
                "I started volunteering at an animal shelter.",
            ),
        ]
    )
    probes = [
        Probe(
            "What is the user's name?",
            base + timedelta(days=120),
            expected="Sarah",
        ),
        Probe(
            "What city does the user currently live in?",
            base + timedelta(days=120),
            expected="Chicago",
        ),
        Probe(
            "What city did the user previously live in?",
            base + timedelta(days=120),
            expected="Boston",
        ),
    ]
    return Scenario(
        name="forgetting",
        dimension="forgetting",
        sessions=sessions,
        probes=probes,
    )


def default_scenarios() -> list[Scenario]:
    return [
        write_precision_scenario(),
        evolution_scenario(),
        forgetting_scenario(),
    ]


def run_scenario(
    scenario: Scenario,
    system_factory: Callable[[], MemorySystem],
) -> ScenarioResult:
    """Replay a scenario through a memory system and score every probe."""
    system = system_factory()
    ordered = sorted(scenario.sessions, key=lambda s: s.timestamp)
    start = perf_counter()
    for session in ordered:
        system.ingest(session)
    ingest_s = perf_counter() - start

    probe_results: list[ProbeResult] = []
    for probe in scenario.probes:
        t0 = perf_counter()
        hypothesis = system.answer(probe.question, probe.question_date)
        answer_s = perf_counter() - t0
        if probe.scoring == "abstain":
            correct = is_abstention(hypothesis)
        else:
            correct = deterministic_match(hypothesis, probe.expected or "")
        probe_results.append(
            ProbeResult(
                probe=probe,
                hypothesis=hypothesis,
                correct=correct,
                answer_s=answer_s,
            )
        )
    return ScenarioResult(
        scenario=scenario,
        probe_results=probe_results,
        ingest_s=ingest_s,
    )


def run_dimensions(
    scenarios: list[Scenario],
    system_factory: Callable[[], MemorySystem],
) -> list[DimensionReport]:
    """Run scenarios grouped by dimension and aggregate per-dimension scores."""
    results = [run_scenario(s, system_factory) for s in scenarios]
    by_dim: dict[str, list[ScenarioResult]] = {}
    for result in results:
        by_dim.setdefault(result.scenario.dimension, []).append(result)
    reports = []
    for dimension, scenario_results in by_dim.items():
        n_probes = sum(s.n_probes for s in scenario_results)
        reports.append(
            DimensionReport(
                dimension=dimension,
                scenario_results=scenario_results,
                n_probes=n_probes,
            )
        )
    return sorted(reports, key=lambda r: r.dimension)