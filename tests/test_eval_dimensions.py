"""Tests for the custom-dimensions harness."""

from __future__ import annotations

from datetime import datetime

from contextmemory.eval import (
    Scenario,
    Session,
    default_scenarios,
    evolution_scenario,
    forgetting_scenario,
    is_abstention,
    run_dimensions,
    run_scenario,
    write_precision_scenario,
)

from .conftest import FakeReader


class _ReaderSystem:
    """A memory system that answers from a FakeReader's behaviors."""

    def __init__(self, reader: FakeReader) -> None:
        self._reader = reader
        self._seen: list[str] = []

    def ingest(self, session: Session) -> None:
        self._seen.append(session.session_id)

    def answer(self, question: str, question_date: datetime) -> str:
        return self._reader.complete([{"role": "user", "content": question}])


def test_default_scenarios_cover_three_dimensions() -> None:
    scenarios = default_scenarios()
    assert {s.dimension for s in scenarios} == {
        "write-precision",
        "evolution",
        "forgetting",
    }


def test_is_abstention_detects_declines() -> None:
    assert is_abstention("I don't have enough information to answer that.")
    assert is_abstention("The transcript does not mention a pet.")
    assert is_abstention("Unknown.")
    assert not is_abstention("The pet's name is Rex.")
    assert not is_abstention("")


def test_evolution_scenario_probes_are_dated() -> None:
    scenario = evolution_scenario()
    assert all(p.question_date >= datetime(2024, 6, 1) for p in scenario.probes)
    assert any(p.expected == "Globex" for p in scenario.probes)
    assert any(p.expected == "Acme Corp" for p in scenario.probes)


def test_write_precision_has_abstention_probes() -> None:
    scenario = write_precision_scenario()
    assert any(p.is_abstention_probe for p in scenario.probes)
    assert any(not p.is_abstention_probe for p in scenario.probes)


def test_run_scenario_scores_correct_and_incorrect() -> None:
    reader = FakeReader(
        behaviors={
            "city did the user move": "Austin",
            "favorite hobby": "trail running",
            "pet's name": "I don't know.",
            "graduate college": "The transcript does not say.",
            "favorite food": "not mentioned",
        }
    )
    result = run_scenario(write_precision_scenario(), lambda: _ReaderSystem(reader))
    assert result.overall == 1.0
    assert all(p.correct for p in result.probe_results)


def test_run_scenario_counts_fabrication_as_failure() -> None:
    reader = FakeReader(
        behaviors={
            "city did the user move": "Austin",
            "favorite hobby": "trail running",
            "pet's name": "Rex",
            "graduate college": "2018",
            "favorite food": "pizza",
        }
    )
    result = run_scenario(write_precision_scenario(), lambda: _ReaderSystem(reader))
    # The three abstention probes are answered with fabricated facts: fail.
    assert result.overall == 2 / 5


def test_evolution_staleness_is_detected() -> None:
    behaviors = {"work": "Acme Corp", "job title": "software engineer"}
    reader = FakeReader(behaviors=behaviors)
    result = run_scenario(evolution_scenario(), lambda: _ReaderSystem(reader))
    # A system stuck on the old employer fails the "current" probes.
    assert not result.probe_results[0].correct
    assert result.probe_results[2].correct  # historical probe still matches


def test_forgetting_current_city_checked() -> None:
    scenario = forgetting_scenario()
    assert any(p.expected == "Chicago" for p in scenario.probes)
    assert any(p.expected == "Boston" for p in scenario.probes)
    assert len(scenario.sessions) >= 10  # long noisy timeline


def test_run_dimensions_aggregates_per_dimension() -> None:
    reader = FakeReader(default="unknown")
    reports = run_dimensions(default_scenarios(), lambda: _ReaderSystem(reader))
    by_dim = {r.dimension: r for r in reports}
    assert set(by_dim) == {"write-precision", "evolution", "forgetting"}
    total_probes = sum(r.n_probes for r in reports)
    assert total_probes == sum(len(s.probes) for s in default_scenarios())


def test_empty_scenario_scores_zero() -> None:
    scenario = Scenario(
        name="empty",
        dimension="write-precision",
        sessions=[],
        probes=[],
    )
    result = run_scenario(scenario, lambda: _ReaderSystem(FakeReader()))
    assert result.overall == 0.0


def test_turn_construction_is_user_only() -> None:
    scenario = write_precision_scenario()
    assert all(t.role == "user" for s in scenario.sessions for t in s.turns)