"""ContextMemory TUI — the visual brain.

Screens:
  1  Live Brain   conversation -> extracted cells -> current profile
  2  Timeline     validity windows, updates, superseded facts
  3  Why          selected evidence, provenance, confidence, routing
  B  Bench Race   ContextMemory vs full context / naive RAG (tokens, latency)
  H  Health       cell/episode/projection counts, extraction telemetry
  R  Replay       scripted offline demo
"""

from __future__ import annotations

from datetime import datetime

from rich.table import Table
from textual.widgets import (
    Static,
)

SECONDS_PER_DAY = 86_400


class ProfilePane(Static):
    """Current-state profile (static + dynamic)."""

    def render(self) -> Table:
        table = Table(title="Current profile", expand=True)
        table.add_column("Memory", no_wrap=False)
        table.add_column("Kind")
        table.add_column("Confidence")
        client = self.app.client  # type: ignore[attr-defined]
        prof = client.profile(datetime.now())
        for f in prof.static_facts[:8]:
            table.add_row(f.text, "static", f"{f.confidence:.2f}")
        return table


class TimelinePane(Static):
    """Timeline of validity windows."""

    def render(self) -> Table:
        table = Table(title="Timeline", expand=True)
        table.add_column("Cell")
        table.add_column("Subject/Predicate")
        table.add_column("Valid from")
        table.add_column("Valid until")
        table.add_column("Status")
        client = self.app.client  # type: ignore[attr-defined]
        # Walk projections -> version chains via search historical
        q = client.recall(
            "what changed over time",
            question_date=datetime.now(),
            top_k=16,
        )
        for h in q.hits:
            vf = datetime.utcfromtimestamp(h.valid_from / 1000).strftime(
                "%Y-%m-%d"
            )
            vu = (
                datetime.utcfromtimestamp(h.valid_until / 1000).strftime(
                    "%Y-%m-%d"
                )
                if h.valid_until < 2**62
                else "open"
            )
            table.add_row(f"M{h.cell_id}", f"{h.subject}/{h.predicate}",
                          vf, vu, _status(h.status))
        return table


def _status(s: int) -> str:
    return {0: "active", 1: "superseded", 2: "expired"}.get(s, "?")


class AnswerPane(Static):
    """Renders the last answer with evidence and routing trace."""

    def render(self) -> Table:
        table = Table(title="Why this answer", expand=True)
        table.add_column("Question")
        table.add_column("Answer")
        table.add_column("Tokens")
        table.add_column("Retrieval ms")
        table.add_column("Route")
        q, answer, tokens, ms_, route = self.app.last_answer  # type: ignore[attr-defined]
        table.add_row(q, answer, str(tokens), f"{ms_:.2f}", route)
        return table


class BenchPane(Static):
    """ContextMemory vs full context vs naive RAG — measured, not faked."""

    def render(self) -> Table:
        table = Table(title="Bench race (measured on this run)", expand=True)
        table.add_column("System")
        table.add_column("Retrieved tokens")
        table.add_column("Retrieval ms")
        table.add_column("Evidence")
        for row in self.app.bench_rows:  # type: ignore[attr-defined]
            table.add_row(*[str(x) for x in row])
        return table


class HealthPane(Static):
    """Memory health counters from the engine."""

    def render(self) -> Table:
        table = Table(title="Memory health", expand=True)
        table.add_column("Metric")
        table.add_column("Value")
        store = self.app.client.engine.store  # type: ignore[attr-defined]
        eng = self.app.client.engine  # type: ignore[attr-defined]
        for metric, value in [
            ("cells", store.cell_count),
            ("episodes", store.episode_count),
            ("projections", store.projection_count),
            ("edges", store.edge_count),
            ("entities", store.entity_count),
            ("extract failures", eng.extract_failures),
            ("fallbacks used", eng.fallback_count),
        ]:
            table.add_row(metric, str(value))
        return table