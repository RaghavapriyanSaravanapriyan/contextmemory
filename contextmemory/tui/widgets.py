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