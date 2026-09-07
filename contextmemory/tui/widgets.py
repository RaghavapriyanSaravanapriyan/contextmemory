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

from datetime import UTC, datetime

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
    """Timeline of validity windows.

    Render path never recalls: the app caches a snapshot (TTL) refreshed
    by the R key or the Brain view. Rendering stays instant at any scale.
    """

    def render(self) -> Table:
        table = Table(title="Timeline", expand=True)
        table.add_column("Cell")
        table.add_column("Subject/Predicate")
        table.add_column("Valid from")
        table.add_column("Valid until")
        table.add_column("Status")
        app = self.app  # type: ignore[attr-defined]
        hits = app.timeline_snapshot() if hasattr(app, "timeline_snapshot") else []
        for h in hits:
            vf = datetime.fromtimestamp(h.valid_from / 1000, UTC).strftime(
                "%Y-%m-%d"
            )
            vu = (
                datetime.fromtimestamp(h.valid_until / 1000, UTC).strftime(
                    "%Y-%m-%d"
                )
                if h.valid_until < 2**62
                else "open"
            )
            table.add_row(f"M{h.cell_id}", f"{h.subject}/{h.predicate}",
                          vf, vu, _status(h.status))
        return table


def _status(s: int) -> str:
    return {
        0: "active",
        1: "superseded",
        2: "expired",
        3: "forgotten",
        4: "disputed",
    }.get(s, "?")


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
    """Bench race: ContextMemory row measured live, baselines estimated.

    The ContextMemory tokens/ms come from the last real retrieval on this
    run. Baseline rows are order-of-magnitude estimates, not measured
    head-to-head runs — see benchmarks/ for real measured comparisons.
    """

    def render(self) -> Table:
        table = Table(title="Bench race (CM measured, baselines est.)",
                      expand=True)
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
            ("persist failures", eng.persist_failures),
            ("fallbacks used", eng.fallback_count),
        ]:
            table.add_row(metric, str(value))
        return table


class ProfilePaneHQ(Static):
    """HQ profile: durable static facts + recent dynamic activity."""

    def render(self) -> Table:
        table = Table(title="Profile", expand=True)
        table.add_column("Scope")
        table.add_column("Memory")
        app = self.app  # type: ignore[attr-defined]
        try:
            prof = app.client.profile(datetime.now())
        except Exception:
            table.add_row("-", "profile unavailable")
            return table
        if not prof.static_facts and not prof.dynamic_facts:
            table.add_row("-", "empty — type `remember: <fact>` in Brain, "
                               "or press R for the demo")
            return table
        for f in prof.static_facts[:10]:
            table.add_row("static", f"[M{f.cell_id}] {f.text}")
        for f in prof.dynamic_facts[:10]:
            table.add_row("recent", f"[M{f.cell_id}] {f.text}")
        return table


class SetupPane(Static):
    """HQ setup: current configuration + what to do next."""

    def render(self) -> str:
        app = self.app  # type: ignore[attr-defined]
        cfg = app.config
        try:
            from ..config import journal_path as _jp
            journal = str(_jp(app.container_tag))
        except Exception:
            journal = "?"
        try:
            cells = app.cell_count()
        except Exception:
            cells = "?"
        lines = [
            "[bold]SETUP[/bold]",
            "",
            f"  Building    {cfg.resolve_building_label()}",
            f"  Provider    {cfg.provider or 'offline'}",
            f"  Model       {app.live_model or cfg.model or '(automatic)'}",
            f"  Container   {app.container_tag}",
            f"  Journal     {journal}",
            f"  Cells       {cells}",
            "",
            "[dim]O reconnect model · C switch container · "
            "`contextmemory setup` in a terminal for the full wizard[/dim]",
        ]
        return "\n".join(lines)


class HelpPane(Static):
    """HQ help: every shortcut, one screen."""

    def render(self) -> str:
        return "\n".join([
            "[bold]HELP[/bold]",
            "",
            "  1-9      Brain · Timeline · Why · Models · Retrieval ·",
            "           Performance · Connections · Health · Profile",
            "  S        Setup (config, container, journal)",
            "  H        This help",
            "  O        Connect / rescan Ollama models",
            "  C        Switch memory container",
            "  R        Replay the offline demo story",
            "  TAB      Move between sidebar and content",
            "",
            "Brain input:",
            "  remember: <fact>   store a fact (works offline)",
            "  <question>          ask (offline recall, or model when live)",
            "",
            "Terminal twins:",
            "  contextmemory recall <q> --json   ranked hits, no model",
            "  contextmemory profile             static + dynamic facts",
            "  contextmemory setup               provider/model wizard",
        ])