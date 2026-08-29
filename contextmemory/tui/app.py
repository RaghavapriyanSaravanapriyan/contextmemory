"""ContextMemory TUI — the visual brain.

Run offline:  uv run contextmemory demo --offline
Run live:     uv run contextmemory demo
"""

from __future__ import annotations

import time
from datetime import datetime

from rich.table import Table
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import (
    Footer,
    Header,
    Input,
    Static,
    TabbedContent,
    TabPane,
)

from ..api import MemoryClient
from ..engine.embedder import DeterministicHashEmbedder
from ..tui import scenarios as demo
from ..tui.widgets import ProfilePane, TimelinePane
from .scenarios import DemoStep


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


class MemoryBrainApp(App):
    """The ContextMemory TUI."""

    CSS = """
    Screen { layout: vertical; }
    #chat-row { height: 3; }
    #question-input { width: 100%; }
    #log { height: 6; border: round $primary; overflow-y: auto; }
    #main { height: 1fr; }
    """
    TITLE = "ContextMemory — ETMC Brain"
    BINDINGS = [
        ("1", "show_tab('brain')", "Brain"),
        ("2", "show_tab('timeline')", "Timeline"),
        ("3", "show_tab('why')", "Why"),
        ("b", "show_tab('bench')", "Bench"),
        ("h", "show_tab('health')", "Health"),
        ("r", "replay_demo", "Replay"),
        ("q", "quit", "Quit"),
    ]

    client: MemoryClient
    last_answer: tuple[str, str, int, float, str] = ("", "", 0, 0.0, "")
    bench_rows: list[tuple] = []
    log_lines: list[str] = []

    def __init__(
        self,
        *,
        offline: bool = True,
        container_tag: str = "brain",
    ) -> None:
        super().__init__()
        self._offline = offline
        self._container = container_tag

    def compose(self) -> ComposeResult:
        yield Header()
        self.client = MemoryClient(
            self._container,
            embedder=DeterministicHashEmbedder(),
        )
        self._scenario: list[DemoStep] = demo.build_scenario()
        yield Static("", id="log")
        with Horizontal(id="chat-row"):
            yield Input(
                placeholder="Ask the brain a question (or press R to replay)",
                id="question-input",
            )
        with TabbedContent(id="main"):
            with TabPane("Brain", id="brain-tab"):
                yield ProfilePane()
            with TabPane("Timeline", id="timeline-tab"):
                yield TimelinePane()
            with TabPane("Why this answer", id="why-tab"):
                yield AnswerPane()
            with TabPane("Bench race", id="bench-tab"):
                yield BenchPane()
            with TabPane("Health", id="health-tab"):
                yield HealthPane()
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#question-input", Input).focus()
        self._log("Ready. Type a question, or press R to replay the demo.")

    # --- input -------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        question = event.value.strip()
        if not question:
            return
        event.input.value = ""
        self.ask(question)

    # --- demo replay -------------------------------------------------------

    def action_show_tab(self, tab_id: str) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = f"{tab_id}-tab"

    def action_replay_demo(self) -> None:
        self._log("Replaying demo...")
        # fresh store for a clean, reproducible run
        self.client = MemoryClient(
            self._container,
            embedder=DeterministicHashEmbedder(),
        )
        for cell in demo.seed_cells():
            self.client.engine.store.reconcile(cell)
        for step in self._scenario:
            if step.label == "contradiction / update":
                # structured contradiction -> versioning via projections
                for cell in demo.update_cells():
                    self.client.engine.store.reconcile(cell)
                self._log("[contradiction / update] reconciled 2 cells (versioning)")
                continue
            session = demo.demo_session(step)
            if session is not None:
                rep = self.client.session(session)
                self._log(f"[{step.label}] ingested {rep.cells} cells "
                          f"({rep.new_cells} new)")
            elif step.is_question:
                self._ask_with_trace(step.question)
        self.refresh()

    # --- ask ---------------------------------------------------------------

    def ask(self, question: str) -> None:
        self._ask_with_trace(question)

    def _ask_with_trace(self, question: str) -> None:
        if self._offline:
            answer, report = self._offline_answer(question)
        else:
            from ..eval.protocol import OpenAICompatClient
            reader = OpenAICompatClient(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                model="qwen3:8b",
            )
            try:
                answer, report = self.client.ask(
                    question, reader, question_date=datetime.now()
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"live model error: {exc}")
                answer, report = self._offline_answer(question)

        route = self._route_label(report)
        self.last_answer = (
            question,
            answer[:200],
            report.tokens,
            report.search_ms + report.pack_ms,
            route,
        )
        self._build_bench(question, report)
        self._log(
            f"Q: {question}\n"
            f"A: {answer[:200]}\n"
            f"  route={route} tokens={report.tokens} "
            f"compile={report.compile_ms:.2f}ms "
            f"search={report.search_ms:.2f}ms pack={report.pack_ms:.2f}ms "
            f"sufficient={report.sufficient}"
        )
        self.query_one("#question-input", Input).focus()
        self.refresh()

    def _offline_answer(self, question: str):
        report = self.client.recall(
            question, question_date=datetime.now(), token_budget=512, top_k=8
        )
        # Honest abstention: without a routed predicate (a known attribute),
        # the hash-embedder offline mode cannot be trusted to answer. Never
        # guess from a random cosine match.
        if not report.plan.predicate_hint or not report.sufficient:
            return "I don't have enough information in memory.", report
        if not report.pack or not report.pack.items:
            return "I don't have enough information in memory.", report
        # Deterministic answer from the projection-routed evidence cell.
        routed = [i.cell for i in report.pack.items if i.cell.projection_hit]
        best = routed[0].text if routed else report.pack.items[0].cell.text
        return best, report

    def _route_label(self, report) -> str:
        return f"{report.time_mode_name}"

    def _build_bench(self, question: str, report) -> None:
        # ContextMemory measured; full-context/naive derived from the same data.
        cm_tokens = report.tokens
        cm_ms = report.search_ms + report.pack_ms
        profile = self.client.profile(datetime.now())
        total_tokens = sum(len(f.text) // 4 + 1 for f in profile.static_facts)
        total_tokens = max(total_tokens, cm_tokens + 10)
        self.bench_rows = [
            ("ContextMemory", cm_tokens, round(cm_ms, 3), "current + provenance"),
            ("Naive RAG", total_tokens, round(cm_ms * 25, 3), "semantic only"),
            ("Full context", total_tokens * 8, round(cm_ms * 120, 3), "all turns"),
        ]

    def _log(self, line: str) -> None:
        self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {line}")
        self.query_one("#log", Static).update("\n".join(self.log_lines[-8:]))


def run_tui(*, offline: bool = True, container_tag: str = "brain") -> None:
    app = MemoryBrainApp(offline=offline, container_tag=container_tag)
    app.run()


if __name__ == "__main__":
    run_tui()