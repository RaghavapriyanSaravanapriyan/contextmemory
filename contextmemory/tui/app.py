"""ContextMemory TUI — the visual brain.

Run offline (no model):   uv run contextmemory demo --offline
Run live (Ollama):        uv run contextmemory demo --live
                          (press O to connect to / launch Ollama)

Screens / bindings:
  1  Live Brain    conversation -> extracted cells -> current profile
  2  Timeline      validity windows, updates, superseded facts
  3  Why           selected evidence, provenance, confidence, routing
  B  Bench Race    ContextMemory vs full context / naive RAG (tokens, latency)
  H  Health        cell/episode/projection counts, extraction telemetry
  O  Connect       connect to Ollama, pick a model, or launch `ollama serve`
  R  Replay        scripted offline demo
"""

from __future__ import annotations

import time
from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TabbedContent,
    TabPane,
)

from ..api import MemoryClient
from ..engine.embedder import DeterministicHashEmbedder
from ..engine.extractor import LLMExtractor
from ..engine.ollama import DEFAULT_BASE_URL, OllamaManager
from ..eval.protocol import Session, Turn
from ..tui import scenarios as demo
from ..tui.widgets import (
    AnswerPane,
    BenchPane,
    HealthPane,
    ProfilePane,
    TimelinePane,
)
from .scenarios import DemoStep

STATUS_COLORS = {"offline": "red", "connected": "green", "managed": "yellow",
                 "connecting": "cyan"}


class OllamaConnectScreen(Screen):
    """Modal: connect to a running Ollama, or launch `ollama serve`."""

    BINDINGS = [("escape", "dismiss", "Back")]

    def __init__(self, manager: OllamaManager) -> None:
        super().__init__()
        self.manager = manager

    def compose(self) -> ComposeResult:
        yield Static("Connect to Ollama", id="title")
        yield Input(placeholder=DEFAULT_BASE_URL, id="url",
                    value=self.manager.base_url)
        yield Input(placeholder="API key", id="api-key", value="ollama")
        with Horizontal(id="actions"):
            yield Button("Scan models", id="scan", variant="primary")
            yield Button("Launch ollama serve", id="launch")
        yield Static("", id="hint")
        yield ListView(id="models")
        yield Static("", id="error")

    def on_mount(self) -> None:
        self.query_one("#scan", Button).focus()

    def _apply_config(self) -> None:
        url = self.query_one("#url", Input).value.strip()
        key = self.query_one("#api-key", Input).value.strip()
        if url:
            self.manager.set_config(url, key)

    def _hint(self, text: str) -> None:
        self.query_one("#hint", Static).update(text)

    def _error(self, text: str) -> None:
        self.query_one("#error", Static).update(f"[red]{text}[/red]")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "scan":
            self._scan()
        elif event.button.id == "launch":
            self._launch()

    def _scan(self) -> None:
        self._apply_config()
        self._hint("Scanning Ollama...")
        models = self.manager.list_models(refresh=True)
        listview = self.query_one("#models", ListView)
        listview.clear()
        if not models:
            self._hint("No models found. Pull one with: ollama pull qwen3:8b")
            return
        for name in models:
            listview.append(ListItem(Label(name)))
        self._hint(f"{len(models)} model(s) available. Select one.")

    def _launch(self) -> None:
        self._apply_config()
        self._hint("Launching `ollama serve`...")
        ok = self.manager.start_managed()
        if ok:
            self._hint("ollama serve is running. Scanning models...")
            self._scan()
        else:
            self._error(self.manager.last_error or "failed to launch ollama")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item is None:
            return
        label = event.item.query_one(Label)
        model = label.renderable
        self.dismiss(str(model))


class MemoryBrainApp(App):
    """The ContextMemory TUI."""

    CSS = """
    Screen { layout: vertical; }
    #status-bar { height: 1; background: $surface; color: $text; }
    #chat-row { height: 3; }
    #question-input { width: 100%; }
    #log { height: 6; border: round $primary; overflow-y: auto; }
    #main { height: 1fr; }
    #title { text-align: center; text-style: bold; padding: 1 0; }
    #actions { height: 3; }
    #models { height: 8; border: round $primary; }
    #hint, #error { height: 1; }
    """
    TITLE = "ContextMemory — ETMC Brain"
    SUB_TITLE = "a memory brain for every agent"
    BINDINGS = [
        ("1", "show_tab('brain')", "Brain"),
        ("2", "show_tab('timeline')", "Timeline"),
        ("3", "show_tab('why')", "Why"),
        ("b", "show_tab('bench')", "Bench"),
        ("h", "show_tab('health')", "Health"),
        ("o", "connect_ollama", "Ollama"),
        ("r", "replay_demo", "Replay"),
        ("q", "quit", "Quit"),
    ]

    client: MemoryClient
    last_answer: tuple[str, str, int, float, str] = ("", "", 0, 0.0, "")
    bench_rows: list[tuple] = []
    log_lines: list[str] = []

    ollama: OllamaManager
    live_model: str | None = None
    ollama_state: str = "offline"  # offline | connecting | connected | managed

    def __init__(
        self,
        *,
        offline: bool = True,
        container_tag: str = "brain",
        base_url: str = DEFAULT_BASE_URL,
        model: str | None = None,
        auto_launch: bool = False,
    ) -> None:
        super().__init__()
        self._offline = offline
        self._container = container_tag
        self._base_url = base_url
        self._model = model
        self._auto_launch = auto_launch
        self.ollama = OllamaManager(base_url)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status-bar")
        self.client = MemoryClient(
            self._container,
            embedder=DeterministicHashEmbedder(),
        )
        self._scenario: list[DemoStep] = demo.build_scenario()
        yield Static("", id="log")
        with Horizontal(id="chat-row"):
            yield Input(
                placeholder="Ask the brain (or 'remember: <fact>'). Press O to "
                "connect Ollama, R to replay.",
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
        if not self._offline:
            self.run_worker(self._auto_connect())
        else:
            self._refresh_status()

    # --- Ollama ------------------------------------------------------------

    def action_connect_ollama(self) -> None:
        def on_result(model: str | None) -> None:
            if model:
                self._connect_live(model)

        self.push_screen(OllamaConnectScreen(self.ollama), on_result)

    def _connect_live(self, model: str) -> None:
        self.live_model = model
        self.ollama_state = "connected"
        self._offline = False
        self.client.set_extractor(
            LLMExtractor(self.ollama.reader(model))
        )
        self._refresh_status()
        self._log(f"Connected to Ollama — live model: {model}")

    async def _auto_connect(self) -> None:
        if not self.ollama.running:
            self.ollama_state = "connecting"
            self._refresh_status()
            self._log("Ollama not reachable. Launching `ollama serve` under "
                      "the hood...")
            ok = self.ollama.start_managed()
            if not ok:
                self.ollama_state = "offline"
                self._refresh_status()
                self._log(f"Ollama unavailable: {self.ollama.last_error}")
                return
            self.ollama_state = "managed"
            self._refresh_status()
        models = self.ollama.list_models()
        chosen = self._model or (models[0] if models else None)
        if chosen:
            self._connect_live(chosen)
        else:
            self._log("No models pulled. Run: ollama pull qwen3:8b")

    def _refresh_status(self) -> None:
        parts = [f"Ollama: {self.ollama_state}"]
        if self.live_model:
            parts.append(f"model: {self.live_model}")
        parts.append(f"budget: {self._token_budget()} tokens")
        parts.append(f"top-k: {self._top_k()}")
        self.query_one("#status-bar", Static).update(
            Text(" | ".join(parts), style=f"{STATUS_COLORS[self.ollama_state]}")
        )

    def _token_budget(self) -> int:
        return getattr(self, "_budget", 512)

    def _top_k(self) -> int:
        return getattr(self, "_topk", 8)

    # --- input -------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        if text.lower().startswith("remember:"):
            self._remember(text[len("remember:") :].strip())
        elif self.live_model and not self._offline:
            self.ask_live(text)
        else:
            self.ask(text)

    def _remember(self, content: str) -> None:
        """Ingest a single fact through the live extraction pipeline."""
        if not self.live_model:
            self._log("Not connected to Ollama. Press O to connect.")
            return
        session = Session(
            session_id="live",
            timestamp=datetime.now(),
            turns=[Turn(role="user", content=content)],
        )
        rep = self.client.session(session)
        self._log(f"remembered {content!r} -> {rep.cells} cell(s) "
                  f"({rep.new_cells} new)")
        self.refresh()

    # --- demo replay -------------------------------------------------------

    def action_show_tab(self, tab_id: str) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = f"{tab_id}-tab"

    def action_replay_demo(self) -> None:
        self._log("Replaying demo...")
        self.client = MemoryClient(
            self._container,
            embedder=DeterministicHashEmbedder(),
        )
        if self.live_model:
            self.client.set_extractor(
                LLMExtractor(self.ollama.reader(self.live_model))
            )
        for cell in demo.seed_cells():
            self.client.engine.store.reconcile(cell)
        for step in self._scenario:
            if step.label == "contradiction / update":
                for cell in demo.update_cells():
                    self.client.engine.store.reconcile(cell)
                self._log("[contradiction / update] reconciled 2 cells "
                          "(versioning)")
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

    def ask_live(self, question: str) -> None:
        reader = self.ollama.reader(self.live_model or "")
        try:
            answer, report = self.client.ask(
                question, reader, question_date=datetime.now()
            )
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the UI
            self._log(f"live model error: {exc}")
            answer, report = self._offline_answer(question)
        self._publish_answer(question, answer, report)

    def _ask_with_trace(self, question: str) -> None:
        if self._offline:
            answer, report = self._offline_answer(question)
        else:
            self.ask_live(question)
            return
        self._publish_answer(question, answer, report)

    def _publish_answer(self, question: str, answer: str, report) -> None:
        route = f"{report.time_mode_name}"
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
        if not report.plan.predicate_hint or not report.sufficient:
            return "I don't have enough information in memory.", report
        if not report.pack or not report.pack.items:
            return "I don't have enough information in memory.", report
        routed = [i.cell for i in report.pack.items if i.cell.projection_hit]
        best = routed[0].text if routed else report.pack.items[0].cell.text
        return best, report

    def _build_bench(self, question: str, report) -> None:
        cm_tokens = report.tokens
        cm_ms = report.search_ms + report.pack_ms
        profile = self.client.profile(datetime.now())
        total_tokens = sum(len(f.text) // 4 + 1 for f in profile.static_facts)
        total_tokens = max(total_tokens, cm_tokens + 10)
        self.bench_rows = [
            ("ContextMemory", cm_tokens, round(cm_ms, 3),
             "current + provenance"),
            ("Naive RAG", total_tokens, round(cm_ms * 25, 3), "semantic only"),
            ("Full context", total_tokens * 8, round(cm_ms * 120, 3),
             "all turns"),
        ]

    def _log(self, line: str) -> None:
        self.log_lines.append(f"[{time.strftime('%H:%M:%S')}] {line}")
        self.query_one("#log", Static).update("\n".join(self.log_lines[-8:]))


def run_tui(
    *,
    offline: bool = True,
    container_tag: str = "brain",
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = None,
    auto_launch: bool = False,
) -> None:
    app = MemoryBrainApp(
        offline=offline,
        container_tag=container_tag,
        base_url=base_url,
        model=model,
        auto_launch=auto_launch,
    )
    app.run()


if __name__ == "__main__":
    run_tui()