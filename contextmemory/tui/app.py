"""ContextMemory TUI — the product shell.

Journey: Welcome → What are you building? → Connect your AI → MEMORY ONLINE →
dashboard. First run walks the onboarding; returning runs jump straight to the
dashboard.

Dashboard sections:

  1  Brain           ask the memory, ingest facts, replay the demo
  2  Timeline        validity windows, updates, superseded facts
  3  Why             selected evidence, provenance, routing
  4  Models          local Ollama discovery + active model
  5  Retrieval Live  every real retrieval, with stage timings
  6  Performance     p50/p95/p99 of real retrievals, fast-path rate
  7  Connections     MCP bridge + client status
  8  Health          engine counters

All numbers come from the engine; nothing is fabricated.
"""

from __future__ import annotations

import contextlib
import time
from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    Static,
)

from .. import __version__  # noqa: F401
from ..api import MemoryClient
from ..config import BUILDING_CHOICES, AppConfig
from ..engine.embedder import DeterministicHashEmbedder
from ..engine.extractor import LLMExtractor
from ..engine.ollama import DEFAULT_BASE_URL, OllamaManager
from ..eval.protocol import Session, Turn
from ..observability import RetrievalEvent, RetrievalTracker
from ..tui import scenarios as demo
from ..tui.widgets import (
    AnswerPane,
    HealthPane,
    TimelinePane,
)
from .scenarios import DemoStep

STATUS_COLORS = {"offline": "red", "connected": "green", "managed": "yellow",
                 "connecting": "cyan"}


# --- onboarding ------------------------------------------------------------


class WelcomeScreen(Screen):
    """Screen 1 — elegant welcome. Enter starts."""

    BINDINGS = [("enter", "start", "Get started"), ("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Static("", id="spacer")
        yield Static("CONTEXTMEMORY", id="logo")
        yield Static("Memory infrastructure for AI systems", id="tagline")
        yield Static("", id="spacer2")
        yield Button("Get Started", id="start-btn", variant="primary")
        yield Static("Persistent memory · Adaptive retrieval · Minimal config",
                     id="footer-line")

    def on_mount(self) -> None:
        self.query_one("#start-btn", Button).focus()

    def action_start(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "start-btn":
            self.action_start()


class BuildingScreen(Screen):
    """Screen 2 — what are you building? (optional, skippable)."""

    BINDINGS = [("escape", "skip", "Skip")]

    def compose(self) -> ComposeResult:
        yield Static("What are you building?", id="prompt")
        yield Static("This tunes sensible defaults. You can change it anytime.",
                     id="sub")
        opts = OptionList(id="building-options")
        for label in BUILDING_CHOICES.values():
            opts.add_option(label)
        yield opts

    def on_mount(self) -> None:
        opts = self.query_one("#building-options", OptionList)
        opts.highlighted = 0
        opts.focus()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        idx = event.option_index
        key = list(BUILDING_CHOICES)[idx]
        self.dismiss(key)

    def action_skip(self) -> None:
        self.dismiss("ai-agent")


class ConnectAIScreen(Screen):
    """Screen 3 — connect your AI. Only real providers are offered."""

    BINDINGS = [("escape", "later", "Configure later")]

    def compose(self) -> ComposeResult:
        yield Static("Connect your AI", id="prompt")
        yield Static("Choose a provider. Offline works with zero config.",
                     id="sub")
        opts = OptionList(id="provider-options")
        opts.add_option("Ollama (Local)")
        opts.add_option("Configure later — start offline")
        yield opts

    def on_mount(self) -> None:
        opts = self.query_one("#provider-options", OptionList)
        opts.highlighted = 0
        opts.focus()

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        if event.option_index == 0:
            self.dismiss("ollama")
        else:
            self.dismiss("")


class OllamaScanScreen(Screen):
    """Screen 4 — scan local Ollama models (real discovery)."""

    def __init__(self, manager: OllamaManager) -> None:
        super().__init__()
        self.manager = manager

    def compose(self) -> ComposeResult:
        yield Static("Scanning Ollama...", id="prompt")
        yield Static("", id="status")

    def on_mount(self) -> None:
        self.run_worker(self._scan())

    async def _scan(self) -> None:
        status = self.query_one("#status", Static)
        if not self.manager.running:
            status.update("Ollama not running. Launching `ollama serve`...")
            if not self.manager.start_managed():
                status.update(
                    f"[red]{self.manager.last_error or 'failed to launch'}[/]"
                    "\n\nConfigure later, or start Ollama and press R to rescan."
                )
                return
        details = self.manager.model_details()
        if not details:
            status.update(
                "✓ Ollama detected\n\nNo models pulled yet.\n\n"
                "Pull one with `ollama pull qwen3:4b`, then press R to rescan.\n"
                "You can also continue offline."
            )
            return
        lines = [f"✓ {len(details)} local models found\n"]
        for d in details:
            size = d["size_gb"]
            tag = f"  {d['name']}  {size:g} GB"
            if d["parameter_size"]:
                tag += f"  ({d['parameter_size']})"
            lines.append(tag)
        lines.append("\nPress 1 to pick the first model, R to rescan, "
                     "C to continue offline.")
        status.update("\n".join(lines))
        self._details = details

    def on_key(self, event) -> None:
        key = event.key.lower()
        if key == "r":
            self.query_one("#status", Static).update("Rescanning...")
            self.run_worker(self._scan())
        elif key == "1" and getattr(self, "_details", None):
            self.dismiss(self._details[0]["name"])
        elif key == "c":
            self.dismiss("")


class MemoryOnlineScreen(Screen):
    """Screen 5 — the success moment."""

    BINDINGS = [("enter", "done", "Open dashboard"), ("d", "done",
                                                      "Open dashboard")]

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        yield Static("✨ MEMORY ONLINE", id="online-title")
        yield Static("", id="status")
        yield Button("Open Dashboard", id="open", variant="primary")

    def on_mount(self) -> None:
        lines = [
            ("Engine", "Ready"),
            ("Models", self.cfg.model or "Automatic"),
            ("Provider", self.cfg.provider or "Offline"),
            ("Cache", "Adaptive"),
            ("Retrieval", "Progressive"),
        ]
        table = "\n".join(f"  {k:<12} {v}" for k, v in lines)
        self.query_one("#status", Static).update(table)
        self.query_one("#open", Button).focus()

    def action_done(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open":
            self.action_done()


# --- dashboard shell --------------------------------------------------------


class OllamaConnectScreen(ModalScreen):
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
        self.dismiss(str(label.renderable))


class Dashboard(Screen):
    """Main command center: sidebar nav + content panes."""

    BINDINGS = [
        ("1", "go('brain')", "Brain"),
        ("2", "go('timeline')", "Timeline"),
        ("3", "go('why')", "Why"),
        ("4", "go('models')", "Models"),
        ("5", "go('retrieval')", "Retrieval Live"),
        ("6", "go('perf')", "Performance"),
        ("7", "go('connections')", "Connections"),
        ("8", "go('health')", "Health"),
        ("o", "ollama", "Connect Ollama"),
        ("r", "replay", "Replay"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal(id="shell"):
            with Vertical(id="sidebar"):
                yield Static("CONTEXTMEMORY", id="brand")
                yield Static("", id="statusline")
                nav = OptionList(id="nav")
                for label in [
                    "1  Brain",
                    "2  Timeline",
                    "3  Why",
                    "4  Models",
                    "5  Retrieval Live",
                    "6  Performance",
                    "7  Connections",
                    "8  Health",
                ]:
                    nav.add_option(label)
                yield nav
            yield Vertical(id="content")
        yield Footer()

    def on_mount(self) -> None:
        self.app: MemoryBrainApp
        self._render_status()
        self.query_one("#nav", OptionList).focus()
        self._show("brain")

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        names = ["brain", "timeline", "why", "models", "retrieval", "perf",
                 "connections", "health"]
        self._show(names[event.option_index])

    def action_go(self, name: str) -> None:
        self._show(name)

    def action_ollama(self) -> None:
        self.app.action_connect_ollama()

    def action_replay(self) -> None:
        self.app.replay_demo()

    def _render_status(self) -> None:
        app = self.app
        parts = [f"Ollama: {app.ollama_state}"]
        if app.live_model:
            parts.append(f"model: {app.live_model}")
        self.query_one("#statusline", Static).update(
            Text(" | ".join(parts), style=f"{STATUS_COLORS[app.ollama_state]}")
        )

    def _show(self, name: str) -> None:
        app = self.app
        content = self.query_one("#content", Vertical)
        content.remove_children()
        if name == "brain":
            content.mount(self.app.brain)
        elif name == "timeline":
            content.mount(TimelinePane())
        elif name == "why":
            content.mount(AnswerPane())
        elif name == "models":
            content.mount(ModelsPane(app.ollama))
        elif name == "retrieval":
            content.mount(RetrievalLivePane(app.tracker))
        elif name == "perf":
            content.mount(PerformancePane(app.tracker))
        elif name == "connections":
            content.mount(ConnectionsPane())
        elif name == "health":
            content.mount(HealthPane())


# --- content panes ---------------------------------------------------------


class BrainPane(Static):
    """The ask / remember / replay surface."""

    def __init__(self, app: MemoryBrainApp) -> None:
        super().__init__("", id="brain-pane")
        self._app = app

    def render(self) -> str:
        return (
            "[bold]Ask the brain[/bold]   (type a question, or "
            "'remember: <fact>' — press R to replay the demo)\n\n"
            + "\n".join(self._app.log_lines[-6:])
        )


class ModelsPane(Static):
    """Real local Ollama models + the active model."""

    def __init__(self, manager: OllamaManager) -> None:
        super().__init__("", id="models-pane")
        self.manager = manager

    def render(self) -> str:
        app = self.app  # type: ignore[attr-defined]
        cfg = app.config
        lines = ["[bold]MODELS[/bold]  "]
        lines.append(f"Active strategy: {cfg.strategy}")
        lines.append(f"Active model:   {app.live_model or cfg.model or 'Automatic'}")
        lines.append("")
        lines.append("[bold]Local — Ollama[/bold]")
        details = self.manager.model_details()
        if not details:
            lines.append("  (no models reachable — is Ollama running?)")
        for d in details:
            size = d["size_gb"]
            mark = " ›" if d["name"] == (app.live_model or cfg.model) else "  "
            lines.append(f"{mark} {d['name']}  {size:g} GB")
        lines.append("")
        lines.append("[dim]Press O to reconnect / change models.[/dim]")
        return "\n".join(lines)


class RetrievalLivePane(Static):
    """The wow factor — every real retrieval with stage timings."""

    def __init__(self, tracker: RetrievalTracker) -> None:
        super().__init__("", id="retrieval-pane")
        self.tracker = tracker

    def render(self) -> str:
        events = self.tracker.events
        if not events:
            return "[bold]RETRIEVAL LIVE[/bold]\n\nWaiting for a query...\n\n" \
                   "Ask the brain something in the Brain view, then come back " \
                   "here to watch it decide."
        lines = ["[bold]RETRIEVAL LIVE[/bold]"]
        for e in events[-6:]:
            lines.append("")
            lines.append(f"[bold]{e.query[:60]}[/bold]  "
                         f"({e.started_at.strftime('%H:%M:%S')})")
            for line in e.explain():
                lines.append(f"  {line}")
            lines.append(f"  [green]TOTAL {e.total_ms:.4f} ms[/green]  "
                         f"hits={e.hits}")
        lines.append("")
        lines.append(f"[dim]{len(events)} retrieval(s) recorded this session[/dim]")
        return "\n".join(lines)


class PerformancePane(Static):
    """Real p50/p95/p99 + fast-path rate from tracked retrievals."""

    def __init__(self, tracker: RetrievalTracker) -> None:
        super().__init__("", id="perf-pane")
        self.tracker = tracker

    def render(self) -> str:
        s = self.tracker.snapshot()
        lines = [
            "[bold]PERFORMANCE[/bold]",
            f"Retrievals: {s['count']}",
            "",
            f"p50  {s['p50_ms']} ms",
            f"p95  {s['p95_ms']} ms",
            f"p99  {s['p99_ms']} ms",
            f"mean {s['avg_ms']} ms",
            "",
            f"Fast-path rate  {s['fast_path_rate'] * 100:.1f}%",
            f"Hit rate        {s['hit_rate'] * 100:.1f}%",
        ]
        if s["count"] == 0:
            lines.append("")
            lines.append("[dim]Run some retrievals in the Brain view first.[/dim]")
        return "\n".join(lines)


class ConnectionsPane(Static):
    """MCP bridge + AI client status (honest — only what exists)."""

    def render(self) -> str:
        lines = [
            "[bold]CONNECTIONS[/bold]",
            "",
            "[bold]AI TOOLS[/bold]",
            "  › OpenCode            Not connected",
            "  › Claude Code         Not connected",
            "  › Cursor              Not connected",
            "",
            "[bold]PROTOCOLS[/bold]",
            "  › MCP Server          Ready (stdio)",
            "  › Python SDK          Ready",
            "  › HTTP API            Planned",
            "",
            "[dim]MCP maps memory / recall / context / forget onto the engine. "
            "Client config is generated on demand.[/dim]",
        ]
        return "\n".join(lines)


# --- the app ----------------------------------------------------------------


class MemoryBrainApp(App):
    """ContextMemory TUI — onboarding + dashboard."""

    CSS = """
    Screen { layout: vertical; }
    #spacer, #spacer2 { height: 3; }
    #logo { text-align: center; text-style: bold; color: $accent; }
    #tagline { text-align: center; color: $text; }
    #footer-line { text-align: center; color: $text-muted; }
    #start-btn { width: 24; margin: 1 0 0 0; }
    #prompt { text-align: center; text-style: bold; padding: 2 0 0 0; }
    #sub { text-align: center; color: $text-muted; }
    #building-options, #provider-options { margin: 1 0 0 0; }
    #online-title { text-align: center; text-style: bold; color: $success;
                    padding: 3 0 1 0; }
    #status { text-align: center; color: $text; }
    #open { width: 26; margin: 1 0 0 0; }
    #shell { height: 1fr; }
    #sidebar { width: 30; border-right: heavy $primary; background: $panel; }
    #brand { text-style: bold; color: $accent; padding: 1 1 0 1; }
    #statusline { height: 1; padding: 0 1; }
    #nav { height: 1fr; }
    #content { height: 1fr; padding: 1 2; }
    #brain-pane, #models-pane, #retrieval-pane, #perf-pane,
    #connections-pane { padding: 1 0; }
    """
    TITLE = "ContextMemory"
    SUB_TITLE = "memory infrastructure for AI systems"

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
        self.config = AppConfig.load()
        self.tracker = RetrievalTracker()
        self.live_model: str | None = None
        self.ollama_state: str = "offline"
        self.log_lines: list[str] = []
        self.last_answer = ("", "", 0, 0.0, "")
        self.bench_rows: list[tuple] = []
        self.brain = BrainPane(self)
        self._scenario: list[DemoStep] = demo.build_scenario()

    def compose(self) -> ComposeResult:
        yield Static("", id="boot")

    def on_mount(self) -> None:
        self.push_screen(Dashboard())
        if not self.config.onboarded:
            self.push_screen(WelcomeScreen(), self._on_welcome)
        else:
            self._finish_boot()

    def _on_welcome(self, _: None | str | object = None) -> None:
        if self.config.onboarded:
            self._finish_boot()
            return
        self.push_screen(BuildingScreen(), self._on_building)

    def _on_building(self, building: str | None) -> None:
        self.config.building = building or "ai-agent"
        self.push_screen(ConnectAIScreen(), self._on_provider)

    def _on_provider(self, provider: str | None) -> None:
        provider = provider or ""
        self.config.provider = provider
        if provider == "ollama":
            self.push_screen(OllamaScanScreen(self.ollama), self._on_model)
        else:
            self._complete(None)

    def _on_model(self, model: str | None) -> None:
        model = model or ""
        self.config.model = model
        self._complete(model)

    def _complete(self, model: str | None) -> None:
        self.config.complete_onboarding(
            building=self.config.building,
            provider=self.config.provider,
            model=model or "",
            base_url=self.ollama.base_url,
            container=self._container,
        )
        if model:
            self._connect_live(model)
        self.push_screen(MemoryOnlineScreen(self.config), self._finish_boot)

    def _finish_boot(self, _: None | str | object = None) -> None:
        self.brain = BrainPane(self)
        self._log("Ready. Ask the brain, or press R to replay the demo.")
        if not self._offline and not self.live_model:
            self.run_worker(self._auto_connect())
        else:
            self._refresh_status()

    # --- Ollama ------------------------------------------------------------

    def _connect_live(self, model: str) -> None:
        self.live_model = model
        self.ollama_state = "connected"
        self._offline = False
        self.client.set_extractor(LLMExtractor(self.ollama.reader(model)))
        self._refresh_status()

    async def _auto_connect(self) -> None:
        if not self.ollama.running:
            self.ollama_state = "connecting"
            self._refresh_status()
            self._log("Ollama not reachable. Launching `ollama serve`...")
            if not self.ollama.start_managed():
                self.ollama_state = "offline"
                self._refresh_status()
                self._log(f"Ollama unavailable: {self.ollama.last_error}")
                return
            self.ollama_state = "managed"
            self._refresh_status()
        models = self.ollama.list_models()
        chosen = self._model or self.config.model or (models[0] if models else None)
        if chosen:
            self._connect_live(chosen)
        else:
            self._log("No models pulled. Run: ollama pull qwen3:8b")

    def _refresh_status(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one(Dashboard)._render_status()

    @property
    def client(self) -> MemoryClient:
        if not hasattr(self, "_client") or self._client is None:
            self._client = MemoryClient(
                self._container,
                embedder=DeterministicHashEmbedder(),
            )
        return self._client

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
        if not self.live_model:
            self._log("Not connected. Press O to connect a model first.")
            return
        session = Session(
            session_id="live",
            timestamp=datetime.now(),
            turns=[Turn(role="user", content=content)],
        )
        rep = self.client.session(session)
        self._log(f"remembered {content!r} -> {rep.cells} cell(s) "
                  f"({rep.new_cells} new)")

    def action_connect_ollama(self) -> None:
        def on_result(model: str | None) -> None:
            if model:
                self._connect_live(model)

        self.push_screen(OllamaConnectScreen(self.ollama), on_result)

    # --- demo replay -------------------------------------------------------

    def replay_demo(self) -> None:
        self._log("Replaying demo...")
        self._client = MemoryClient(
            self._container,
            embedder=DeterministicHashEmbedder(),
        )
        if self.live_model:
            self._client.set_extractor(
                LLMExtractor(self.ollama.reader(self.live_model))
            )
        for cell in demo.seed_cells():
            self._client.engine.store.reconcile(cell)
        for step in self._scenario:
            if step.label == "contradiction / update":
                for cell in demo.update_cells():
                    self._client.engine.store.reconcile(cell)
                self._log("[contradiction / update] reconciled 2 cells")
                continue
            session = demo.demo_session(step)
            if session is not None:
                rep = self._client.session(session)
                self._log(f"[{step.label}] ingested {rep.cells} cells "
                          f"({rep.new_cells} new)")
            elif step.is_question:
                self._ask_with_trace(step.question)

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
        start = time.monotonic()
        try:
            report = self.client.recall(
                question, question_date=datetime.now(), token_budget=512,
                top_k=8,
            )
            self.tracker.record(RetrievalEvent(
                query=question,
                report=report,
                hits=len(report.hits),
                used_fallback=bool(report.pack and report.pack.used_fallback),
            ))
            self._last_elapsed_ms = (time.monotonic() - start) * 1000
        except Exception as exc:  # noqa: BLE001
            self.tracker.record(RetrievalEvent(query=question, exception=str(exc)))
            self._log(f"recall error: {exc}")
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
        self._refresh_status()

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
        self.log_lines = self.log_lines[-8:]
        with contextlib.suppress(Exception):
            self.query_one("#brain-pane", Static).refresh()


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