<p align="center">
  <strong>ContextMemory</strong>
</p>

<p align="center">
  <strong>Give AI agents a memory that actually remembers.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> ·
  <a href="#install">Install</a> ·
  <a href="#test">Test</a> ·
  <a href="#benchmark">Benchmark</a> ·
  <a href="#how-memory-works">How it works</a> ·
  <a href="#the-api">API</a>
</p>

<p align="center">
  <strong>A C++ memory engine that recalls in microseconds, tracks how facts
  change over time, and tells the truth when it doesn't know.</strong><br/>
  <strong>~0.1ms reads · 1 LLM call per conversation · zero external services</strong>
</p>

---

Your AI forgets everything between conversations. Ask it "what did I say last
week about my trip to Japan?" and it has no idea. ContextMemory fixes that: it
stores what matters, updates facts when they change, forgets what's stale, and
hands back the right context in microseconds — with a real C++ engine that
runs entirely on your laptop.

| | |
|---|---|
| ⚡ **Fast** | The core is C++, not Python. Measured answer reads are **~0.1ms** — the industry bar is ~300ms. |
| 🕰️ **Knows time** | Facts carry a "when was this true" timeline. Move from New York to Seattle and it remembers *both* — and always answers with the current one. |
| 🎯 **Writes precisely** | Tests score what every other memory system fakes: **write precision** (only stores what was really said) and **evolution** (updates when facts change). |
| 🚫 **Tells the truth** | No fabricated cells. When memory doesn't cover the question, it says "I don't have enough information." |
| 🧩 **Any model** | Bring a frontier API or a free local model. Works with anything OpenAI-compatible, including **Ollama** on a machine with no GPU. |

---

## Quickstart

One command takes you from a bare machine to the running brain:

```bash
git clone https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git
cd contextmemory
```

Then run the launcher for your OS — it checks every dependency, installs
whatever is missing, builds the C++ engine, and launches the TUI:

| OS | One command |
|---|---|
| macOS / Linux | `./run.sh` |
| Windows (cmd.exe) | `run.bat` |
| Windows (PowerShell) | `.\run.ps1` |
| Any OS | `python run.py` |

Once installed, `contextmemory` opens the TUI directly:

```bash
contextmemory              # onboarding → dashboard
contextmemory --live       # connect to / auto-launch Ollama
```

The first run walks a short onboarding — **what are you building?** →
**connect your AI** (it scans your local Ollama models automatically) →
**MEMORY ONLINE** → the dashboard. Returning runs go straight to the
dashboard.

**No model, no API key, no database, no cloud** for the offline demo; add
`--live` and your local model for the full experience.

### Live demo with your model

Works with **any model Ollama serves** — qwen, llama, gemma, or anything you've
pulled. Add `--live` to connect to / auto-launch Ollama, `--model` to pick a
model up front, `--url` for a custom endpoint:

macOS / Linux:

```bash
./run.sh                                   # offline demo, zero config
./run.sh --live                            # connect to / auto-launch Ollama
./run.sh --live --model qwen3:4b           # pick the model up front
./run.sh --live --model qwen3:4b --url http://localhost:11434
```

Windows (cmd.exe):

```cmd
run.bat
run.bat --live
run.bat --live --model qwen3:4b
run.bat --live --model qwen3:4b --url http://localhost:11434
```

Inside the TUI, press `O` to open the **Connect to Ollama** screen — endpoint,
API key, and model are all editable right there, and the TUI launches
`ollama serve` itself if nothing is listening:

```text
Connect to Ollama
  Endpoint   http://localhost:11434
  API key    ollama
  [Scan models]  [Launch ollama serve]
  → pick a model from the list
```

`remember: <fact>` stores a fact through LLM extraction; ask questions and the
brain answers with citations (`[M2]`), routing, and token telemetry. Other
keys: `1` Brain · `2` Timeline · `3` Why · `B` Bench race · `H` Health ·
`R` Replay · `Q` Quit.

---

## Install

Requirements: **Python 3.11+ and an internet connection** — everything else is
fetched automatically. The one-command flow above is the supported path; here
is what it does under the hood:

1. **Python 3.11+** — `uv` fetches a managed interpreter if your system one is
   too old.
2. **`uv`** — installed via the official one-liner (the only external tool).
3. **C++ compiler** — Linux: `g++` via your package manager; macOS: Xcode
   Command Line Tools; Windows: MSVC Build Tools via `winget`.
4. **CMake + Ninja** — installed as wheels during the build (no system install).
5. **Ollama** — only for `--live` mode.
6. **Python deps + C++ core** — `uv sync` builds the engine.

For AI-tool integrations, `contextmemory mcp` runs the MCP bridge (stdio) and
exposes `memory` / `recall` / `context` / `forget` to any MCP client — Claude
Code, Cursor, OpenCode, VS Code, Cline. Config lives in
`~/.config/contextmemory/config.json` (first-run detection + your choices).

Or install the pieces explicitly:

```bash
uv sync --extra dev          # installs deps and builds the C++ core
```

Everything is a library you import, a CLI you call, and a C++ core you can
benchmark. No server to start, no Redis, no container.

---

## Test

One command runs the whole deterministic suite — C++ core tests, Python tests,
lint, and the microsecond latency bench. No model, no API key:

```bash
./scripts/test-all.sh
```

(Windows: run this inside Git Bash or WSL, or run `uv run pytest` and
`uv run ruff check contextmemory tests` directly in PowerShell.)

What it runs:

```text
C++ core      → 11 test suites / 49 checks   (capture, reconcile, versioning,
                                               projections, packing, persistence)
Python        → 53 tests                     (wrapper, eval harness, CLI, API)
Lint          → ruff (clean)
Latency bench → ingest p50 ~0.05ms · answer p50 ~0.13ms  (200 sessions)
```

---

## Benchmark

The same rig that measures us also scores any system and any model you want.
Three commands, one shared harness:

| Command | What it measures | Why it matters |
|---|---|---|
| `contextmemory bench` | Ingest/answer latency, no reader | Microseconds vs 300ms+ SOTA |
| `contextmemory dims` | Write precision, evolution, forgetting | The dimensions nobody else scores |
| `contextmemory eval` | Answer accuracy on LongMemEval | Head-to-head vs Mem0, Zep, Supermemory |

### With your own Ollama model

Point the harness at any model you have pulled:

```bash
uv run contextmemory dims \
  --system contextmemory \
  --reader-api-base http://localhost:11434 \
  --reader-api-key ollama --reader-model qwen3:4b
```

`dims` replays 13 probes with known ground truth and scores **write precision,
evolution, and forgetting** — the behaviors that decide whether a memory layer
is trustworthy.

`eval` replays LongMemEval (500 instances — start with a slice):

```bash
uv run contextmemory eval \
  --data benchmarks/data/longmemeval_oracle.json \
  --system contextmemory --max-instances 10 \
  --reader-api-base http://localhost:11434 \
  --reader-api-key ollama --reader-model qwen3:4b \
  --out reports/runs/run-qwen34b.jsonl
```

CPU-only tip: a 4B model takes roughly 15-25s per extraction on CPU, and every
session ingested plus every probe answered costs one call — budget ~10-20
minutes for `dims`, and keep `--max-instances` small for `eval`. See your
models with `ollama list`; pull one with `ollama pull qwen3:4b`. Small local
models occasionally produce verbose or imperfect extractions; a capable model
or a GPU makes runs dramatically cleaner and faster.

First measured run (qwen3:4b, CPU-only, no GPU), via `contextmemory dims`:

```text
write-precision  1.0000 (5 probes)   only stores what was really said
evolution        0.8000 (5 probes)   updates when facts change
forgetting       1.0000 (3 probes)   stale facts expire and abstain
```

---

## The API

```python
from contextmemory.api import MemoryClient
from contextmemory.engine.extractor import LLMExtractor
from contextmemory.engine.ollama import OllamaManager
from contextmemory.eval.protocol import Session, Turn
from datetime import datetime

# Any OpenAI-compatible reader — Ollama, vLLM, LM Studio, a frontier API.
reader = OllamaManager().reader("qwen3:4b")   # native /api/chat, thinking off

brain = MemoryClient("user_123", extractor=LLMExtractor(reader))

# Store a conversation (write path: capture + 1 LLM extraction call)
brain.session(Session(
    session_id="conv-1",
    timestamp=datetime.now(),
    turns=[Turn(role="user", content="I moved to Seattle and joined Globex.")],
))

# Ask — recall evidence, pack it, generate a cited answer
answer, report = brain.ask("Where does the user live?", reader)
print(answer)              # "The user lives in Seattle [M5]."
print(report.tokens)       # 17
```

Everything else is a thin read path: `search(query)`, `recall(query)`,
`profile()` (static facts + recent context), `projection(subject, predicate)`
(the current value of any fact, with its version history). Memory is scoped by
**container tag**, so one brain can serve many users, repos, or clients.

---

## How memory works

```
conversation / tool trace
        │
  CAPTURE        → immutable Episode recorded, no LLM, <1ms
        │
  EXTRACT        → one LLM pass → compact structured cells
        │          (subject / predicate / object / when-true)
        │
  RECONCILE      → deterministic C++ core: dedup, version, project.
        │          "I moved to Seattle" supersedes "I live in NYC" —
        │          it closes the old cell and opens the new one
        │
  QUERY COMPILE  → question → bounded plan, NO read-path LLM
        │
  EVIDENCE PACK  → minimum-sufficient context under a hard token budget
        │
  ANSWER         → your model, given only that evidence; cites memory ids,
                   abstains when evidence is insufficient
```

Three layers, each with one job:

- **C++ core (`core/`)** — the ETMC engine. Episodes are immutable; cells are
  bi-temporal (when they became true, when they stopped being true); a query
  compiles into a bounded plan and searches without an LLM. No external
  databases, no network — just fast code. This is where the microseconds come
  from.
- **Python layer (`contextmemory/`)** — the write path's single LLM call and
  everything around it: extraction, embedding, orchestration, the
  `MemoryClient` API, and the TUI. Model-agnostic by design.
- **Evaluation rig (`contextmemory/eval/`)** — replays real benchmark datasets
  through any memory system and scores it. You can't beat what you can't
  measure, so we measure everything — including ourselves.

Why it's trustworthy: the write path prefers **no fact over a guessed fact**;
contradictions are resolved deterministically and *both* versions are kept with
their validity windows; and the system scores itself on **write precision**
(did it store only what was said?), **evolution** (did it update when facts
changed?), and **forgetting** (did stale facts expire?) — the dimensions no
public benchmark measures.

---

## Repository map

```text
core/                          C++ ETMC memory engine (the fast part)
  include/cmcore/              cells, episodes, projections, store, indexes
  src/                         capture / reconcile / compile / search / pack
  python/bindings.cpp          nanobind surface (_core extension)
  tests/test_core.cpp          dependency-free C++ test suite (11 suites)
contextmemory/                 the Python package
  core.py                      typed facade over _core
  config.py                    first-run detection + persisted app config
  observability.py             real retrieval events + rolling performance
  mcp.py                       MCP bridge (memory/recall/context/forget)
  engine/                      extractor, embedder, ETMC orchestration, Ollama
  eval/                        benchmark rig (replay, dimensions, latency)
  tui/                         onboarding + dashboard (Textual)
  api.py, cli.py               MemoryClient API + contextmemory CLI
tests/                         Python test suite (68 tests)
run.py                         one-command installer + launcher (all OSes)
run.sh                         macOS/Linux entry point (thin wrapper)
run.bat, run.ps1               Windows entry points (cmd.exe / PowerShell)
scripts/test-all.sh            one command: full deterministic suite (no model)
benchmarks/data/               LongMemEval datasets (oracle + cleaned)
reports/                       research + architecture decisions
tasks/active/                  what we're working on
```

## Status

- ✅ Engine: ETMC core built (capture, reconcile, projections, query compiler,
  evidence packing, persistence)
- ✅ Measurement rig: benchmark + latency + custom dimensions
- ✅ TUI: onboarding → dashboard (Brain, Timeline, Why, Models, Retrieval
  Live, Performance, Connections, Health)
- ✅ MCP bridge: `contextmemory mcp` → memory/recall/context/forget
- ✅ One-command flows: install (`./run.sh`), open (`contextmemory`),
  test (`./scripts/test-all.sh`), benchmark (`contextmemory bench | dims | eval`)
- 🔄 Pushing LongMemEval accuracy; Mem0/Zep head-to-head adapters next

**Mission and roadmap:** `tasks/active/T001-beat-frontier-memory-layers.md`

**SOTA architecture and implementation plan:**
`docs/architecture/2026-08-29-sota-memory-brain-action-plan.md`

---

*You can't beat what you can't measure. So we measure everything.*