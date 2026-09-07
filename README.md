<div align="center">

<img width="1600" height="533" alt="image" src="https://github.com/user-attachments/assets/67ad9bce-94a9-43c3-97d4-f26d3d9231c3" />
                                                                                                                                
### Give your local model a memory that does not lie.

**A fast, temporal memory layer for Ollama, MCP agents, and AI applications.**

<br />

![Python](https://img.shields.io/badge/python-3.11%2B-8b9cff?style=flat-square)
![C++](https://img.shields.io/badge/core-C%2B%2B20-65e6b0?style=flat-square)
![Ollama](https://img.shields.io/badge/ollama-local-ffb86b?style=flat-square)
![MCP](https://img.shields.io/badge/MCP-ready-c9a7ff?style=flat-square)

<br />

*Your model is brilliant, fast, and forgetful.*

*ContextMemory is the quiet brain beside it.*

</div>

---

## Install

Requirements: Python 3.11+, a C++ compiler, CMake, Ninja. Ollama is
optional (needed only for live-model answers; memory itself works offline).

```bash
git clone https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git
cd contextmemory
uv sync
uv run contextmemory setup
```

`setup` is an interactive wizard: what you're building → provider
(Ollama local / offline) → model (auto-detected from `ollama serve`) →
memory container. Re-run anytime; existing values become defaults.
`contextmemory setup --show` prints the current config.

Verify the install:

```bash
./scripts/verify.sh   # pytest + ruff (C++ suite: see below)
```

Windows note: `verify.sh` needs Git Bash. Native PowerShell equivalent:

```powershell
uv run pytest
uv run ruff check contextmemory tests
```

The C++ core compiles on MSVC 2022, GCC, and Clang (warning dialects are
per-toolchain in `core/CMakeLists.txt`). If a stale `build/` cache ever
fights you, delete it and re-run `uv sync`.

## One-command benchmark (any machine, any OS)

One command downloads everything, installs it, and runs every benchmark
**serially with the same model** — ContextMemory fully first, then
Supermemory, then the baselines — streaming progress live and writing a
portable Markdown report. Windows, macOS, Linux — only git and Python
3.11+ required:

```bash
# Linux / macOS
git clone https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git \
  && cd contextmemory \
  && python3 scripts/cmbench.py --model qwen3:4b --with-supermemory
```

```powershell
# Windows (PowerShell)
git clone https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git;
  cd contextmemory; py scripts/cmbench.py --model qwen3:4b --with-supermemory
```

```bash
# The full gauntlet instead of the default ~10% fast subsets
python scripts/cmbench.py --model qwen3:4b --with-supermemory \
  --suites all --systems contextmemory,supermemory,full-history \
  --full --judge --yes
```

What the command downloads (official sources only, URLs pinned in
`scripts/cmbench.py` and verified):

| Download | Official source |
| --- | --- |
| LongMemEval oracle + S | `huggingface.co/datasets/xiaowu0162/longmemeval-cleaned` (upstream README) |
| LoCoMo (10 convos) | `github.com/snap-research/locomo` (`data/locomo10.json`) |
| BEAM-100K | `huggingface.co/datasets/Mohammadta/BEAM` (CC-BY-SA-4.0) |
| Supermemory reference | `github.com/supermemoryai/supermemory` (+ official `supermemory` SDK pip) |
| ContextMemory | this repo (C++ core builds locally) |

What the command runs (5 phases, all visible live):

1. **ENV** — installs ContextMemory incl. the C++ core build.
2. **MODEL** — probes your reader (Ollama default), pulls `--model` on request.
3. **DATA** — fetches the table above with progress (skipped when present).
4. **RUN** — fixed order: `contextmemory` fully first, then `supermemory`,
   then baselines. `dims` + `bench` run **every** system separately;
   official suites share one replay runner and one judge.
5. **REPORT** — `reports/runs/cmbench-<ts>/` with `REPORT.md` (paste it into
   a PR), `summary.json`, and per-run logs.

| Flag | Job |
| --- | --- |
| `--model / --base-url / --api-key` | the one reader for ALL systems |
| `--systems` | lineup (default `contextmemory,full-history`; `supermemory` auto-joins when ready) |
| `--suites` | `dims,bench,longmemeval,locomo,beam` or `all` |
| `--fast` (default) / `--full` | ~10% subsets (50 LME Q · 1 LoCoMo convo · 1 BEAM convo) vs full official sets |
| `--judge` | official-style LLM judge for LongMemEval (judge model recorded in report) |
| `--timeout / --keep-going` | per-run timeout, don't stop on failure |
| `--with-supermemory` | clone reference + install SDK; lineup join needs `SUPERMEMORY_API_KEY` |
| `--check` | install + probe + datasets only, run nothing |

Supermemory runs through `benchmarks/adapters/` on its **official SDK**
(`add` → poll `documents.get` to done → `search.memories`), answering
with the **same reader and prompt shape** as our engine. No key/package
means skipped-with-reason, never zeros.

How this stays unbiased (also printed in every `REPORT.md`): one rig, one
model, serial order, explicit system names (typos exit loudly), shared
judge, fail-closed third parties, side-by-side tables for every suite.

`REPORT.md` looks like this (portable, checkable):

```markdown
# cmbench report
**PASS** · 2026-09-08 05:20 UTC · model `qwen3:4b` · run order `contextmemory,supermemory,full-history`

| Run                  | Status | Time (s) | Result                    |
|---|---|---|---|
| bench:contextmemory  | ok     | 0.1      | ingest p50 0.043 ms; answer p50 0.202 ms |
| bench:supermemory    | ok     | 41.0     | ingest p50 812 ms; answer p50 640 ms      |
| longmemeval          | ok     | 412.0    | contextmemory: det 0.500  |
...
```

Metrics glossary: **det** = deterministic containment score (cheap lane);
**judge** = official-prompt LLM score (publish lane); **p50/p95** =
retrieval latency without any model; **tokens** = evidence packed per
query. Caveats travel with the report: same rig + same model or the
numbers mean nothing.

## Use it in 60 seconds

```bash
# Watch the brain work (offline scripted demo, no model needed)
uv run contextmemory demo

# Talk with memory (Ollama voice, ContextMemory brain)
uv run contextmemory chat --model qwen3:4b

# Shell-level memory: store, retrieve, profile — no model involved
uv run contextmemory ingest --turn 'user:I moved to Seattle.'
uv run contextmemory recall 'Where do I live?' --json
uv run contextmemory profile
```

```python
from contextmemory.api import MemoryClient

brain = MemoryClient("user_123")          # persistent journal, auto-loaded
brain.session(session)                    # ingest a conversation
report = brain.recall("Where do I live?") # ranked hits + evidence pack
answer, report = brain.ask("Where do I live?", reader)
profile = brain.profile()                 # static (durable) + dynamic (recent)
```

Memory is scoped by **container**: a user, project, repo, or agent.
Pass `--container` (CLI/MCP/HTTP) or the tag (API). Switch containers in
the TUI with `C`.

## Surfaces

| Surface | Entry point | Tools / endpoints |
| --- | --- | --- |
| TUI HQ | `contextmemory` / `demo` | Brain, Timeline, Why, Models, Retrieval Live, Performance, Connections, Health, **Profile, Setup, Help** |
| CLI | `contextmemory <cmd>` | `chat demo ask ingest recall profile setup mcp eval dims bench` |
| MCP (stdio) | `contextmemory mcp --container brain` | `memory recall context profile timeline forget` |
| HTTP | `:8765` (observatory + API) | `GET health graph metrics events profile` · `POST ask recall memories forget` |
| Python | `contextmemory.api.MemoryClient` | `add session search recall ask profile projection save load` |

MCP client config (OpenCode, Claude Code, Cursor, Cline — any MCP client):

```json
{
  "mcpServers": {
    "contextmemory": {
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/contextmemory",
               "contextmemory", "mcp", "--container", "brain"]
    }
  }
}
```

## How the brain works

```text
conversation / tool trace
          |
       CAPTURE      immutable episode, cheap and lossless (no LLM)
          |
       EXTRACT      one LLM call per session → compact cells
                    (offline fallback: deterministic, model-free)
          |
      RECONCILE     dedup → version → project truth (C++, deterministic)
          |
        RECALL      bounded temporal query plan, no LLM
          |
         PACK       minimum-sufficient evidence under a token budget
          |
        ANSWER      Ollama, a frontier model, or your own reader —
                    or an honest "I don't have enough information"
```

**Three layers, three jobs:**

- **C++ ETMC core** (`core/`, zero third-party deps): episodes, bi-temporal
  cells (`observed_at` × `valid_from/until`), state projections, hybrid
  retrieval (BM25 + dense + entity + projection channels fused with RRF),
  evidence packing, atomic journal persistence.
- **Python layer** (`contextmemory/`): orchestration, extraction, Ollama,
  public API, MCP, HTTP, TUI, evaluation. It never reimplements retrieval.
- **Evaluation rig** (`contextmemory/eval/`, `benchmarks/`): LongMemEval
  replay, custom dimensions (write precision, evolution, forgetting),
  deterministic latency bench.

**The temporal model.** "I live in New York" → "I moved to Seattle" keeps
both truths: Seattle is current, New York is history. Late-arriving older
events link as history but never rewind current truth. Forgotten cells
leave the read path but stay auditable. Empty evidence abstains instead
of fabricating.

**Why it's fast.** The read path calls no model and is fully bounded:
a compiled query plan caps candidates, channels run over a narrowed
region, and the packer stops at the token budget. Measured on this rig:
ingest p50 ~0.05 ms, answer p50 ~0.17 ms (deterministic path, no model;
`contextmemory bench`). LLM extraction is one completion per session by
design and is reported separately from retrieval latency.

**Why it survives.** Journals commit atomically (tmp + rename), load
transactionally (corruption never destroys live state), checksums cover
record types, old journal formats still load, enums and vector dims are
validated, version-chain walks are cycle-guarded, ranking is fully
deterministic (score ↓, id ↑), and the store is thread-safe. The C++
suite (15 tests, 65 checks) is ASan/UBSan clean.

## Benchmarks (honest numbers only)

Frontier instruments: **LoCoMo** (near-saturated regression test),
**LongMemEval** (the discriminative conversational bar),
**BEAM** (10M-token frontier), plus our `dims`/`bench` for the gaps no
public benchmark covers (write precision, evolution, forgetting,
deterministic latency). See `docs/research/frontier-memory-benchmarks.md`.

Measured here (CPU-only rig, deterministic):

- `bench`: ingest p50 ~0.05 ms · answer p50 ~0.17 ms
- `dims` + full-history reader (qwen2.5:1.5b): evolution 0.80 ·
  forgetting 1.00 · write-precision 0.60

Not claimed: LLM-judged leaderboard scores. Our one-rig harness
(`benchmarks/run_official.py`: BEAM + LongMemEval + LoCoMo, identical
reader) is the instrument for head-to-heads — run it before believing
any comparison, including ours. Published vendor numbers (Supermemory
#1 LongMemEval/LoCoMo/ConvoMem; Mem0 92.5/94.4/64.1/48.6) are
self-reported on different harnesses and are not head-to-head with us.

## ContextMemory vs the field

| Capability | ContextMemory | Supermemory (self-host) |
| --- | --- | --- |
| Local-first setup | `uv sync` + `setup` wizard, 294 KB wheel | single binary + wizard |
| Offline memory | yes (deterministic, model-free) | via local Ollama |
| Isolation | container tag | containerTag |
| Truth over time | validity windows, no-rewind projections | Updates/Extends/Derives |
| Read latency | sub-ms deterministic path (measured) | ~50 ms profiles (self-report) |
| MCP / HTTP / CLI / SDK / TUI | 6 tools · 9 endpoints · 11 commands · SDK · HQ | MCP + plugins · full API · dashboard |
| Deps to operate | 2 Python pkgs, no vector DB | embedded engine, zero-config |
| Connectors / RAG / files | out of scope (memory ≠ RAG) | Drive/Gmail/Notion/GitHub, SuperRAG |

Pick ContextMemory when you want a dependency-free, auditable,
crash-safe memory core beside your own model and stack. Pick a platform
when you want hosted connectors and file pipelines today.

## Repository map

```text
core/                 C++ ETMC engine + dependency-free tests
contextmemory/        API, engine, eval, MCP, HTTP, TUI, CLI, setup
tests/                100 automated tests (isolated from real journals)
benchmarks/           official-protocol runs + data (data excluded from sdist)
docs/architecture/    current design (production-hardening.md is newest)
docs/research/        durable synthesis (frontier-memory-benchmarks.md)
reports/runs/         dated run evidence (2026-09-07-hardening-and-surface.md)
scripts/verify.sh     one-command verification
```

## License

Apache-2.0.
