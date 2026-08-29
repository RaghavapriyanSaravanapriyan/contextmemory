# ContextMemory

> a better persistent memory and context optimization layer for modern agentic systems.

ContextMemory is an engineering effort to build a memory layer for agentic
systems that measurably outperforms the 2026 frontier (Mem0, Zep/Graphiti,
Letta, LangMem, Supermemory, and the LongMemEval-S cluster) on **both** the
standard memory benchmarks and the dimensions the field does not measure:
write precision, temporal evolution, forgetting, and read-path latency.

The guiding principle: **you cannot beat what you cannot measure.** The
evaluation harness comes first, and every claim is backed by a reproducible
run.

- Mission and milestones: `tasks/active/T001-beat-frontier-memory-layers.md`
- Landscape analysis: `reports/research/2026-08-29-frontier-memory-landscape.md`
- System deep-dive: `reports/research/2026-08-29-agent-memory-systems-deep-dive.md`
- Architecture decision (C++ core): `reports/architecture/2026-08-29-graph-temporal-core.md`
- Harness architecture: `docs/architecture/evaluation-harness.md`

---

## Design overview

ContextMemory is a **two-layer system**.

### The C++ core (`core/`) — the memory engine

The entire memory engine is C++, dependency-free, compiled into the Python
package. It is an in-process **bi-temporal temporal graph store**:

- **Facts** carry a *validity window* (`valid_from`/`invalid_at`, when the
  fact was true) and a *transactional timeline* (`created_at`/`expired_at`,
  when it was ingested or superseded) — the model proven by Zep/Graphiti,
  combined with Supermemory's version-chain edge semantics (`updates` /
  `extends` / `derives` / `related` / `causal`) and Hindsight's fact-belief
  separation (`world` / `opinion` / `preference` / `episode`).
- **Updating is versioning, not overwrite.** `update_fact` invalidates the
  old fact and creates an `updates` edge; the new fact wins at its validity
  start, history is preserved and auditable.
- **Deterministic read path, no LLM.** Time-aware candidate filtering →
  hybrid channels (BM25 + vector + entity boost + recency) → Reciprocal Rank
  Fusion → token-budget-constrained assembly. `profile()` returns static
  (durable) vs dynamic (recent) memory with pure in-process operations.
- **Persistence** is an append-only binary journal (CRC32, length-prefixed)
  replayed on load.

Indexes are built on write: a hand-rolled BM25 inverted index, a brute-force
normalized vector index with an AVX2/FMA fast path, entity adjacency, and
per-container hard isolation (Supermemory-style namespaces). At single-user
scale (thousands of facts) brute-force cosine is microseconds; HNSW is a
measured follow-up, not a default.

### The Python layer (`src/contextmemory/`) — orchestration, not engine

Python drives the slow, model-dependent work and exposes a clean API:

- `engine/extractor.py` — the **single-pass write-path extraction** (one LLM
  call per session, Mem0/Hindsight style: coarse facts, no per-entity/per-edge
  calls). Model-agnostic: any OpenAI-compatible endpoint. A `NullExtractor`
  stores turns verbatim for tests and latency baselines.
- `engine/embedder.py` — pluggable embedding interface with a deterministic
  hash embedder for reproducible tests.
- `engine/memory.py` — the `MemoryEngine` tying extractor + embedder + store
  into the `ingest`/`recall`/`answer`/`profile` surface.
- `core.py` — typed Python facade (`MemoryStore`) over the C++ `_core`
  extension.
- `eval/` — the measurement rig (see below).

### The evaluation rig (`eval/`) — how we measure

Three CLI subcommands (all model-agnostic):

- `eval` — replays a LongMemEval dataset (oracle/S/M) through a memory system
  chronologically and scores answers. Deterministic proxy for iteration,
  official-style LLM judge (`--judge-model`) for published numbers.
- `dims` — runs the custom-dimension scenarios the public benchmarks don't
  cover: **write precision** (stores only what was said, abstains not
  fabricates), **evolution** (facts track updates/contradictions over time),
  **forgetting** (core facts survive consolidation, superseded facts stop
  contaminating current answers).
- `bench` — measures deterministic ingest/answer latency with a null reader,
  isolating the system's *own* cost (no LLM/network time). The interactive
  bar is sub-200ms p50 on the read path.

---

## Repository map

```text
core/                         C++ memory engine (cmcore)
  include/cmcore/             types (facts/edges/entities), store, indexes
  src/                        store + index implementations, journal
  python/bindings.cpp         nanobind surface (_core extension)
  tests/test_core.cpp         dependency-free C++ test suite
src/contextmemory/
  core.py                     typed Python facade over _core
  engine/                     extractor, embedder, memory orchestration
  eval/                       protocol, data, runner, scoring, systems,
                              dimensions, latency bench
  cli.py                      contextmemory CLI (eval / dims / bench)
tests/                        Python test suite (harness + engine)
benchmarks/data/              downloaded LongMemEval datasets (gitignored)
scripts/verify.sh             repository-level verification entry point
docs/architecture/            current system architecture
reports/research/             research investigations
reports/architecture/         architecture decisions
reports/runs/                 experiment records
tasks/active/                 active tasks
```

---

## Installation

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), a C++ compiler
(g++/clang), and CMake >= 3.20.

```bash
git clone https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git
cd contextmemory
uv sync --extra dev
```

`uv sync` builds the C++ core and compiles the `_core` extension in-place, so
the whole package (engine + harness) is usable from `uv run`. Everything is
embedded; there are no external services.

### Running on a fresh machine — quick verification

```bash
cd contextmemory
uv sync --extra dev              # builds C++ core + installs package
scripts/verify.sh                # Python tests + ruff
```

The C++ core has its own dependency-free test suite:

```bash
cmake -S core -B build/core && cmake --build build/core -j
./build/core/cmcore_test        # 12/12 suites
```

A direct smoke test of the engine through the Python binding:

```bash
uv run python -c "
from contextmemory.core import MemoryStore, WORLD
s = MemoryStore('demo')
f = s.add_fact('User prefers TypeScript over Python', kind=WORLD, is_static=True, ts=1700000000000)
print('fact id:', f, '| search:', s.search('programming TypeScript', at_time=1700000000001)[0].text)
"
```

---

## Usage

### Evaluate on LongMemEval

Download the data first:

```bash
mkdir -p benchmarks/data
cd benchmarks/data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

Then run a baseline. The reader client speaks to any OpenAI-compatible
endpoint: frontier APIs, vLLM, Ollama, LM Studio.

```bash
uv run contextmemory eval \
  --data benchmarks/data/longmemeval_oracle.json \
  --system full-history \
  --reader-api-base https://api.openai.com/v1 \
  --reader-api-key $OPENAI_API_KEY \
  --reader-model gpt-4o-mini \
  --out reports/runs/run.jsonl
```

Add `--judge-model <model>` to score with an LLM judge using the official
LongMemEval prompts (for published numbers).

### Run the custom-dimension scenarios

```bash
uv run contextmemory dims --system full-history \
  --reader-api-base https://api.openai.com/v1 \
  --reader-api-key $OPENAI_API_KEY --reader-model gpt-4o-mini
# write precision, evolution, forgetting — the dimensions benchmarks don't measure
```

### Measure deterministic latency (no model needed)

```bash
uv run contextmemory bench --system full-history --sessions 200
```

## Development

```bash
scripts/verify.sh     # run the full verification (tests + lint)
uv run pytest         # Python test suite
uv run ruff check src tests
cmake -S core -B build/core && cmake --build build/core -j && ./build/core/cmcore_test
```

---

## Status

| Milestone | Status |
|---|---|
| M0 Foundation | done |
| M1 Measurement rig (eval + dims + bench) | done |
| M2 Baselines (Mem0, Zep/Graphiti, Letta adapters) | pending |
| M3 ContextMemory architecture v1 (C++ core + engine) | in progress |
| M4 Iterate (ablate, measure, run reports) | pending |
| M5 Benchmark push (LoCoMo, BEAM) | pending |