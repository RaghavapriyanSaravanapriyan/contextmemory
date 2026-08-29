# Architecture Decision — Graph + Temporal Memory Core in C++

**Date:** 2026-08-29
**Status:** Accepted (v1 scope defined; iterate on evidence)
**Problem:** Design the ContextMemory memory layer to beat Supermemory
(supermemoryai) — the 2026 #1 on LongMemEval/LoCoMo/ConvoMem at ~85%
production accuracy, ~720 tokens/query context, sub-300ms search, ~50ms
profiles — on token usage, latency, and benchmark score. Direction from the
human: a graph + temporal memory layer, adopting the successful methods of
Supermemory and other leading memory layers, with a C++ core for bare-metal
speeds and a single-command install.

## Research basis

Full evidence in `reports/research/2026-08-29-agent-memory-systems-deep-dive.md`
and `reports/research/2026-08-29-frontier-memory-landscape.md`. Key facts:

- **Supermemory (target):** TS/Bun, one embedded Postgres (PGlite) + pgvector
  HNSW + Postgres FTS + relational graph tables. Write path = agentic
  tool-calling "dreaming" model (1 invocation, up to 40 tool rounds) with
  create/update/forget/relation tools. Memory model: facts/preferences/
  episodes, `isLatest` versioning, `updates`/`extends`/`derives` edges,
  `forgetAfter` expiry, static vs dynamic memories. Read path = hybrid search
  + graph traversal; optional LLM query rewriting (+~400ms) and LLM merge for
  benchmark aggregation. Profiles = pure SQL (~50ms). ~305MB Bun binary,
  one-command curl install. Their engine is closed source.
- **Zep/Graphiti:** bi-temporal validity windows (t_valid/t_invalid +
  t'created/t'expired), LLM-judged edge invalidation (never delete). Write is
  expensive (~6-10 LLM calls/episode, ~$0.018/episode, ~160 calls/5KB doc).
  Read is LLM-free: cosine + BM25 + BFS, MMR rerank.
- **Mem0 v3:** single-pass ADD-only extraction (~1 LLM call + lookup), hash
  dedup, nothing overwritten. Read = parallel fusion (semantic/keyword/
  entity/temporal), no LLM.
- **Hindsight:** 4 networks (world/experience/opinion/observation), 2-5 coarse
  narrative facts per conversation, TEMPR + CARA. Read = 4-way parallel
  retrieval (semantic/BM25/graph/temporal) → RRF + cross-encoder, token-budget
  constrained. Postgres + pgvector. Best published LongMemEval 91.4%
  (Gemini-3 Pro), 83.6% (OSS-20B).
- **SwiftMem:** route-then-search + temporal index → 10.8ms/query, 47x faster
  than Zep. Deterministic retrieval can be ~10ms.
- **Graph DBs 2026:** no permissive embedded graph engine remains (Kuzu
  archived Oct 2025). Postgres+pgvector (boring) won anyway.
- **Forgetting research:** FadeMem (adaptive decay, 45% storage cut),
  SleepGate (evict superseded traces), Nemori (prediction-error distillation),
  memory-mgmt empirics (selective addition + deletion-on-evidence → +10%).

## Constraints

- No GPU; CPU-friendly and local-model friendly (model-agnostic by design).
- Single-command install (pip-installable, embedded, zero external services).
- Read path must be deterministic and fast (target p50 < 20ms in-process;
  sub-200ms is the interactive bar; Supermemory ~300ms).
- Write path LLM cost must be lower than Supermemory's agentic dreaming.
- Correctness first; measure before optimizing.

## Alternatives considered

1. **Postgres + pgvector (Hindsight's choice).** Boring, proven, benchmark-
   winning. But: external dependency (or PGlite WASM like Supermemory — adds
   ~200MB), JSON/IPC round-trip per query, not "bare metal". Rejected for the
   core hot path; acceptable later as a pluggable backend.
2. **Embedded graph DB (Kuzu/FalkorDBLite).** Kuzu dead; FalkorDBLite SSPL
   (restrictive). Rejected.
3. **Rust core.** Equally valid, but the project's C++ engineering standard
   (AGENTS.md §17) and the explicit human direction favor C++.
4. **Pure Python engine.** Read path would be ~10-100x slower than C++ for
   BM25/vector/fusion; loses the bare-metal differentiator. Rejected.

## Decision

Two-layer system. The **entire memory engine core is C++**, zero external
dependencies (no SQL, no graph DB, no vector DB), compiled into the Python
package. Python is the LLM-orchestration and public-API layer only.

### C++ core (`cmcore`) — in-process temporal graph store

- **Data model:** entities; facts with **bi-temporal validity windows**
  (`valid_from`/`invalid_at`, `created_at`/`expired_at`), kind
  (world/opinion/preference/episode), `is_static`, `confidence` (opinion
  reinforcement), `is_latest` + version chain (`parent`/`root`), `forget_after`
  auto-expiry, source provenance; edges typed
  (`updates`/`extends`/`derives`/`related`/`causal`) — the Supermemory edge
  model combined with Graphiti's validity windows and Hindsight's
  fact-belief separation.
- **Indexes (all in-process, built on write):** BM25 inverted index (own
  tokenizer + IDF), vector index (SIMD cosine; HNSW later, measured), graph
  adjacency, per-container namespaces (Supermemory-style hard isolation).
- **Write path:** atomic batch ops (`create`/`update`/`link`/`expire`/`forget`).
  `update` invalidates the old fact (`is_latest=false`, closes `invalid_at`)
  and creates an `updates` edge — versioning, not overwrite (Supermemory +
  Graphiti). Ops applied synchronously in microseconds; the Python layer does
  the slow LLM work.
- **Read path (deterministic, no LLM):** time-aware candidate filtering (facts
  valid at question time) → hybrid channels (BM25 + vector + entity boost +
  recency) → RRF fusion → token-budget-constrained assembly (Hindsight). This
  is the main difference from Supermemory: no LLM query rewriting or LLM merge
  on the hot path.
- **Persistence:** append-only binary journal (write ops log, CRC32,
  length-prefixed), snapshot later. Boot-time load in microseconds.

### Python layer (`contextmemory`)

- **Engine:** single-pass extraction (Mem0/Hindsight: 1 LLM call per session,
  coarse facts, no per-entity calls), pluggable model client (existing
  OpenAI-compatible `protocol.OpenAICompatClient`), pluggable embedder.
- **API:** `add`/`session`, `search`/`recall`, `profile`, `forget` — mirrors
  Supermemory's surface for head-to-head.
- **Bindings:** nanobind over scikit-build-core → `pip install contextmemory`
  compiles the C++ core (single-command install).

### What we deliberately do not build yet (measure first)

HNSW (brute-force SIMD is fast at single-user scale; benchmark first), SwiftMem
temporal index (O(n) scan is microseconds at thousands of facts; benchmark
first), cross-encoder rerank (needs a model; add behind a flag), scheduled
consolidation jobs (v2).

## Expected outcomes (targets vs Supermemory)

- Accuracy: ≥85% LongMemEval-S on a shared rig (Hindsight proves 83.6% with an
  OSS-20B backbone; the architecture carries most of it).
- Context: < 700 tokens/query (vs Supermemory ~720).
- Read latency: p50 < 20ms in-process (vs Supermemory ~300ms search,
  ~50ms profile) — the C++ core is the moat here.
- Write LLM cost: 1 extraction call per session (vs agentic multi-round
  dreaming).

## Consequences

- Correctness and measurement first: every claim must come from the shared
  harness (`scripts/verify.sh` + eval harness), not vendor numbers.
- The C++ core is the durable engineering asset; Python stays thin.
- Persistence v1 is a journal; durability semantics are write-on-apply with
  fsync configurable.
- Re-evaluate HNSW, temporal index, and rerank once we have baseline numbers.

## Open questions

- Embedding model for the shared rig (local CPU-friendly vs API).
- Whether benchmark aggregation needs an LLM merge step (Supermemory's 95%
  Recall@15 uses it) or whether deterministic assembly suffices; test both.