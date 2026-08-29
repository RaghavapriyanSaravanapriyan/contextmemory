# T001 — Beat Frontier Memory Layers

## Objective

Build ContextMemory into a memory layer for agentic systems that measurably
outperforms the frontier memory layers (Mem0, Zep/Graphiti, Letta, LangMem,
and the 2026 LongMemEval-S cluster: Hindsight, MemMachine, Honcho, etc.).

The human has defined "beat" as **both**:

1. **Benchmark-competitive**: head-to-head versus leading OSS systems on
   standard memory benchmarks (LongMemEval, then LoCoMo and BEAM) run on a
   single shared rig with identical reader models.
2. **Demonstrably better where the field is weak**: write precision,
   temporal evolution (updates, contradictions, staleness), forgetting, and
   low-latency deterministic retrieval — dimensions no public benchmark
   measures.

Constraint: the write path must be **model-agnostic and local-friendly**
(works with open-weight/local models and frontier APIs).

## Why this is winnable

- First-stage retrieval is close to saturated; the field polishes retrieval
  numbers while end-to-end answer accuracy lags (one system: 90% retrieval,
  57% end-to-end on preferences).
- Write precision, forgetting, and memory evolution are barely measured and
  widely reported as failing in production (stale facts, contradictions,
  overwrite-not-evolve).
- Latency: sub-200ms is the "feels native" bar; several systems run LLM calls
  on the read path and miss it by 100x (LangMem p50 ~18s).
- AMA-Bench (2026) shows systems fail to capture causal and objective
  information, relying on lossy similarity retrieval.

## Core principle

You cannot beat what you cannot measure. The measurement rig comes first.

## Milestones

- M0 Foundation: task, research report, project skeleton, verify.sh. **DONE**
  (2026-08-29): pyproject/uv env, `.gitignore`, 18 tests green, ruff clean,
  CLI (`contextmemory`) with full-history + recency baselines, end-to-end
  smoke run against the oracle dataset verified.
- M1 Measurement rig: LongMemEval replay harness + custom-dimensions harness
  (write precision, evolution, forgetting) + deterministic latency bench.
  **DONE** (2026-08-29): replay harness; `dims` scenarios (synthetic, known
  ground truth, abstention-vs-fabrication scoring) and `bench` (null-reader
  deterministic p50/p95) via CLI subcommands `eval`/`dims`/`bench`; `eval`
  gained official-style `--judge-model`. 38 tests green, ruff clean.
- M2 Baselines: full-history reader baseline, Mem0, Zep/Graphiti, Letta
  adapters on one rig, identical reader model.
- M3 ContextMemory architecture v1: model-agnostic incremental write path
  (extraction via pluggable LLM client), temporal store with evolution
  semantics, deterministic read path, consolidation/forgetting.
- M4 Iterate: ablate, measure, publish run reports.
- M5 Benchmark push: LoCoMo, BEAM.

## Definition of done (per milestone)

Each milestone closes with evidence: tests green via `scripts/verify.sh`,
a run report in `reports/runs/` or `reports/testing/`, and honest numbers.

## Open questions

- Which local reader/extraction models to standardize on for the shared rig
  (no GPU here; CPU-friendly 7-8B quantized via Ollama/vLLM, or frontier API
  when allowed).
- LongMemEval official scoring needs an OpenAI-style judge; plan a
  deterministic scoring path for dev iteration and use the LLM judge for
  final published numbers.