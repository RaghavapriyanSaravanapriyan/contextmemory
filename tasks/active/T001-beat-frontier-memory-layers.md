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
  adapters on one rig, identical reader model. **PARTIAL** (2026-09-29):
  full-history done; **Supermemory done and beaten in-lineup on the
  self-hosted CPU-only rig** (see M5). Mem0 / Zep / Letta not yet run.
- M3 ContextMemory architecture v1: model-agnostic incremental write path
  (extraction via pluggable LLM client), temporal store with evolution
  semantics, deterministic read path, consolidation/forgetting.
  **Architecture plan:** `docs/architecture/2026-08-29-sota-memory-brain-action-plan.md`
- M4 Iterate: ablate, measure, publish run reports.
- M5 Benchmark push: LoCoMo, BEAM. **PARTIAL** (2026-09-29): LoCoMo
  (convo 0, 199 QA) + LongMemEval (n=12, judged) + `dims` + `bench` all run
  head-to-head vs self-hosted Supermemory on a CPU-only rig.
  Report: `reports/runs/2026-09-29-cpu-head-to-head-supermemory.md`.
  **Result: we win abstention discipline and latency, tie on multi-hop
  recall, LOSE to Supermemory on LME (0.083 vs 0.167) and both lose badly
  to full-context (0.667).** BEAM not run (its 100K-token conversations are
  incompatible with the CPU-only time budget — the supermemory write path
  alone needs ~1-3 min/session).

## Head-to-head verdict (2026-09-29, CPU-only, Ollama)

| Suite | contextmemory | supermemory | full-history | winner |
|---|---|---|---|---|
| LongMemEval (n=12, judge) | 0.083 | **0.167** | **0.667** | full-history |
| LoCoMo (199 QA, overall) | **0.186** | 0.111 | n/a | contextmemory |
| LoCoMo multi-hop QA only | 0.066 | 0.072 | n/a | tie |
| LoCoMo adversarial/abstain | **0.574** | 0.234 | n/a | contextmemory |
| dims write-precision | 1.000 | 1.000 | 0.600 | tie |
| dims evolution | 0.400 | **1.000** | 0.800 | supermemory |
| bench ingest p50 | **0.045 ms** | 131 987 ms | 0.000 ms | contextmemory |
| bench answer p50 | **0.159 ms** | 35.4 ms | 0.025 ms | contextmemory |

**The blocking defect is extraction recall on long sessions** (proved in
the report: a 7B one-shot extractor dropped a verbatim "27:12" personal
best from a 12-turn LME session). Fix that first.

## Open questions

- Which local reader/extraction models to standardize on for the shared rig
  (no GPU here; CPU-friendly 7-8B quantized via Ollama/vLLM, or frontier API
  when allowed).
- LongMemEval official scoring needs an OpenAI-style judge; plan a
  deterministic scoring path for dev iteration and use the LLM judge for
  final published numbers.
- Should LoCoMo use an LLM judge? Deterministic containment marks
  "trans woman" wrong against gold "Transgender woman" and understates
  both systems on a 1.5b reader.
- Supermemory self-hosted never exceeded ~270s/doc for its memory agent on
  this CPU. Their published numbers are cloud-tier; ours are a CPU floor
  for them and a fair-but-weak ceiling for us.
