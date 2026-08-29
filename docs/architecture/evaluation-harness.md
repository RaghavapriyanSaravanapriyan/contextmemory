# Evaluation Harness

The measurement rig is the foundation of ContextMemory's challenge: you
cannot beat what you cannot measure. This document describes the harness
that benchmarks memory systems.

## Components

```
src/contextmemory/eval/
  protocol.py   MemorySystem (write+read seams), ReaderClient (model client)
  data.py       LongMemEval dataset loading (oracle / S / M)
  runner.py     chronological replay-and-answer loop, timing on both paths
  scoring.py    deterministic proxy + official-style LLM judge
  systems.py    reference baselines (full-history, recency window)
  dimensions.py custom dimensions the benchmarks don't measure
                (write precision, evolution, forgetting)
  latency.py    deterministic latency bench with a null reader
```

## Protocol seams

* `MemorySystem.ingest(session)` -- the write path. Called once per session,
  in chronological order, across an instance's history.
* `MemorySystem.answer(question, question_date)` -- the read path. Must
  answer using only stored memory.
* `ReaderClient.complete(messages)` -- the model. `OpenAICompatClient`
  speaks to any OpenAI-compatible endpoint, so the same harness runs
  against frontier APIs (OpenAI/Anthropic/Gemini-compatible) and local
  open-weight models (vLLM/Ollama/LM Studio). This is how model-agnosticism
  is enforced: extraction and answer generation share this client.

## Fairness rules

* Every system under test is built fresh per question instance (via a
  factory), so no state leaks between questions.
* Sessions are replayed in chronological order regardless of file order.
* The same `ReaderClient` (same model, same endpoint) is used for every
  system in a comparison run.
* Published scores must be LLM-judge scores using the replicated official
  LongMemEval prompts; the deterministic proxy is for development only.

## Scoring

* `deterministic_match` -- cheap lenient proxy for iteration; not official.
* `judge_results` -- LLM judge with the official LongMemEval answer-check
  prompts (replicated verbatim, including the per-task variants and
  abstention prompt), producing per-type and overall accuracy.

## CLI

```
contextmemory eval --data <longmemeval.json> --system <name> \
  --reader-api-base <base> --reader-model <model> \
  [--judge-model <model>] --out <run.jsonl>
contextmemory dims --system <name> --reader-api-base <base> \
  --reader-api-key <key> --reader-model <model>
contextmemory bench --system <name> [--sessions N]
```

Records question_id, hypothesis, judged label, and ingest/answer timing per
instance, which is the raw material for `reports/runs/`.

## Custom dimensions (`dims`)

`dimensions.py` runs fully-controlled synthetic timelines whose ground truth
is known, so scores are reproducible without a dataset:

* **write-precision** — the write path must store what was actually said, and
  the read path must abstain rather than fabricate when asked about something
  never discussed. Recall-style probes are scored with `deterministic_match`;
  abstention probes with `is_abstention` (a curated marker list, a proxy for
  the official abstention prompt).
* **evolution** — facts must track updates and contradictions over time:
  current value vs historical value, no staleness (a system stuck on an old
  employer fails the current-state probes).
* **forgetting** — stable core facts must survive consolidation while
  superseded facts stop contaminating current answers (long noisy timeline,
  then durable-name and current/historical-city probes).

`run_dimensions(scenarios, system_factory)` returns a per-dimension
`DimensionReport` with overall accuracy. These are development proxies;
published numbers come from an LLM judge.

## Deterministic latency bench (`bench`)

`latency.py` measures the memory system's *own* deterministic cost by running
ingest and answer against a `NullReader` that returns instantly, so timings
exclude LLM/network time. `bench_latency(system_factory, ...)` replays a
deterministic synthetic corpus and reports p50/p95 ingest and answer latency
in milliseconds. On one rig the cross-system comparison is the signal; the
interactive bar is sub-200ms p50 on the read path.

## Latency measurement

`Timing` records write-path (ingest) and read-path (answer) wall time per
instance. Comparing these across systems on one rig is the latency evidence;
the field's interactive bar is sub-200ms p50 on the read path.