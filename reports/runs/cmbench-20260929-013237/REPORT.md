# cmbench report

**FAIL** · 2026-09-29 01:32 UTC · model `qwen2.5:1.5b` · extract `qwen2.5:7b` · run order `contextmemory, supermemory, full-history`

> RAN: contextmemory, full-history, lineup, supermemory · (all requested contenders ran)

## Rig

| Field | Value |
|---|---|
| date (UTC) | 2026-09-29 01:32 UTC |
| os | Linux 6.12.74+deb13+1-amd64 (x86_64) |
| cpu cores | 16 |
| ram | 32.2 GB |
| python | 3.13.5 |
| model (ALL systems) | qwen2.5:1.5b |
| extract model (ALL write paths) | qwen2.5:7b |
| reader base URL | http://localhost:11434 |
| run order | contextmemory, supermemory, full-history |
| judge | qwen2.5:1.5b (reader-as-judge) |
| repo | https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git |

Reproduce:

```bash
python scripts/cmbench.py --model qwen2.5:1.5b --extract-model qwen2.5:7b --base-url http://localhost:11434 --systems contextmemory,supermemory,full-history --suites dims,bench,longmemeval --n 12 --bench-sessions 30 --judge
```

## Summary

| Run | Status | Time (s) | Result |
|---|---|---|---|
| dims:contextmemory | ok | 1293.7 | evolution overall: 0.4000 (5 probes); forgetting overall: 0.3333 (3 probes); write-precision overall: 1.0000 (5 probes) |
| dims:supermemory | ok | 1780.9 | evolution overall: 1.0000 (5 probes); forgetting overall: 0.3333 (3 probes); write-precision overall: 1.0000 (5 probes) |
| dims:full-history | ok | 25.0 | evolution overall: 0.8000 (5 probes); forgetting overall: 1.0000 (3 probes); write-precision overall: 0.6000 (5 probes) |
| bench:contextmemory | ok | 0.1 | ingest  p50    0.045 ms  p95    0.081 ms  mean    0.054 ms  (n=30); answer  p50    0.159 ms  p95    0.224 ms  mean    0.155 ms  (n=4) |
| bench:supermemory | ok | 4965.5 | ingest  p50 131987.061 ms  p95 340346.073 ms  mean 165503.754 ms  (n=30); answer  p50   35.439 ms  p95   42.978 ms  mean   37.038 ms  (n=4) |
| bench:full-history | ok | 0.1 | ingest  p50    0.000 ms  p95    0.000 ms  mean    0.000 ms  (n=30); answer  p50    0.025 ms  p95    0.058 ms  mean    0.033 ms  (n=4) |
| longmemeval | exit 1 | 6998.6 | contextmemory: det 0.000 judge 0.0833 |

## dims — contextmemory

| System | Dimension | Score | Probes |
|---|---|---|---|
| contextmemory | evolution | 0.4000 | 5 |
| contextmemory | forgetting | 0.3333 | 3 |
| contextmemory | write-precision | 1.0000 | 5 |

Full log: `dims-contextmemory.log`

## dims — supermemory

| System | Dimension | Score | Probes |
|---|---|---|---|
| supermemory | evolution | 1.0000 | 5 |
| supermemory | forgetting | 0.3333 | 3 |
| supermemory | write-precision | 1.0000 | 5 |

Full log: `dims-supermemory.log`

## dims — full-history

| System | Dimension | Score | Probes |
|---|---|---|---|
| full-history | evolution | 0.8000 | 5 |
| full-history | forgetting | 1.0000 | 3 |
| full-history | write-precision | 0.6000 | 5 |

Full log: `dims-full-history.log`

## bench — contextmemory

| System | Kind | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|---|
| contextmemory | ingest | 0.045 | 0.081 | 0.054 |
| contextmemory | answer | 0.159 | 0.224 | 0.155 |

Full log: `bench-contextmemory.log`

## bench — supermemory

| System | Kind | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|---|
| supermemory | ingest | 131987.061 | 340346.073 | 165503.754 |
| supermemory | answer | 35.439 | 42.978 | 37.038 |

Full log: `bench-supermemory.log`

## bench — full-history

| System | Kind | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|---|
| full-history | ingest | 0.000 | 0.000 | 0.000 |
| full-history | answer | 0.025 | 0.058 | 0.033 |

Full log: `bench-full-history.log`

## longmemeval (lineup)

| System | Deterministic | Judge |
|---|---|---|
| contextmemory | 0.000 | 0.0833 |

Full log: `longmemeval.log`

## Datasets (official sources only)

| File | Size | Items | Official source |
|---|---|---|---|
| benchmarks/data/longmemeval_oracle.json | 15.4 MB | 500 | https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json |

## Third-party readiness

| System | Status | Detail |
|---|---|---|
| supermemory | in lineup | ready (supermemory 3.62.0, self-hosted at http://localhost:6767) |

## Bias controls (how this stays honest)

- One rig, one reader model, serial execution — every system sees identical sessions in identical order.
- Official suites share one replay runner and one judge; judge: `qwen2.5:1.5b (reader-as-judge)`.
- Third parties use the same reader and the same prompt shape as the local engine (only retrieval/storage is theirs).
- Skipped contenders are reported as skipped with the reason — never as zero scores.
- `dims`/`bench` are ContextMemory harnesses for gaps public benchmarks don't cover; they run for EVERY lineup system, and the tables above show all of them side-by-side.
- Do not compare these numbers with other harnesses, judges, or dates.

## Checkpoints

- Per-run logs: `<outdir>/<suite>[-<system>].log`
- Official-run JSONL: `benchmarks/results/`
- This report: `REPORT.md` (here) + `summary.json`
