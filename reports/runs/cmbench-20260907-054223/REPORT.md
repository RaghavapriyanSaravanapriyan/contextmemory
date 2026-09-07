# cmbench report

**PASS** · 2026-09-07 05:42 UTC · model `qwen3:4b` · run order `contextmemory, full-history` · skipped: dims

## Rig

| Field | Value |
|---|---|
| date (UTC) | 2026-09-07 05:42 UTC |
| platform | Windows 10 (AMD64) |
| python | 3.11.2 |
| model (ALL systems) | qwen3:4b |
| reader base URL | http://localhost:11434 |
| run order | contextmemory, full-history |
| judge | deterministic-only (no LLM judge) |
| repo | https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git |

Reproduce:

```bash
python scripts/cmbench.py --model qwen3:4b --base-url http://localhost:11434 --systems contextmemory,full-history --suites bench --n 50
```

## Summary

| Run | Status | Time (s) | Result |
|---|---|---|---|
| bench:contextmemory | ok | 0.6 | ingest  p50    0.079 ms  p95    0.128 ms  mean    0.085 ms  (n=200); answer  p50    0.129 ms  p95    0.378 ms  mean    0.180 ms  (n=4) |
| bench:full-history | ok | 0.4 | ingest  p50    0.000 ms  p95    0.000 ms  mean    0.000 ms  (n=200); answer  p50    0.151 ms  p95    0.195 ms  mean    0.160 ms  (n=4) |

## bench — contextmemory

| System | Kind | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|---|
| contextmemory | ingest | 0.079 | 0.128 | 0.085 |
| contextmemory | answer | 0.129 | 0.378 | 0.180 |

Full log: `bench-contextmemory.log`

## bench — full-history

| System | Kind | p50 (ms) | p95 (ms) | mean (ms) |
|---|---|---|---|---|
| full-history | ingest | 0.000 | 0.000 | 0.000 |
| full-history | answer | 0.151 | 0.195 | 0.160 |

Full log: `bench-full-history.log`

## Datasets (official sources only)

| File | Size | Official source |
|---|---|---|
| (synthetic) | - | generated in-process, no download |

## Third-party readiness

| System | Status | Detail |
|---|---|---|
| supermemory | skipped | no key (export SUPERMEMORY_API_KEY=... (get one at supermemory.ai)) |

## Bias controls (how this stays honest)

- One rig, one reader model, serial execution — every system sees identical sessions in identical order.
- Official suites share one replay runner and one judge; judge: `deterministic-only (no LLM judge)`.
- Third parties use the same reader and the same prompt shape as the local engine (only retrieval/storage is theirs).
- Skipped contenders are reported as skipped with the reason — never as zero scores.
- `dims`/`bench` are ContextMemory harnesses for gaps public benchmarks don't cover; they run for EVERY lineup system, and the tables above show all of them side-by-side.
- Do not compare these numbers with other harnesses, judges, or dates.

## Checkpoints

- Per-run logs: `<outdir>/<suite>[-<system>].log`
- Official-run JSONL: `benchmarks/results/`
- This report: `REPORT.md` (here) + `summary.json`
