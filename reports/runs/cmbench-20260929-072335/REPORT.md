# cmbench report

**FAIL** · 2026-09-29 07:23 UTC · model `qwen2.5:1.5b` · extract `qwen2.5:7b` · run order `contextmemory, supermemory`

> RAN: lineup · (all requested contenders ran)

## Rig

| Field | Value |
|---|---|
| date (UTC) | 2026-09-29 07:23 UTC |
| os | Linux 6.12.74+deb13+1-amd64 (x86_64) |
| cpu cores | 16 |
| ram | 32.2 GB |
| python | 3.13.5 |
| model (ALL systems) | qwen2.5:1.5b |
| extract model (ALL write paths) | qwen2.5:7b |
| reader base URL | http://localhost:11434 |
| run order | contextmemory, supermemory |
| judge | deterministic-only (no LLM judge) |
| repo | https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git |

Reproduce:

```bash
python scripts/cmbench.py --model qwen2.5:1.5b --extract-model qwen2.5:7b --base-url http://localhost:11434 --systems contextmemory,supermemory --suites locomo --n 500 --bench-sessions 200
```

## Summary

| Run | Status | Time (s) | Result |
|---|---|---|---|
| locomo | exit 1 | 2134.6 | convo 0 contextmemory: 32/199=0.161 |

## locomo (lineup)

| Convo | System | Score |
|---|---|---|
| 0 | contextmemory | 32/199=0.161 (ingest 1303s) |

Full log: `locomo.log`

## Datasets (official sources only)

| File | Size | Items | Official source |
|---|---|---|---|
| benchmarks/data/locomo10.json | 2.8 MB | 10 | https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json |

## Third-party readiness

| System | Status | Detail |
|---|---|---|
| supermemory | in lineup | ready (supermemory 3.62.0, self-hosted at http://localhost:6767) |

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
