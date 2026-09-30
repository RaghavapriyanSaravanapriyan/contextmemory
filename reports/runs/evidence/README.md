# Benchmark evidence — raw records

The per-question / per-QA records behind every number published in
[`BENCHMARKS.md`](../../BENCHMARKS.md). `benchmarks/results/` is
gitignored as scratch space, so the evidence backing the published claims
is committed here, where it can be audited.

`locomo_*.jsonl` — one record per QA item (`qid`, `category`, `q`, `gold`,
`hyp`, `ok`). `longmemeval_*.jsonl` — one record per question (`qid`,
`type`, `q`, `gold`, `hyp`, timings). `*.json` — the scored roll-up
including per-type judge scores and every judged hypothesis.

## LongMemEval (n=12 stratified, LLM judge, `qwen2.5:1.5b`)

| File | System | Score |
|---|---|---|
| `longmemeval_20260930-104745.json` (+ `_contextmemory.jsonl`) | contextmemory, **post-fix** | **judge 0.5833 (7/12)** |
| `longmemeval_20260929-113247.json` (+ `_supermemory.jsonl`) | supermemory (self-hosted) | judge 0.1667 (2/12) |
| `longmemeval_20260929-133817.json` (+ `_full-history.jsonl`) | full-history | judge 0.6667 (8/12) |
| `longmemeval_20260929-091705_contextmemory.jsonl` (+ `_rescored.json`) | contextmemory, **pre-fix** | judge 0.0833 (1/12) |

Same rig, same reader, same judge, same `qwen2.5:7b` on both write paths.
The pre/post pair is the extraction-windowing change
(`contextmemory/engine/extractor.py`).

## LoCoMo (official convo 0, all 199 QA, deterministic containment)

| File | System | Score |
|---|---|---|
| `locomo_20260930-120434.jsonl` | contextmemory, **post-fix** | **27/199 = 0.136** |
| `locomo_20260929-141641.jsonl` | contextmemory, pre-fix | 37/199 = 0.186 |
| `locomo_20260929-133709.jsonl` | supermemory (self-hosted) | 22/199 = 0.111 |

Category breakdown in each run report (`2026-09-29-cpu-head-to-head-supermemory.md`,
`2026-09-30-extraction-windowing-fix.md`); the roll-ups are in
`../2026-09-29-locomo-head-to-head.json` and `../2026-09-30-locomo-postfix.json`.

Not included: `locomo_20260929-125337.jsonl`, a superseded run that
competed for the CPU with a concurrent job and scored 32/199. It is not
part of any published number.

## dims + bench

Console transcripts for all three systems are in
`../cmbench-20260929-013237/` (`dims-*.log`, `bench-*.log`, `REPORT.md`).

## How to check a claim

```bash
# LongMemEval roll-up, including every judged hypothesis
python -m json.tool reports/runs/evidence/longmemeval_20260930-104745.json | head -40

# recount LoCoMo straight from the per-QA records
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("reports/runs/evidence/locomo_20260930-120434.jsonl")]
qa  = [r for r in rows if r["cat"] != 5]
adv = [r for r in rows if r["cat"] == 5]
f = lambda xs: f"{sum(r['ok'] for r in xs)}/{len(xs)}"
print("overall", f(rows), "| multi-hop", f(qa), "| adversarial", f(adv))
PY
```

Caveats that travel with these numbers: same rig + same reader + same
judge, or the numbers mean nothing; n=12 and one conversation are small
samples; nothing here is comparable to a vendor self-report.
