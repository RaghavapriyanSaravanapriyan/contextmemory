# Benchmark Results

**Latest run: 2026-09-29 · CPU-only · head-to-head vs Supermemory (self-hosted)**

Full write-up and judgement: [`reports/runs/2026-09-29-cpu-head-to-head-supermemory.md`](reports/runs/2026-09-29-cpu-head-to-head-supermemory.md)

## Rig

| Field | Value |
|---|---|
| Hardware | AMD Ryzen 7 5700G (8C/16T), 32 GB RAM — **no GPU** |
| OS / runtime | Debian, Python 3.13, Ollama (local models only) |
| Reader + judge (ALL systems) | `qwen2.5:1.5b` |
| Extraction (ALL write paths) | `qwen2.5:7b` — same model for both contenders |
| Contender | `supermemory-server` v0.0.8 (their own release) + official SDK `supermemory==3.62.0` |
| Contender embeddings | local `Xenova/bge-base-en-v1.5` (768d, their default) |
| Execution | serial, one shared replay runner, one judge, same prompts |

Fairness rules held: one reader, one judge, one write-path model, identical
inputs in identical order, no zeros for skipped work.

## Results

### Accuracy

| Suite | Metric | contextmemory | supermemory | full-history | Winner |
|---|---|---|---|---|---|
| **LoCoMo** (convo 0, 199 QA) | overall | **0.186** | 0.111 | — | contextmemory |
| LoCoMo | multi-hop QA (cat 1-4, n=152) | 0.066 | 0.072 | — | tie |
| LoCoMo | adversarial / abstain (cat 5, n=47) | **0.574** | 0.234 | — | contextmemory |
| **LongMemEval** (n=12, LLM judge) | judged | 0.083 | **0.167** | **0.667** | full-history |
| LongMemEval | deterministic containment | 0.000 | 0.083 | 0.333 | full-history |
| `dims` | write-precision | **1.000** | **1.000** | 0.600 | tie |
| `dims` | evolution | 0.400 | **1.000** | 0.800 | supermemory |
| `dims` | forgetting | 0.333 | 0.333 | **1.000** | full-history |

### Latency (same synthetic workload, n=30)

| System | ingest p50 | ingest p95 | answer p50 | answer p95 |
|---|---|---|---|---|
| **contextmemory** | **0.045 ms** | **0.081 ms** | **0.159 ms** | **0.224 ms** |
| supermemory (self-hosted) | 131 987 ms | 340 346 ms | 35.4 ms | 43.0 ms |
| full-history | 0.000 ms | 0.000 ms | 0.025 ms | 0.056 ms |

## Verdict

- **We win** LoCoMo overall (0.186 vs 0.111) — and the win is **epistemic
  discipline**, not recall: on multi-hop QA the two are tied (0.066 vs
  0.072); we abstain correctly on 0.574 of adversarial items vs their 0.234.
- **We lose** LongMemEval (0.083 vs 0.167).
- **We lose badly to no memory at all**: `full-history` scores 0.667 on
  LongMemEval — 4-8x both memory layers. On a 1.5b reader, dumping the
  transcript into context beats both memory systems. That result travels
  with these numbers.
- **We win latency decisively and architecturally** — ~2 900x on ingest,
  ~220x on answer — because our read path calls no model.
- Supermemory's CPU config is a **floor, not their ceiling**: their memory
  agent exceeded its internal ~270s budget on 4 of 19 LoCoMo documents, and
  their own cron later recovered some (0 -> 15 memories observed). Their
  published "#1" claims are cloud-tier with proprietary extraction models.

## Known defect (blocking)

**Extraction recall on long sessions.** Our one-shot LLM extractor drops
salient facts when a session is long. Reproduced directly: for LongMemEval
instance `6a1eabeb` the haystack states verbatim "I recently set a
personal best time in a charity 5K run with a time of 27:12" — the
extractor returned 3 cells, none of them that fact, and the read path then
had nothing to retrieve. This single trace explains most of the LongMemEval
score. Second defect: unqualified present-tense questions ("where does the
user work?") resolve to a superseded value (`dims` evolution 0.40).

## Caveats

- Same rig, same reader, same judge, same write-path model — or the numbers
  mean nothing. **Not comparable to any vendor self-report.**
- Small samples (LME n=12, 1 LoCoMo conversation): directional, not
  significance.
- CPU-only; a GPU host would lift both systems' absolute scores and would
  change their relative ingest latency far more than ours.
- LoCoMo is scored by deterministic containment, which understates both
  systems with a 1.5b reader (gold "Transgender woman" vs hypothesis "trans
  woman" is marked wrong).
- **BEAM not run** — its 100K-token conversations do not fit a CPU-only time
  budget. Skipped with reason, never as a zero.

## Reproduce

```bash
# 1. self-hosted Supermemory (CPU-only, local models)
OPENAI_BASE_URL=http://localhost:11435/v1 OPENAI_API_KEY=ollama \
OPENAI_MODEL=qwen2.5:7b OPENAI_FAST_MODEL=qwen2.5:1.5b \
SUPERMEMORY_DATA_DIR=/tmp/sm ./supermemory-server
# export SUPERMEMORY_API_KEY=<key it prints> SUPERMEMORY_BASE_URL=http://localhost:6767

# 2. the head-to-head
python scripts/cmbench.py --model qwen2.5:1.5b --extract-model qwen2.5:7b \
  --systems contextmemory,supermemory,full-history \
  --suites dims,bench,longmemeval --n 12 --bench-sessions 30 --judge --yes
```

## Historical runs

| Date | Suite | Result |
|---|---|---|
| 2026-09-07 | bench (Windows) | ingest p50 0.079 ms, answer p50 0.129 ms |
| 2026-09-29 | dims + full-history reader | evolution 0.80 · forgetting 1.00 · write-precision 0.60 |
| 2026-09-29 | head-to-head | see above |
