# Benchmark Results

**Latest run: 2026-09-30 · CPU-only · head-to-head vs Supermemory (self-hosted)**

Full write-up and judgement: [`reports/runs/2026-09-30-extraction-windowing-fix.md`](reports/runs/2026-09-30-extraction-windowing-fix.md) ·
(first run: [`2026-09-29-cpu-head-to-head-supermemory.md`](reports/runs/2026-09-29-cpu-head-to-head-supermemory.md))

## Headline

| Suite | contextmemory | supermemory | full-history | Winner |
|---|---|---|---|---|
| **LongMemEval** (n=12, LLM judge) | **0.583** | 0.167 | 0.667 | contextmemory |
| LoCoMo (convo 0, 199 QA) | 0.136 | 0.111 | — | contextmemory |
| bench ingest p50 | **0.045 ms** | 131 987 ms | 0.000 ms | contextmemory |

On the discriminative conversational benchmark we score **3.5× Supermemory**
(7/12 vs 2/12 correct) on the same rig, same reader, same 7B extraction
model — and sit close to the full-context upper bound (0.667).

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
| **LongMemEval** (n=12, LLM judge) | judged | **0.583** | 0.167 | 0.667 | contextmemory |
| LongMemEval | deterministic containment | 0.083 | 0.083 | 0.333 | full-history |
| **LoCoMo** (convo 0, 199 QA) | overall | **0.136** | 0.111 | — | contextmemory |
| LoCoMo | multi-hop QA (cat 1-4, n=152) | 0.066 | 0.072 | — | tie |
| LoCoMo | adversarial / abstain (cat 5, n=47) | **0.362** | 0.234 | — | contextmemory |
| `dims` | write-precision | **1.000** | **1.000** | 0.600 | tie |
| `dims` | evolution | 0.400 | **1.000** | 0.800 | supermemory |
| `dims` | forgetting | 0.333 | 0.333 | **1.000** | full-history |

LongMemEval per-type judge, contextmemory after the extraction fix:
`knowledge-update 0.5 · multi-session 0.5 · single-session-assistant 0.5 ·
single-session-preference 1.0 · single-session-user 0.5 ·
temporal-reasoning 0.5` (before the fix: 0.0 on five of six types).

LoCoMo re-measured **with** the fix: 0.136 (was 0.186 pre-fix) — the fix
stores more cells, so the reader commits to answers more often, which costs
adversarial abstention (0.574 → 0.362) while leaving multi-hop recall
unchanged. We still beat Supermemory (0.136 vs 0.111). The deliberate trade:
+7× on LongMemEval against -0.050 on a suite where we were already ahead.

### Latency (same synthetic workload, n=30)

| System | ingest p50 | ingest p95 | answer p50 | answer p95 |
|---|---|---|---|---|
| **contextmemory** | **0.045 ms** | **0.081 ms** | **0.159 ms** | **0.224 ms** |
| supermemory (self-hosted) | 131 987 ms | 340 346 ms | 35.4 ms | 43.0 ms |
| full-history | 0.000 ms | 0.000 ms | 0.025 ms | 0.056 ms |

## Verdict

- **We win LongMemEval 0.583 vs 0.167** (3.5×) and **LoCoMo 0.136 vs
  0.111**, on the same rig with the same reader, judge and write-path model.
- The LongMemEval win came from fixing a real defect, not from tuning:
  windowed extraction recovered the facts a single pass was dropping
  (0.083 → 0.583). See the fix report.
- LoCoMo's win is **epistemic discipline**, not recall: on multi-hop QA the
  two are tied (0.066 vs 0.072); we abstain correctly on 0.362 of
  adversarial items vs their 0.234. We are not claiming better recall.
- **We lose `dims` evolution to Supermemory** (0.400 vs 1.000): unqualified
  present-tense questions can resolve to a superseded value. Open defect.
- We win latency decisively and architecturally — ~2 900× on ingest,
  ~220× on answer — because our read path calls no model.
- We still trail the `full-history` upper bound on LongMemEval
  (0.583 vs 0.667): dumping the transcript into context is still the
  strongest single configuration on a 1.5b reader.
- Supermemory's CPU config is a **floor, not their ceiling**: their memory
  agent exceeded its internal ~270s budget on 4 of 19 LoCoMo documents, and
  their own cron later recovered some (0 → 15 memories observed). Their
  published "#1" claims are cloud-tier with proprietary extraction models.

## Known defects (open)

1. **Evolution.** Unqualified present-tense questions ("where does the
   user work?") can resolve to a superseded value; `dims` evolution 0.400
   vs Supermemory's 1.000.
2. **Residual recall misses.** 3 of 5 remaining LongMemEval failures are
   abstentions where the fact was in the haystack but not retrieved.
3. **Date arithmetic.** One answer invented a duration ("365 days ago")
   where the evidence gave a relative reference.

## Caveats

- Same rig, same reader, same judge, same write-path model — or the numbers
  mean nothing. **Not comparable to any vendor self-report.**
- Small samples (LME n=12, 1 LoCoMo conversation): directional, not
  significance. LongMemEval judge scores move in 1/12 steps.
- Deterministic containment understates both systems with a 1.5b reader
  (gold "Transgender woman" vs hypothesis "trans woman" is marked wrong);
  the LME judge score is the meaningful column there.
- CPU-only; a GPU host would lift both systems' absolute scores and would
  change their relative ingest latency far more than ours.
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
| 2026-09-29 | head-to-head, pre-fix | LME 0.083 · LoCoMo 0.186 · ingest p50 0.045 ms |
| 2026-09-30 | after extraction windowing | **LME 0.583** · LoCoMo 0.136 · ingest p50 0.045 ms |
