# Run Report — CPU-only head-to-head vs Supermemory (self-hosted) (2026-09-29)

## What this is

The first **real, local, CPU-only head-to-head** between ContextMemory and
Supermemory on one rig with one reader model — no cloud keys, no GPUs.
Supermemory was run from its own GitHub release (`supermemory-server`
v0.0.8, the self-hosted "Supermemory local" binary) with the official
Python SDK (`supermemory==3.62.0`) pointed at it, exactly as their docs
prescribe for offline use.

Rig: AMD Ryzen 7 5700G (8C/16T, **no GPU**), 32 GB RAM, Debian, Ollama
only. Reader + judge: `qwen2.5:1.5b`. Write path (extraction, both
systems): `qwen2.5:7b`. Supermemory embeddings: local
`Xenova/bge-base-en-v1.5` (768d, their default, no key).

## Rig fairness rules held

* One reader model (`qwen2.5:1.5b`) generated every answer for every system.
* One judge (same 1.5b) scored every hypothesis, same prompts, same order.
* The **same extraction model (`qwen2.5:7b`) drives both write paths** —
  ContextMemory's `LLMExtractor` and Supermemory's memory agent. Neither
  side got the weaker write path.
* Serial execution, one shared replay runner, no silent fallbacks, no
  zeros for skipped work.
* Supermemory's ingest was polled to `done` (their pipeline is async), so
  no probe ever raced indexing.

## Results

### LongMemEval (official oracle, n=12 stratified, LLM judge)

| System | det | **judge** | ingest (s) | answer (s) |
|---|---|---|---|---|
| contextmemory | 0.000 | **0.083** | 2241 | 12.7 |
| supermemory (self-hosted) | 0.083 | **0.167** | 4446 | 14.8 |
| full-history (reference) | 0.333 | **0.667** | 0 | 738 |

Per-type judge: contextmemory `knowledge-update 0.5, multi-session 0.0,
single-session-assistant 0.0, single-session-preference 0.0,
single-session-user 0.0, temporal-reasoning 0.0`; supermemory
`0.0 / 0.5 / 0.5 / 0.0 / 0.0 / 0.0`.

**ContextMemory loses this suite.** It is not close, and it is not close to
the in-context upper bound either. See the diagnosis below — this is a
product defect, not noise.

### LoCoMo (official convo 0, all 199 QA, deterministic containment)

| System | Overall | Multi-hop QA (cat 1-4) | Adversarial/abstain (cat 5) | Ingest (s) |
|---|---|---|---|---|
| contextmemory | **37/199 = 0.186** | 10/152 = 0.066 | 27/47 = **0.574** | 3916 |
| supermemory (self-hosted) | 22/199 = 0.111 | 11/152 = **0.072** | 11/47 = 0.234 | 11935 |

**Read this carefully — the headline is misleading.** On actual multi-hop
QA the two systems are statistically tied (0.066 vs 0.072, n=152: a
3-item difference). The entire overall gap comes from **adversarial
category 5**, where the correct behaviour is to *abstain* rather than
answer: ContextMemory abstains correctly 0.574 of the time, Supermemory
0.234. So the LoCoMo win is a win on **epistemic discipline (not
fabricating)**, not on recall. That is a real property of our design and
it is worth keeping — but it is not evidence that we retrieve better,
and I am not going to claim it is.

Two further caveats: LoCoMo here is scored by **deterministic
containment**, which understates both systems with a 1.5b reader — e.g.
gold "Transgender woman" vs hypothesis "Caroline identifies as a trans
woman" is marked **wrong**. And Supermemory ingested 19 of 19 sessions
with 4 documents whose memory agent exceeded their internal budget
(0 memories), so its 0.111 is a floor, not its ceiling.

I verified the abstention wins are real and not an artifact of the
marker-based `is_abstention` heuristic: the counted ContextMemory
answers are genuine declines ("Melanie is not mentioned in the provided
information…"). Both systems still fabricate on ~20 of the 45
adversarial items; we simply decline 25 times where they decline 9.

### dims (synthetic: write precision, evolution, forgetting)

| System | write-precision | evolution | forgetting |
|---|---|---|---|
| contextmemory | **1.000** | 0.400 | 0.333 |
| supermemory (self-hosted) | **1.000** | **1.000** | 0.333 |
| full-history (reference) | 0.600 | 0.800 | **1.000** |

Supermemory beat us on evolution 1.00 vs 0.40; we tied on write precision.

### bench (deterministic latency, same synthetic workload)

| System | ingest p50 | ingest p95 | answer p50 | answer p95 |
|---|---|---|---|---|
| contextmemory | **0.045 ms** | 0.081 ms | **0.159 ms** | 0.224 ms |
| supermemory (self-hosted) | **131 987 ms** | 340 346 ms | 35.4 ms | 43.0 ms |
| full-history | 0.000 ms | 0.000 ms | 0.025 ms | 0.056 ms |

This is the one dimension where the gap is not close and is not
model-dependent: our read path calls no model and is bounded; theirs
pays a full LLM pipeline (chunk → embed → memory-agent dream) on the
write path and a real search round-trip on the read path. ~3 000× on
ingest, ~220× on answer.

## Judgement — the honest read

**Mixed. We win on abstention discipline and latency. We lose on the
headline accuracy benchmark. Nobody wins against full-context.**

1. **LoCoMo: we win 0.186 vs 0.111 — but the win is abstention, not
   recall.** Decomposed above: multi-hop QA is a tie (0.066 vs 0.072).
2. **LongMemEval: we lose 0.083 vs 0.167.** Supermemory wins even with a
   CPU-crippled config. This is the headline task and we fail it.
3. **The most damning number is not a head-to-head at all:** on LME,
   simply putting the transcript in context (`full-history`) scores
   **0.667** — 4-8× both memory layers. On a 1.5b reader, "no memory at
   all" is a very strong baseline. That result damns *both* memory
   systems and should temper every claim either of us makes.
4. **The CPU-only self-hosting handicap is real and asymmetric.**
   Supermemory's memory agent has an internal ~270s budget; on CPU it
   timed out on large documents and their own cron later recovered some
   (0 → 15 memories observed). Their published "#1" numbers come from a
   cloud tier with faster extraction models. Their 0.167 / 0.111 are
   floors. We do not get a comparable excuse: our failures are ours.
5. **Latency is our one unambiguous, large, architecture-level win**
   (0.045 ms vs 132 s ingest; 0.159 ms vs 35 ms answer) — a property of
   the design, not the rig.
6. **Write precision is a tie (1.00 both)**; **evolution is a loss**
   (0.40 vs 1.00).

## Root cause of our LME failure (the actionable part)

The dominant failure is **lossy extraction, not retrieval**. Evidence,
reproduced directly against the store:

For LME instance `6a1eabeb` ("What was my personal best time in the
charity 5K run?"), the haystack session states verbatim:

> "I recently set a personal best time in a charity 5K run with a time of
> 27:12."

Running our own extractor on that exact session (7B, temperature 0,
JSON mode, 8192 ctx) returns **3 cells, none of them the personal best** —
it kept three *tennis* facts and dropped the 27:12. The read path then
had nothing to retrieve, so the system abstained. That single trace
explains the bulk of the 0.083.

Secondary defects visible in the same run:

* **Evolution failure mode:** we answered "The user works at Acme" for a
  question asked *after* the Globex move (dims evolution 0.40). Our
  projection returned a stale value for an unqualified present-tense
  question. Supermemory got this right (1.00).
* **Temporal errors:** wrong dates surfaced (guitar service "June 5th
  2023" for a gold of "Main St."; Nordstrom sale "3 weeks ago" for gold
  "2"). We are over-attributing dates on retrieval.
* **Reader limits:** several gold answers are *behavioural preferences*
  ("the user would prefer responses tailored to Premiere Pro"). A 1.5b
  reader cannot express that even when the memory is present.

## What I would fix first (in order)

1. **Extraction recall on long sessions** — the 7B one-shot extractor
   drops the single most salient fact in a 12-turn session. Fix by
   (a) chunking long sessions before extraction, (b) adding a
   recall-oriented pass, or (c) a "must-capture numerics/dates" rule.
   This is the highest-value change in the repo.
2. **Present-tense qualification on evolution** — unqualified "where does
   the user work?" must resolve to the *current* projection; we answered
   with a superseded value.
3. **Tighten date attribution** — stop inventing dates not present in the
   evidence span.

## Reproduce

```bash
# self-hosted Supermemory (CPU-only, local models)
OPENAI_BASE_URL=http://localhost:11435/v1 OPENAI_API_KEY=ollama \
OPENAI_MODEL=qwen2.5:7b OPENAI_FAST_MODEL=qwen2.5:1.5b \
SUPERMEMORY_DATA_DIR=/tmp/sm ./supermemory-server
# key it prints on first boot → SUPERMEMORY_API_KEY, plus SUPERMEMORY_BASE_URL

python scripts/cmbench.py --model qwen2.5:1.5b --extract-model qwen2.5:7b \
  --systems contextmemory,supermemory,full-history \
  --suites dims,bench,longmemeval --n 12 --bench-sessions 30 --judge --yes
```

Artifacts: `reports/runs/cmbench-20260929-013237/REPORT.md` (dims + bench
+ LME), `benchmarks/results/longmemeval_20260929-091705_contextmemory.jsonl`
(+ `_rescored.json`), `longmemeval_20260929-113247.json` (supermemory),
`longmemeval_20260929-133817.json` (full-history),
`locomo_20260929-141641.jsonl` (contextmemory),
`locomo_20260929-133709.jsonl` (supermemory).

## Caveats that travel with these numbers

* Same rig, same reader, same judge, same write-path model — or the
  numbers mean nothing. Do not compare with vendor self-reports.
* 12 LME instances (stratified) and 1 LoCoMo conversation — small
  samples; treat as directional, not significance.
* CPU-only. Cloud hosts with a fast GPU would raise both systems'
  absolute scores and would change their relative ingest latency far
  more than ours.
* The proxy (`benchmarks/ollama_proxy.py`) and adapter
  (`benchmarks/adapters/supermemory_adapter.py`) were extended to make
  this run possible at all: the proxy now passes `tool_calls` through
  (Supermemory's memory agent is tool-driven and silently produced zero
  memories without it) and disables Qwen3 thinking; the adapter supports
  the self-hosted `base_url` and treats self-hosted ingest failures as
  soft (counted, not fatal), so a flaky CPU extraction step degrades the
  score instead of aborting the contender.
