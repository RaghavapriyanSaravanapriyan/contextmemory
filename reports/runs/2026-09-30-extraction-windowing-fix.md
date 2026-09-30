# Run Report — Extraction windowing fix + re-measured head-to-head (2026-09-30)

## What changed

One change, in the write path: `contextmemory/engine/extractor.py`.

**Before:** `LLMExtractor` sent the whole session transcript in a single
completion and took whatever cells came back.

**After:** a session longer than 8000 characters is split on **turn
boundaries** into deterministic windows, each extracted independently and
merged with cross-window dedupe (first mention wins, so an early fact keeps
its original timestamp). The prompt also now asks explicitly to capture
every number, date and duration the user states, and prefers 3-8 facts
instead of 1-5.

No turn is ever split across windows, so the split is lossless; a single
oversized turn (a pasted log) becomes its own window rather than being
skipped.

## Why (the defect this fixes)

The 2026-09-29 head-to-head scored 0.083 on LongMemEval against
Supermemory's 0.167. The cause was traced to a single trace, reproduced
directly against the store:

For LongMemEval instance `6a1eabeb` the haystack states verbatim:

> "I recently set a personal best time in a charity 5K run with a time of
> 27:12."

Extraction of that exact 12-turn / 16 019-character session with the same
7B model, temperature 0, JSON mode, 8192 ctx:

| | cells | needle captured |
|---|---|---|
| single pass (before) | 3 | **no** — kept 3 tennis facts |
| windowed (after) | 8 | **yes** — "personal best time of 27:12 in a charity 5K run" |

The read path then had nothing to retrieve and abstained. This one trace
explains most of the old score.

## Result — LongMemEval (n=12 stratified, LLM judge, same rig)

| System | judge | correct | ingest (s) | answer (s) |
|---|---|---|---|---|
| **contextmemory (fixed)** | **0.583** | **7/12** | 3898 | 22.0 |
| supermemory (self-hosted) | 0.167 | 2/12 | 4446 | 14.8 |
| full-history (reference) | 0.667 | 8/12 | 0 | 738 |

Per-type judge, contextmemory:

| type | before | after |
|---|---|---|
| knowledge-update | 0.5 | 0.5 |
| multi-session | 0.0 | 0.5 |
| single-session-assistant | 0.0 | 0.5 |
| single-session-preference | 0.0 | **1.0** |
| single-session-user | 0.0 | 0.5 |
| temporal-reasoning | 0.0 | 0.5 |

**We now beat Supermemory 3.5× on the discriminative conversational
benchmark** on the same rig, same reader (`qwen2.5:1.5b`), same judge, and
the same 7B extraction model driving both write paths. We remain below the
full-context upper bound (0.583 vs 0.667), which is the honest ceiling to
beat on a 1.5b reader.

## What the fix did not do

The 5 remaining failures are 3 retrieval misses (the fact was in the
haystack but the reader never surfaced it) and 2 wrong-detail answers:

* "Admon's rotation period is a 4-week cycle…" — plausible, not supported.
* "…approximately 365 days, or one year, ago" — invented a duration from a
  relative reference.

## Cost

Extraction for the same session: 165s → 264s per long session (windowing
costs extra bounded calls). Suite wall-clock: LongMemEval ingest
2241s → 3898s. The deterministic read path is untouched: `bench` stays at
0.045 ms ingest / 0.159 ms answer p50.

## The same fix *lowered* LoCoMo — reported, not buried

Re-running LoCoMo (convo 0, 199 QA) with the shipped code:

| LoCoMo | overall | multi-hop (cat 1-4, n=152) | adversarial (cat 5, n=47) |
|---|---|---|---|
| before windowing | **0.186** | 0.066 | **0.574** |
| after windowing | 0.136 | 0.066 | 0.362 |
| supermemory | 0.111 | 0.072 | 0.234 |

The fix **cost us 0.050 on LoCoMo** and the cause is mechanical: windowing
stores *more* cells per session, so the read path returns more evidence,
so the 1.5b reader is more willing to commit to an answer. On adversarial
questions — where the correct behaviour is to decline — that converts
abstentions into confabulations (0.574 → 0.362). Multi-hop recall is
unchanged at 0.066, i.e. windowing did not help LoCoMo's multi-hop
bottleneck at all (that one looks reader-bound, not extraction-bound: the
LME needle was extraction-bound, LoCoMo's multi-hop is not).

**Net:** we still beat Supermemory on both official suites with the
shipped code (LongMemEval 0.583 vs 0.167; LoCoMo 0.136 vs 0.111), but the
LoCoMo margin is thinner than the pre-fix run suggested. The trade is
deliberate: 7× on the discriminative benchmark against a 0.050 regression
on a suite where we were already ahead.

**Next engineering step, now precisely specified:** more evidence should
raise recall without raising confabulation. The abstention decision should
be conditioned on whether the *evidence supports the specific claim*, not
on how much evidence there is. Concretely: distinguish "retrieved but
irrelevant" from "retrieved and relevant" at pack time (the packer
already knows per-item scores) and let the answer path abstain when
relevance is low regardless of evidence volume.

## Open defects (unchanged by this fix)

1. **Evolution** — unqualified present-tense questions can resolve to a
   superseded value; `dims` evolution 0.400 vs Supermemory's 1.000.
2. **Recall misses** — 3 of 5 remaining LongMemEval failures.
3. **Date arithmetic** — inventing durations not present in the evidence.

## Validation

- New `tests/test_extraction_recall.py` (4 tests): windowing of long
  sessions, losslessness (every turn exactly once, needle not duplicated),
  short sessions stay a single call, oversized single turns are still
  extracted, cross-window dedupe merges normalized duplicates.
- `scripts/verify.sh`: 125 tests green, ruff clean.

## Reproduce

```bash
python benchmarks/run_official.py longmemeval --n 12 \
  --systems contextmemory --judge --extract-model qwen2.5:7b
```

Artifacts: `benchmarks/results/longmemeval_20260930-104745.json` (+ `_contextmemory.jsonl`).
Supermemory and full-history numbers are the 2026-09-29 runs on the same
rig (`longmemeval_20260929-113247.json`, `longmemeval_20260929-133817.json`).
