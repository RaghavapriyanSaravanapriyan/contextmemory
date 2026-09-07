# Production Hardening — ETMC Core v2 (2026-09-07)

How the memory engine went from research prototype to a shippable,
bulletproof local-first layer. Companion to the
`2026-08-29-graph-temporal-core.md` report; this document is current.

## What changed and why

| Area | Before | After |
| --- | --- | --- |
| Journal writes | `trunc` on live path (crash = empty journal) | tmp + rename (atomic commit) |
| Journal CRC | payload only (flipped type byte mis-deserializes) | type + payload, old journals still load (compat fallback) |
| Journal load | cleared live state, then threw (corruption = data loss) | parse-to-temp, validate, then swap (transactional) |
| Enum values | unchecked `static_cast` from journal/FFI (UB shift) | range-validated at journal load + nanobind boundary |
| Vector dims | mismatched dims stored, later OOB read | dimension lock, reject on mismatch |
| Version chains | unbounded `while` walks (crafted cycle = hang) | depth cap (128) + visited set on every walk |
| Projection | late event rewinds current truth | out-of-order links as history, never rewinds |
| Ranking | score-only sort (ties nondeterministic) | score desc, cell id asc everywhere (BM25, dense, RRF, profile) |
| Time routing | substring match (`was` in `wash`, `old` in `gold`) | token word-boundary match |
| Packing | over-budget best cell silently dropped | best cell always returned, overrun reported in `tokens` |
| Profile recency | `float(observed_at)` (all recent cells tie) | int64-exact `observed_at` sort, rank scores |
| Concurrency | no locking, dangling interior pointers | per-Store recursive mutex, documented pointer lifetimes |
| Python API | `journal_path=None` meant "default" (docstring lied) | sentinel: omitted = default, `None` = in-memory |
| Config | unknown keys crashed startup | filtered; atomic save |
| Scoring | `"yes" in text` (`eyes` = correct) | word-boundary parse; judge flakes mark unjudged, not abort |
| Metrics | hardcoded p50/p95 constants | measured from real query events, null when no data |

## Invariants (the contract)

1. **Episodes are immutable.** Raw evidence is append-only; reconcile never
   edits episodes.
2. **Cells carry two clocks.** `observed_at` (when learned) gates knowledge;
   `valid_from/until` (event time) gates truth. A cell is usable at `t` only
   when observed by `t` and its window covers `t`.
3. **Projections point at newest-known truth.** Late-arriving older events
   become `Related` history; the projection never rewinds.
4. **Read path is LLM-free and bounded.** Compile → channels → RRF fuse →
   rerank → token-budget pack. No model call, deterministic output.
5. **Insufficient evidence abstains.** `pack.sufficient == false` means the
   reader is told to say it does not know.
6. **Persistence is atomic and migratable.** Crash-safe commit; old journal
   format loads and re-saves forward.

## Layout

```text
core/                  C++20 ETMC engine (no third-party deps)
  include/cmcore/      types.hpp (model + enum validation)
                       index.hpp (BM25 + dense vector index)
                       store.hpp (Store: capture/reconcile/search/pack/journal)
  src/                 store.cpp (temporal versioning, query compiler, RRF)
                       index.cpp (Okapi BM25, AVX2 cosine)
  python/bindings.cpp  nanobind surface (validated enums, GIL-friendly)
  tests/test_core.cpp  15 dependency-free tests (65 checks, ASan/UBSan clean)
contextmemory/         Python orchestration (never reimplements retrieval)
  api.py               MemoryClient: add/session/search/recall/ask/profile/…
  engine/memory.py     MemoryEngine: capture→extract→reconcile→embed→persist
  engine/extractor.py  NullExtractor (deterministic) | LLMExtractor (1 call)
  engine/embedder.py   DeterministicHashEmbedder (no weights, reproducible)
  core.py              typed facade over the C++ store
```

## Performance (measured, deterministic path, no model)

`contextmemory bench --system contextmemory` (200 sessions, this rig):

- ingest p50 ~0.05 ms, answer p50 ~0.15–0.17 ms — sub-millisecond,
  model-free. LLM extraction dominates wall-clock when enabled (one
  completion per session by design), which is why the benchmark rig
  reports ingest/answer separately from extraction.
