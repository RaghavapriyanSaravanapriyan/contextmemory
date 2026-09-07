# Run Report — Production Hardening + HQ Surface (2026-09-07)

## Objective

Bulletproof the memory layer (bugs, UB, persistence, determinism),
widen the product surface (MCP/HTTP/CLI/TUI), and record honest measured
numbers. No-GPU rig: no LLM-judged benchmark claims; deterministic and
static evidence only.

## What changed

- C++ ETMC core: atomic journal (tmp+rename), type-covered CRC with old
  journal compat, transactional load, enum validation, dim lock,
  chain-walk guards, no-rewind projections, word-boundary routing,
  deterministic sorts, single-best packing, int64 profile recency,
  per-Store locking. `SearchResult.observed_at` plumbed end-to-end.
- Python: in-memory sentinel, config forward-compat + atomic save,
  cross-session dup accounting, persist-failure counter, judge
  word-boundary parse + flake tolerance, MCP/HTTP input validation,
  measured (not hardcoded) server latency, honest TUI bench labels.
- Surface: MCP 4→6 tools (+profile, +timeline), HTTP +3 endpoints
  (+profile, +recall, +forget; 400s on bad input), CLI +3 commands
  (+recall, +profile, +setup), TUI HQ (Profile/Setup/Help panes,
  container switcher, TTL-cached renders, offline remember, empty states).
- Packaging: sdist 203 MB → 157 KB (data/vendor excluded), wheel 294 KB.

## Validation (evidence)

- pytest: 100 passed (`uv run pytest -q`), ruff clean.
- C++: 15/15 tests, 65 checks (`core/tests/test_core.cpp`); fresh Debug
  build; ASan+UBSan clean (LD_PRELOAD libasan).
- `bench --system contextmemory`: ingest p50 ~0.04–0.05 ms,
  answer p50 ~0.15–0.17 ms (200 sessions, deterministic, no model).
- `dims --system full-history` (qwen2.5:1.5b reader): evolution 0.80,
  forgetting 1.00, write-precision 0.60.
- `dims --system contextmemory` + qwen2.5:1.5b: near-zero — root-caused
  to the 1.5B extractor returning 0 cells (NCELLS=0 probe), NOT a
  retrieval regression: with an empty store every probe abstains.
  Requires a capable extraction model (7B+, per 2026-08-29 report @0.92);
  unrunnable on this CPU-only rig in reasonable time. No LLM-judged
  numbers are claimed.
- Wheel + sdist build cleanly (`uv build`).

## Honest comparison vs leading OSS (static, dated 2026-09-07)

No shared-rig run was possible here (no GPU). Feature-level only:

| Capability | ContextMemory | Supermemory (self-host) |
| --- | --- | --- |
| Local-first, one-binary-ish install | `uv sync` + `setup` wizard, wheel 294 KB | single binary :6767 + wizard |
| Model-agnostic / offline | Ollama / OpenAI-compat / offline NullExtractor | Ollama (gpt-oss) / OpenAI / Anthropic / Gemini |
| Isolation primitive | container tag (user/project/agent) | containerTag |
| Versioning | subject/predicate validity windows, no-rewind | Updates/Extends/Derives, isLatest |
| Forgetting | explicit + status model; auto-expiry planned | automatic (time/contradiction/noise) |
| MCP tools | 6 (memory/recall/context/profile/timeline/forget) | memory/search (+plugins for CC/OpenCode/Claw) |
| HTTP API | health/graph/metrics/events/profile/ask/recall/memories/forget | full Memory API + SuperRAG + connectors |
| Profiles | static + dynamic, ms latency | static + dynamic, ~50 ms (self-report) |
| Connectors/RAG/files | not built (out of scope: memory ≠ RAG) | Drive/Gmail/Notion/GitHub, SuperRAG, multimodal |

Where they are ahead: connectors, file processing, hosted cloud,
plugin ecosystem, published benchmark scores. Where we are ahead:
dependency-free C++ core (no vector DB to operate), deterministic
sub-ms model-free read path, journal that survives crashes and
corruption, zero bloat (2 runtime deps).

## Remaining concerns

- Extraction quality gates everything: a weak extractor yields an empty
  store and universal abstention. Mitigations open: extraction
  self-check (cell-count floor → warn), heuristic subject/predicate
  backfill for extractor-less cells, recency prior for Current mode
  without projections.
- No shared-rig head-to-head yet (needs GPU or API budget):
  `benchmarks/run_official.py` (BEAM/LongMemEval/LoCoMo, one rig,
  identical reader) is the instrument; run before any leaderboard claim.
- Auto-forgetting (time-based expiry, contradiction auto-resolve) is
  designed (status model) but not yet scheduled; currently explicit.
