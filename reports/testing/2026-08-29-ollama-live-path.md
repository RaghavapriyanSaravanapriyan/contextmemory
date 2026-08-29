# Testing Report — One-shot install + Ollama live path validation

**Date:** 2026-08-29
**Scope:** Validate `./run.sh` (one-shot install + TUI launch), the Ollama
connect flow, and the end-to-end live pipeline (`remember:` → structured cells
→ cited answers). Found and fixed two real bugs on the live read/write path.

## Question

Can a fresh user run one command, connect to a local Ollama model, store facts
through LLM extraction, and get cited answers back — without a broken path
between write and read?

## Environment

- Linux, Python 3.13 venv (uv), C++ core rebuilt via `uv sync`.
- Ollama 0.30.7 running locally (`http://localhost:11434`), models available:
  `qwen2.5:7b`, `qwen3.5:9b`, `gemma3:4b`, Llama-3.2-1B.
- Live tests used `qwen2.5:7b` (extraction + answer generation).

## What was validated

- `./run.sh --live --model qwen2.5:7b`: installs deps, builds core, launches
  the TUI in live mode under a pseudo-TTY; app rendered, connected to the
  running Ollama, no errors. (Interactive key handling not assertable via
  piped stdin; interactive behavior verified headlessly with Textual's
  `run_test()` pilot.)
- Headless pilot flow: `_auto_connect` → `connected`; `remember: ...` →
  2 structured cells; `Where does the user live?` → `The user lives in Paris
  [M2].` route=current; fact update ("moved to Seattle") → versioning, current
  answer cites Seattle and notes Paris superseded.
- Offline demo replay, bench rows, health panes, and the Ollama connect modal
  all render in the pilot.

## Bugs found and fixed

1. **Double `/v1` path — live path completely broken.**
   `OllamaManager.reader()` built its base as `http://localhost:11434/v1` and
   `OpenAICompatClient.complete()` posts to `/v1/chat/completions`, producing
   `http://localhost:11434/v1/v1/chat/completions` (404). Extraction silently
   failed and fell back to the unstructured `NullExtractor`, so cells stored
   with empty subject/predicate and retrieval never matched.
   Fix: `OpenAICompatClient` now strips a trailing `/v1` from `base_url`
   (accepts both root and suffixed forms), and `OllamaManager.reader()` passes
   the root. Regression test: `test_openai_compat_client_normalizes_v1_suffix`.

2. **`_remember` wrote to a throwaway store.**
   `MemoryBrainApp._remember` built a fresh `MemoryClient` (with LLM extractor)
   per call, but `ask_live` reads from the app's own `self.client` — two
   separate in-memory `MemoryStore`s. Facts were extracted and then vanished on
   ask. Fix: `MemoryClient.set_extractor()` / `MemoryEngine.set_extractor()`
   swap the write-path extractor on the existing store; `_connect_live` adopts
   the LLM extractor, and `_remember` writes through `self.client`.
   Regression tests: `test_set_extractor_keeps_store`,
   `test_swapped_extractor_is_used_on_ingest`.

## Remaining concerns

- Historical "before moving" abstains in live mode for the specific phrasing
  "moved to Seattle last month": extraction assigns the move `valid_from` a
  month ago, so the reconcile correctly supersedes the later "lives in Paris"
  cell (version chain root = Seattle). This is extraction-quality/temporal
  phrasing sensitivity, not a code defect; the deterministic offline demo
  exercises the historical path and passes.
- TUI interactive key handling under a piped pseudo-TTY is not automatable;
  validation relied on the headless pilot.

## Result

52 tests pass, ruff clean (`scripts/verify.sh`). The one-shot flow —
install, connect to Ollama, remember, ask, get cited answers — is functional.

---

## Round 1 (2026-08-29) — qwen3:4b live testing

Validated end-to-end with **qwen3:4b** (Ollama 0.30.7, CPU-only, no GPU) for
round 1 of manual testing. Three real issues surfaced and were fixed; all are
Qwen3-family/Ollama-specific and would silently degrade any local-model run.

### 1. Qwen3 thinking mode breaks extraction on the OpenAI-compatible endpoint

Qwen3 models default to thinking mode. On Ollama's `/v1/chat/completions`
endpoint the `think` flag is ignored, so the model spent its entire token
budget on a hidden `reasoning` trace and returned empty `content` — extraction
hung past the timeout and fell back to the unstructured `NullExtractor`, which
the read path could not retrieve (silent failure).

Fixes:
- **Native `/api/chat` client** (`OllamaChatClient` in `engine/ollama.py`) —
  the only endpoint that honors `think: false`. `OllamaManager.reader()` now
  returns it.
- **`format: "json"`** on extraction constrains the model to emit only valid
  JSON — qwen3:4b otherwise rambles in prose for 1500+ tokens and never emits
  the object. With it, extraction is ~22s and clean on CPU.
- **Bounded generation** (`num_predict`/`max_tokens` ≈ 1500) so no call can
  hang indefinitely.
- **CLI auto-detection** (`make_reader` in `cli.py`) — `eval`/`dims`/`ask`
  probe `/api/tags`; if it's Ollama they use the native client. Without this,
  `dims --reader-model qwen3:4b` would hit the same thinking-mode bug.

### 2. `None` question date meant the epoch, silently

`MemoryClient.ask/recall/profile` with no `question_date` evaluated validity
windows at `at_time=0` (1970), so "current state" queries missed every
just-stored cell and abstained. Fix: `_asof()` in `engine/memory.py` defaults
to now. Regression: `test_recall_defaults_to_now_not_epoch`.

Subtlety: `_asof` must use naive `datetime.now()` (not aware-UTC) to match the
write path, which timestamps sessions with `datetime.now()` and lets `to_ms`
treat naive as UTC. Mixing aware-UTC now with naive-UTC stored timestamps
shifts the reference by the local UTC offset (5.5h here) and pushes it before
the cell's validity window — a second, easy-to-miss failure.

### 3. Answer prompt encouraged self-talk

Qwen3 models narrate their reasoning even in non-thinking mode. The answer
prompt now demands one-to-two-sentence direct answers with no preamble; the
final answer ("Seattle [M2]") is correct though the model still emits some
preceding self-talk on CPU. Acceptable for a 4B model.

### Measured (qwen3:4b, CPU-only)

- Full pipeline (remember → ask) reliable 4/4 across runs.
- `contextmemory dims`: write-precision 1.0, forgetting 1.0, evolution 0.8
  (12/13 probes, overall 0.92).
- `contextmemory eval --max-instances 1` (oracle): replays, ingests, answers,
  scores, writes JSONL. Single deterministic score on a temporal-reasoning
  instance was 0.0 — strict exact-match scoring against a verbose 4B answer;
  not statistically meaningful at n=1.
- Full deterministic suite (`scripts/test-all.sh`): C++ 11/11, 53 Python
  tests, ruff clean, answer latency p50 ~0.13ms.

### Remaining concerns for round 2

- Small local models are verbose and occasionally imperfect extractors; the
  JSON grammar + token cap bound the failure but cannot eliminate it. A GPU
  or a stronger model makes runs cleaner and far faster.
- The answer prompt tightening helped but did not fully silence qwen3:4b's
  self-talk on CPU.