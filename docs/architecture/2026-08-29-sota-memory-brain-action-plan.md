# ContextMemory Brain: SOTA OSS Action Plan

**Date:** 2026-08-29  
**Status:** Proposed implementation plan  
**Audience:** ContextMemory contributors, hackathon team, implementation agents  
**Primary competitor:** Supermemory  
**Primary local stack:** Qwen3-8B + Qwen3-Embedding-0.6B + C++ core

This document is the implementation handoff for building ContextMemory into a
credible open-source state-of-the-art memory layer. It records the research
evidence, the architecture decision, the benchmark strategy, the TUI demo
story, and the failure mitigations we should be able to explain after the
hackathon.

The most important rule is simple:

> Do not claim SOTA until the exact run, model, judge, dataset split, token
> budget, and latency measurement are committed beside the result.

Some numbers below are published claims from other projects, not ContextMemory
measurements. They are targets and comparison points only.

---

## 1. Mission And Winning Thesis

### Mission

Build a memory brain that lets any agent remember years of interaction while
using less context and less time than today's memory layers.

### Winning thesis

Current systems usually optimize one or two of these properties:

- **Accuracy:** retrieve enough information to answer difficult questions.
- **Token cost:** compress or filter the history before the answer model sees it.
- **Latency:** avoid expensive work on the interactive read path.
- **Trust:** preserve evidence, track changes, and know when not to answer.

ContextMemory should optimize all four by separating responsibilities:

1. A local model performs expensive semantic interpretation **once during
   ingestion**, not on every read.
2. A C++ engine stores immutable evidence and fast temporal projections.
3. A deterministic query compiler chooses the smallest useful search plan.
4. A budgeted evidence packer sends only the minimum sufficient context to the
   answer model.
5. A TUI makes the hidden process visible: facts, updates, graph paths,
   evidence, latency, and token savings.

The practical hackathon claim is not "we invented graphs". It is:

> ContextMemory combines evidence-preserving temporal versioning, query-aware
> graph routing, and minimum-sufficient evidence packing into a local-first
> memory engine that is measurable end to end.

That composition is the proposed novelty. We must validate whether the
composition actually produces a better quality/latency/token frontier.

---

## 2. Research Snapshot

### 2.1 Supermemory: the benchmark target

Supermemory's current LongMemEval-S report claims, using its GPT-4o setup:

- **95% overall** at Recall@15 with aggregation.
- Approximately **720 mean context tokens** added per query.
- Category scores of 97% single-session user, 100% assistant, 90%
  preference, 99% knowledge update, 91% temporal reasoning, and 93%
  multi-session.
- A design based on contextual atomic memories, raw source chunks,
  relational versioning (`updates`, `extends`, `derives`), document/event dates,
  and hybrid search.

The report is vendor-published and the headline number is not directly
comparable to a Qwen3-8B run. Supermemory is still the correct product target
because its operating point is strong: high accuracy, very small context, and
production-oriented retrieval.

**What we must beat:** its quality/token/latency frontier under a matched
configuration, not merely its marketing number under a different model.

### 2.2 Current published landscape

| System | Published result or signal | Lesson |
|---|---|---|
| Supermemory | 95% LongMemEval-S with about 720 added tokens in its GPT-4o report | Atomic memories, source fallback, versioning, temporal grounding |
| Mastra Observational Memory | 84.23% with GPT-4o; 94.87% with GPT-5-mini | Stable observation prefix, background Observer/Reflector, prompt-cache friendliness |
| Mem0 | 94.4% LongMemEval and 92.5% LoCoMo in its 2026 report, about 6.8K tokens/query | Single-pass extraction, entity linking, multi-signal retrieval; benchmark setup is vendor-specific |
| Hindsight | 91.4% LongMemEval with Gemini-3 Pro; 83.6% with OSS-20B | Separate world, experience, observation, and opinion networks; confidence and reflection |
| SwiftMem | 10.8ms LoCoMo search in its v2 paper, with competitive quality | Route to a relevant temporal/tag region before dense retrieval |
| SimpleMem | Qwen3-8B LoCoMo average F1 33.45 with 621 tokens in its table | Semantic compression, online synthesis, intent-aware retrieval; immediate local comparison opportunity |
| MAGMA | Multi-graph architecture with semantic, temporal, causal, and entity graphs | Policy-guided traversal makes retrieval intent explicit |
| EverMemOS | MemCells -> MemScenes -> reconstructive recollection | Memory should have a lifecycle, not only an append-only index |
| MemoryOS | Short-, mid-, and long-term tiers with heat-based movement | Working-set ideas can control context and storage growth |
| LIGHT / BEAM | Episodic, working, and scratchpad memories; BEAM reaches up to 10M tokens | Long-context memory needs multiple complementary representations |
| A-MEM | Zettelkasten-style notes, links, and memory evolution | Associative links can be useful without a fixed ontology |

### 2.3 Why Qwen3-8B is the right local backbone

The official Qwen3-8B model card describes an 8.2B-parameter Apache-2.0 model
with 32,768 native context tokens and validated 131,072-token context using
YaRN. It supports thinking and non-thinking modes. The official guidance is:

- Use **non-thinking mode** for fast extraction and routine answers.
- Use **thinking mode** only for explicitly difficult reasoning or an optional
  diagnostic path.
- Use the official chat template and a recent runtime.

Qwen3 is supported through Ollama, llama.cpp, vLLM, and other OpenAI-compatible
servers. Ollama supports JSON schema structured outputs, which is useful for
the extraction contract. For a reproducible local benchmark, use one pinned
Qwen3-8B quantization and record the runtime, quantization, hardware, context
length, and generation parameters.

### 2.4 Benchmark reality

LongMemEval-S is still useful for historical conversational memory. LongMemEval
V2 adds a more realistic agent setting with 451 questions, up to 500
trajectories, and static-state, dynamic-state, workflow, gotcha, and premise
awareness abilities. MemoryArena tests interdependent Memory-Agent-Environment
loops instead of passive question answering. BEAM tests coherent conversations
up to 10M tokens and ten memory abilities.

This creates three distinct tracks:

1. **Conversational recall:** LongMemEval-S and LoCoMo.
2. **Extreme context efficiency:** BEAM.
3. **Agent usefulness:** LongMemEval-V2 and MemoryArena.

For the hackathon, build and publish the first track, include a small
agent-like stress suite from the second/third track, then expand after the
demo. A credible result on one hard public track is better than five
unreproducible claims.

---

## 3. Proposed Architecture: Event-Sourced Temporal Memory Compiler

Working name: **ETMC**, the Event-Sourced Temporal Memory Compiler.

The name describes the key design choice. Raw interaction is treated as an
immutable event stream. The system compiles that stream into progressively more
useful representations, but never destroys the evidence required to recover or
audit a decision.

```text
                    agent conversation / tool trace
                                  |
                     FAST CAPTURE | < 1 ms, no LLM
             immutable episode + hashes + timestamps + source spans
                                  |
              LOCAL ENCODER      | one Qwen3-8B / session, async
             facts + events + entities + tags + temporal anchors
                                  |
              RECONCILER         | deterministic first, model only if ambiguous
        version chains + validity windows + causal/entity/semantic links
                                  |
             C++ PROJECTIONS     | indexes built on write
     current state | timeline | tag router | lexical | dense | graph adjacency
                                  |
              QUERY COMPILER     | no LLM on normal read path
       intent + time + entity + complexity -> bounded retrieval plan
                                  |
           EVIDENCE PACKER       | minimum sufficient context under budget
       current facts + old evidence when requested + provenance + confidence
                                  |
             answer model        | Qwen3-8B or any OpenAI-compatible reader
```

### 3.1 Three representations, not one

Every useful interaction can exist in three linked forms:

1. **Episode:** immutable raw conversation or tool trace. This protects exact
   details, auditability, and recovery from lossy extraction.
2. **Memory cell:** compact self-contained facts/events with entities,
   timestamps, type, confidence, source spans, and tags. This is the normal
   retrieval unit.
3. **State projection:** the current answer to an entity/attribute question,
   such as `user.location = Seattle`, pointing back to the active memory cell
   and its history chain.

This intersects Supermemory's atomic memories and raw chunk fallback, TriMem's
multi-granularity idea, EverMemOS's cells/scenes, and event-sourced database
projections. The important behavior is that compaction can reduce the active
working set without making the system forget where a claim came from.

### 3.2 Memory cell schema

The first implementation should use a compact, explicit schema:

```text
MemoryCell
  id                 stable uint64
  container_id       hard tenant/agent isolation
  subject            canonical entity id or user id
  predicate          normalized attribute or relation, optional
  object             normalized value, optional
  text               self-contained natural-language statement
  kind               world | preference | opinion | experience | procedure
  source_episode     immutable episode id
  source_spans       character/token offsets into source episode
  observed_at        ingestion timestamp
  valid_from         event validity start
  valid_until        event validity end, open when current
  status             active | superseded | expired | forgotten | disputed
  confidence         extraction/evidence confidence, not truth probability
  salience           stable importance estimate
  access_heat        retrieval/use reinforcement signal
  root_id            original version-chain root
  parent_id          prior version, if this is an update
  tags               bounded semantic routing tags
```

Relations are typed and append-only:

```text
updates  old claim -> replacement claim
extends  new detail -> claim it enriches
derives  inferred claim -> supporting claims
causal   cause -> effect
related  associative link
mentions cell -> entity
```

Do not store only free-form text. The `subject` and `predicate` fields make
obvious updates deterministic. For example, a new `user/location` value can
close the old validity window without asking an LLM to compare every memory in
the database.

### 3.3 Bi-temporal truth model

Every cell has two clocks:

- **Event time:** when the statement was true in the world.
- **System time:** when ContextMemory observed and committed the statement.

This distinguishes:

- "I moved to Seattle last month" said today.
- "I lived in New York last year" said last year.
- A correction received late or out of order.

Current-state search uses event validity and active projection pointers. A
historical query uses the same graph with an `at_time` constraint. Superseded
cells remain searchable only when the query asks for history or when audit
mode is enabled.

### 3.4 Lifecycle: capture, encode, reconcile, consolidate

#### Capture: synchronous and cheap

Immediately append:

- raw turns or tool events;
- source offsets and session metadata;
- event/system timestamps;
- content hash for exact duplicate detection;
- lexical tokens for emergency retrieval.

The user-facing write operation returns after capture. No LLM should block the
interactive agent merely to decide whether a message is memorable.

#### Encode: one local model call per session

Qwen3-8B runs in non-thinking mode with a strict JSON schema. It emits:

- 0 to 5 coarse memory cells for a normal conversation;
- explicit entities and subject/predicate candidates;
- event time anchors, including relative date normalization;
- 3 to 8 routing tags;
- evidence spans into the episode;
- salience and extraction confidence;
- optional causal/procedural relations.

The prompt must prefer an empty list over invented facts. It must distinguish
user statements, assistant claims, tool observations, plans, and speculation.

#### Reconcile: deterministic before generative

For each new cell:

1. Exact hash check: duplicate -> no-op.
2. Same `(subject, predicate)` and non-overlapping value -> create update
   version and close the old validity window.
3. Same value with new detail -> `extends`.
4. Strong lexical/entity match but unclear semantics -> add a candidate pair.
5. Only then, optionally ask Qwen3 to adjudicate the small candidate set.

This avoids the expensive Graphiti-style per-entity/per-edge LLM fan-out while
keeping an ambiguity escape hatch.

#### Consolidate: background and measurable

Consolidation runs after a cell threshold, at idle, or when a query observes
fragmentation. It may:

- merge duplicate cells while preserving source links;
- synthesize a scene/profile projection from cells;
- calculate access heat and salience;
- expire planned events after their validity window;
- mark low-value episodic details cold;
- create causal/procedural links from repeated successful traces.

Consolidation must be non-destructive. A compact projection can replace a set
of hot cells in the active index, but the immutable episode and version chain
remain recoverable.

---

## 4. Query Compiler And Hybrid Tuning

The read path is the main differentiator. Normal recall must not call an LLM
before the answer model. A query should compile into a bounded plan in C++.

### 4.1 Query plan

```text
QueryPlan
  time_mode       current | historical | interval | relative | none
  time_start      optional epoch ms
  time_end        optional epoch ms
  entity_seeds    canonical entity ids
  relation_mode   direct | multi_hop | causal | procedural | none
  memory_kinds    allowed kinds
  tags            routed semantic tags
  candidate_cap   4 | 8 | 16 | 32
  expansion_cap   0 | 1 | 2 hops
  evidence_budget  target tokens, hard maximum
  fallback        enabled unless query is exact/current
```

### 4.2 Cheap intent routing

Use a cascade, not a read-time LLM:

1. Date parser and temporal phrase lexicon: `before`, `now`, `last week`,
   `when`, explicit dates, `used to`, `currently`.
2. Entity matcher over exact aliases and lexical candidates.
3. Predicate/topic classifier using compact keyword features.
4. Query embedding to route to semantic tags.
5. Complexity estimate from question grammar and number of entities/constraints.

The classifier should expose its decision to the TUI and logs. A wrong route
must be diagnosable, not hidden inside a black-box ranking score.

### 4.3 Candidate channels

Run the channels in parallel over the routed subset:

- lexical BM25 for names, dates, numbers, and exact terms;
- dense cosine for paraphrases and open-domain questions;
- entity adjacency for shared people, organizations, locations, and projects;
- temporal interval lookup for explicit or inferred dates;
- relation traversal for multi-hop and causal queries;
- current-state projection for direct `now/current` questions;
- recency/access heat as a tie-breaker, never as truth.

Fuse with rank-based Reciprocal Rank Fusion first. Rank fusion is more stable
than pretending raw BM25, cosine, and graph scores share a calibrated scale.
Then apply a small, interpretable rerank over features:

```text
final_score =
    route_weight[type].lexical       * lexical_rank_score +
    route_weight[type].dense         * dense_rank_score +
    route_weight[type].entity        * entity_score +
    route_weight[type].temporal      * temporal_score +
    route_weight[type].relation      * relation_score +
    route_weight[type].current       * current_state_score +
    route_weight[type].evidence      * evidence_coverage_score
```

The weights are not hand-tuned forever. Tune them offline on a development
split with a latency/token constraint, then freeze the configuration for the
test split.

### 4.4 Query-aware search region

Use SwiftMem's key idea, combined with MAGMA's relational views:

1. Resolve top semantic tags from a compact tag index.
2. Expand only a bounded tag frontier.
3. Intersect with temporal interval and container scope.
4. Seed from exact entities and current-state projections.
5. Search dense vectors only inside the remaining candidate set.
6. Traverse graph relations only when query complexity requires it.

This avoids both extremes:

- searching every vector for every query;
- forcing every query through an expensive graph walk.

Use a recall-preserving fallback: if the candidate set is empty or evidence
coverage is low, widen one level and record that fallback. Never silently
return an overconfident answer from an empty route.

### 4.5 Minimum-sufficient evidence packing

This is the token-efficiency mechanism.

Given ranked candidate cells and a target budget, select the smallest set that
covers the query's required slots:

```text
maximize    answer_evidence_coverage(cells)
then        confidence + temporal_fit + source_quality
subject to  token_count(cells) <= budget
```

Use a greedy weighted set-cover/knapsack approximation in C++. The packer
should prefer:

1. a current state cell for direct current questions;
2. a prior version when the question asks `before`, `used to`, or a date;
3. a second supporting cell for multi-hop/causal questions;
4. the shortest source span that resolves ambiguity;
5. the full episode only as a trace/debug fallback.

The answer prompt receives compact evidence with stable IDs:

```text
[M17 | current | valid 2025-06-01..open | source S44]
User lives in Seattle.

[M09 | superseded | valid 2022-01-01..2025-05-31 | source S12]
User lived in New York.
```

The answer model must cite the memory IDs internally or in a debug channel and
must say that the information is insufficient when coverage is below a
threshold. This supports both trust and a TUI explanation of why an answer was
made.

### 4.6 Hybrid tuning protocol

Create a development split from each benchmark and tune only on that split:

1. Establish lexical-only, dense-only, graph-only, and fixed-fusion baselines.
2. Tune route weights separately for single-hop, temporal, update,
   preference, and multi-session query types.
3. Sweep candidate caps `{4, 8, 16, 32}` and token budgets `{256, 384, 512,
   700}`.
4. Optimize quality subject to p95 retrieval latency and token budget.
5. Freeze weights and run the untouched test split once.
6. Publish the full config and ablation table.

Required ablations:

- no temporal index;
- no graph edges;
- no query routing;
- no current-state projection;
- no source-span fallback;
- fixed 700-token packer;
- minimum-sufficient packer;
- no consolidation;
- raw chunk baseline;
- SimpleMem-style compressed cell baseline.

The result should answer which component creates the win. If the score only
improves after many benchmark-specific rules, call it overfitting and remove
them.

---

## 5. C++ And Python Implementation Plan

The existing repository already has a dependency-free C++ store, nanobind
bindings, Python extraction/embedding interfaces, and an evaluation harness.
The next coding agent should evolve those pieces rather than replace them.

### Phase A: make the core a real engine

**Files:** `core/include/cmcore/`, `core/src/`, `core/tests/`

Implement or verify:

- `MemoryCell`, `Episode`, `Entity`, `Edge`, and `QueryPlan` types;
- append-only journal records for episodes, cells, updates, edges, and
  tombstones;
- exact duplicate hashes;
- bi-temporal validity and out-of-order event handling;
- current-state projection keyed by `(container, subject, predicate)`;
- per-container isolation;
- sorted temporal index;
- tag-to-cell inverted index;
- BM25 and vector indexes over cells, not only raw episodes;
- bounded entity/causal/semantic adjacency;
- candidate caps and token-budget evidence packing;
- deterministic result traces for the TUI and benchmark JSONL.

Acceptance tests:

- update New York -> Seattle; current query returns Seattle;
- historical query at 2024 returns New York;
- late-arriving event is placed by event time but recorded by system time;
- duplicate ingestion creates no second cell;
- superseded cells are absent from current search and available in audit mode;
- a tenant cannot retrieve another tenant's cells;
- token budget is never exceeded;
- graph expansion obeys hop and candidate caps;
- journal save/load preserves all IDs, windows, edges, and source links.

### Phase B: production-quality local encoder

**Files:** `src/contextmemory/engine/`

Implement:

- JSON schema extraction with Qwen3 non-thinking mode;
- Ollama native structured-output adapter using `format` JSON schema;
- OpenAI-compatible adapter for llama.cpp/vLLM/LM Studio;
- robust JSON parsing and validation without a repair LLM call;
- relative date normalization with explicit reference date;
- evidence span validation against the source episode;
- bounded candidate context for ambiguous reconciliation;
- async/background ingestion queue with backpressure;
- extraction telemetry: calls, prompt tokens, output tokens, failures,
  cells/session, milliseconds/session.

Extraction must not invent a fact when the source span cannot support it. A
failed or malformed extraction should preserve the raw episode and increment a
visible failure counter.

### Phase C: benchmark adapters

**Files:** `src/contextmemory/eval/`, `benchmarks/`, `reports/runs/`

Add a common provider adapter interface:

```python
class BenchmarkProvider(Protocol):
    def reset(self, container_id: str) -> None: ...
    def ingest(self, session: Session) -> None: ...
    def retrieve(self, question: str, question_date: datetime) -> Retrieval: ...
    def answer_context(self, retrieval: Retrieval) -> list[Message]: ...
    def telemetry(self) -> dict[str, object]: ...
```

Adapters to implement in this order:

1. ContextMemory full engine.
2. Full-context baseline.
3. Naive BM25/dense RAG baseline.
4. SimpleMem-like compressed-cell baseline.
5. Mem0 OSS, pinned commit/config.
6. Zep/Graphiti, pinned version/config.
7. Supermemory API or local binary adapter if access is available.

The answer model and judge must be configurable independently. Record both
actor and judge model IDs in every run.

### Phase D: local model harness

Recommended local setup:

```text
Qwen3-8B Q4_K_M or Q5_K_M       actor and extraction model
Qwen3-Embedding-0.6B            local dense embeddings
Ollama or llama.cpp              OpenAI-compatible local server
ContextMemory C++ core           retrieval, graph, temporal store
Qwen3 non-thinking               extraction and ordinary answer generation
Qwen3 thinking                   optional difficult-query diagnostic only
```

For strict published comparisons, use the official benchmark judge where
allowed and run every provider through the same judge. For a fully local track,
use the same local judge for every provider and label the results as a local
matched track, not as directly equivalent to vendor GPT-4o numbers.

### Phase E: TUI

Use Textual and Rich for the first frontend. Textual supports interactive
terminal and browser execution, reactive state, widgets, async workers, and
SSH-friendly deployment. Keep the engine and TUI separate so the benchmark can
run headless.

Suggested module layout:

```text
src/contextmemory/tui/
  app.py              Textual application and screen routing
  state.py            reactive demo state
  widgets.py          timeline, graph, evidence, metrics widgets
  scenarios.py        deterministic hackathon scenarios
  styles.tcss         visual system
```

TUI screens:

- **Live Brain:** conversation input, extracted cells, current profile;
- **Timeline:** event dates, validity windows, updates, superseded facts;
- **Why This Answer:** selected evidence, graph path, confidence, fallback;
- **Bench Race:** ContextMemory vs full context / RAG bars for tokens,
  retrieval ms, end-to-end ms, and correctness;
- **Memory Health:** cell count, active/superseded ratio, bytes, extraction
  failures, cache hits, and tenant/container.

Every widget should consume an engine event/trace object. Do not scrape debug
strings to render the UI.

---

## 6. Benchmark Plan And Honest Win Conditions

### 6.1 Primary hackathon benchmark

Start with **LoCoMo under a Qwen3-8B matched local configuration** because the
published SimpleMem table provides an immediate local reference point:

- SimpleMem Qwen3-8B: average F1 33.45, 621 reported tokens.
- Mem0 Qwen3-8B: average F1 25.80, 1,015 reported tokens.

These are reported paper values, not a replacement for our rerun. The target
for ContextMemory is:

```text
average F1 >= 35.0 on the same selected LoCoMo split
mean context <= 500 tokens
p50 retrieval <= 15 ms locally
one extraction call/session, zero read-path routing calls
```

If the full 1,540-question run cannot fit the hackathon window, publish a
pre-registered 10-conversation development slice and clearly call it a slice.
Do not call a slice the full benchmark.

### 6.2 Secondary benchmark

Run **LongMemEval-S** against:

- full-context Qwen3-8B;
- naive dense/BM25 RAG with the same answer model;
- ContextMemory;
- SimpleMem or Mem0 if an adapter is reproducible locally.

Target a Pareto win rather than an ungrounded absolute claim:

```text
ContextMemory accuracy >= best matched local baseline
and mean context <= 50% of that baseline
and p50 retrieval <= 25% of that baseline
```

Then optionally run the official judge configuration for a cross-paper
comparison. The report must show actor model, judge model, ingestion model,
dataset hash, prompt templates, top-k, token budget, and latency boundaries.

### 6.3 BEAM stretch goal

Use a small BEAM subset first. BEAM exists specifically to expose the failure
of long contexts at 128K, 500K, 1M, and 10M tokens. The stretch target is:

```text
retain or improve matched-baseline quality while reducing answer context by
at least 10x and keeping retrieval p95 below 50 ms for the local store
```

Do not attempt the 10M-token run until the 128K and 500K runs are stable.

### 6.4 Metrics to publish

Every run reports these separately:

**Quality**

- official accuracy or judge score;
- per-category scores;
- abstention accuracy;
- knowledge-update accuracy;
- temporal accuracy;
- multi-session accuracy;
- evidence-grounded answer rate.

**Efficiency**

- retrieved context tokens: mean, median, p95;
- total actor input/output tokens;
- extraction input/output tokens;
- model calls per session and query;
- stored bytes per raw input token;
- facts/cells per session.

**Latency**

- capture time;
- extraction time;
- reconciliation time;
- index update time;
- retrieval time only;
- evidence packing time;
- answer end-to-end time;
- cold-start and warm-cache p50/p95.

**Integrity**

- unsupported-fact rate;
- duplicate rate;
- update precision and recall;
- stale-answer rate;
- abstention false-positive/false-negative rate;
- cross-container leakage rate;
- provenance coverage.

Do not compress these into one magic score. Show a Pareto plot of quality,
tokens, and latency, plus a separate integrity panel.

### 6.5 Fairness controls

- Pin every dependency and provider commit.
- Use the same actor model, judge model, prompts, temperature, and max output
  tokens for all providers in a comparison.
- Ingest chronologically and reset state for every question instance.
- Warm up local models before timing.
- Separate memory retrieval time from model generation time.
- Record CPU/GPU, RAM/VRAM, quantization, runtime, thread count, and context
  length.
- Tune on development data only; never tune on the final test questions.
- Store raw per-question outputs and retrieval traces.
- Run at least three latency repetitions and report variance.
- Distinguish vendor-reported values from our reproduced values.

---

## 7. The Three-Minute Demo

The demo should show behavior, not architecture slides.

### Scene 1: memory forms

Seed the agent with a few turns:

```text
I live in New York and work at Acme.
I usually use Vim and prefer concise answers.
I am planning a hiking trip in Colorado.
```

The TUI animates three cards appearing:

- stable preference: concise answers;
- current state: Acme / New York;
- episode: Colorado hiking plan.

The side panel shows one extraction call, source spans, and stored bytes.

### Scene 2: contradiction becomes history, not corruption

Add:

```text
I moved to Seattle and joined Globex this month.
```

The timeline visibly closes New York/Acme, creates Seattle/Globex, and draws an
`updates` edge. Ask:

```text
Where do I live now?
```

The answer panel shows Seattle, the selected current cell, and sub-millisecond
or low-millisecond retrieval timing.

Then ask:

```text
Where did I live before moving?
```

The answer panel shows New York and the old validity window. This is the
strongest proof that the system understands evolution rather than merely
boosting the newest vector.

### Scene 3: demonstrate token economics

Press `B` for Bench Race. Show the same query through:

- full transcript;
- naive retrieval;
- ContextMemory.

The TUI displays:

```text
full context       100% evidence       slowest
naive RAG          medium evidence      may mix stale facts
ContextMemory      minimum evidence     current + provenance + fast
```

Use actual measured numbers from the current run. Never hard-code a fake
benchmark result into the demo.

### Scene 4: trust and failure

Ask an unanswerable question such as:

```text
What is my passport number?
```

The TUI should show `ABSTAIN`, no fabricated memory cell, and a reason:
`no evidence coverage`. This makes write precision and safety visible in a way
that ordinary benchmark scoreboards do not.

### TUI interaction keys

- `1`: Live Brain
- `2`: Timeline
- `3`: Why This Answer
- `B`: Bench Race
- `F`: inject a fact update
- `A`: ask an abstention probe
- `R`: replay the scripted demo
- `Q`: quit

The scripted demo must run without a network connection using deterministic
fixtures. Live Qwen mode is an optional second path.

---

## 8. Hackathon Execution Order

Do not parallelize everything. Land a visible vertical slice first.

### Track 0: lock the baseline

1. Pin Qwen3 runtime and quantization.
2. Run the existing full-history and naive baselines.
3. Save a run manifest and per-question outputs.
4. Verify token and latency instrumentation boundaries.

**Exit condition:** a baseline result can be reproduced from one command.

### Track 1: make updates undeniable

1. Implement event-sourced cells and current-state projections.
2. Add update/history/late-arrival C++ tests.
3. Add retrieval traces.
4. Build the contradiction demo.

**Exit condition:** current and historical questions both pass in the TUI and
unit tests.

### Track 2: win efficiency

1. Add temporal/tag query routing.
2. Add minimum-sufficient evidence packing.
3. Add per-type fusion tuning.
4. Compare context tokens and p50/p95 retrieval against full context and RAG.

**Exit condition:** a measured Pareto improvement on the local development
slice.

### Track 3: local Qwen encoder

1. Add Ollama/llama.cpp structured JSON adapter.
2. Run non-thinking extraction.
3. Validate spans and timestamps.
4. Measure extraction cost and failure rate.

**Exit condition:** the same session produces the same validated cells across
three warm runs at temperature 0.

### Track 4: TUI polish

1. Implement scripted offline mode.
2. Add timeline and evidence panels.
3. Add live metric cards.
4. Add benchmark race view.
5. Record a short screen capture and rehearse the explanation.

**Exit condition:** a judge understands the problem, sees the update happen,
sees the evidence, and sees measured savings in under three minutes.

### Track 5: publish the result

1. Run the untouched test split.
2. Run ablations.
3. Generate markdown tables and plots from JSONL, not manual edits.
4. Commit the run manifest, report, and exact command.
5. Update README only with verified numbers.

**Exit condition:** another machine can reproduce the headline result.

---

## 9. Failure Modes And Mitigations

This section is part of the project story. A strong hackathon project can
explain not only what worked, but what failed and how it was fixed.

| Failure | Detection | Mitigation |
|---|---|---|
| Qwen emits invalid JSON | schema validation and parse-failure counter | native JSON schema, strict prompt, preserve raw episode, no hallucinated repair |
| Qwen thinks during extraction and becomes slow | output contains reasoning or latency spikes | force non-thinking mode; cap output; use thinking only in offline diagnostics |
| Relative dates are wrong | temporal probe failures and source/date trace | attach session reference date; normalize to ISO/epoch; test `yesterday`, `next month`, and late arrival |
| Extraction invents facts | unsupported source-span rate | require source spans; reject cells without supporting text; abstain when evidence is absent |
| New fact does not supersede old fact | update precision/recall suite | deterministic `(subject,predicate)` reconciliation before semantic fallback |
| Old fact pollutes current answer | stale-answer probe | validity-window filter and current-state projection; old cell only in historical mode |
| Graph grows without bound | edges/cell ratio and traversal visits | typed edges, bounded degree, hop/candidate caps, background compaction |
| Query router misses a relevant region | recall@K and empty-route count | one-step widened fallback; retain lexical global safety net; log route decisions |
| Dense vectors drown exact names | proper-noun and number probes | BM25/entity channel receives a protected minimum rank share |
| Graph traversal adds latency to simple queries | per-channel timings | activate relation traversal only for multi-hop/causal/procedural intent |
| Compression loses exact detail | source-span fallback rate and adversarial detail probes | keep immutable episodes and fetch shortest supporting spans on demand |
| Memory is fast but answer quality drops | judge score and evidence coverage | tune route weights and budget on development split; do not optimize latency alone |
| Benchmark score is not comparable | run manifest audit | matched actor/judge/prompt/config; label vendor numbers and local tracks separately |
| Local model is too slow | extraction tokens/sec and p95 | Q4/Q5 quantization, non-thinking mode, async queue, micro-batched embedding |
| C++ build fails on another machine | clean CI build | CMake >= 3.20, compiler matrix, no required external database, documented fallback |
| TUI freezes during model work | interaction latency and worker status | Textual worker/thread boundary; keep capture and UI event loop non-blocking |
| Demo depends on internet | offline replay run | deterministic fixture mode with the same trace/event interfaces as live mode |
| User data leaks between agents | cross-container test | hard container filter at every core operation, not only in Python |
| Forgetting deletes needed history | historical recovery test | tombstones/expiry in projections, immutable source and version chains retained |

---

## 10. Handoff Checklist For The Coding Agent

The implementation agent should follow this order:

1. Read this document and `AGENTS.md`.
2. Inspect the existing C++ core and Python eval harness before adding modules.
3. Preserve the current public protocols where possible.
4. Implement one vertical path: session -> cells -> update projection -> query
   plan -> evidence trace -> answer.
5. Add a regression test before optimizing.
6. Keep all model calls behind interfaces.
7. Keep normal reads LLM-free before the final answer model.
8. Measure retrieval and answer time separately.
9. Add the offline scripted TUI before adding visual polish.
10. Run `scripts/verify.sh`, C++ tests, and a local smoke benchmark.
11. Write a run report before making a SOTA claim.
12. Commit coherent milestones and push them to GitHub.

The first implementation milestone is complete only when this command works:

```bash
uv run contextmemory demo --offline
```

It must show:

- two versions of a changed fact;
- a current answer and a historical answer;
- selected evidence and provenance;
- context-token and latency counters;
- an abstention example;
- no network or API key required.

The second milestone is complete only when this command produces a reproducible
JSONL result:

```bash
uv run contextmemory eval \
  --data benchmarks/data/longmemeval_s_cleaned.json \
  --system contextmemory \
  --reader-api-base http://localhost:11434/v1 \
  --reader-api-key ollama \
  --reader-model qwen3:8b \
  --out reports/runs/contextmemory-qwen3-lme-s.jsonl
```

The exact CLI may evolve during implementation, but the semantics and
measurement boundaries must remain stable.

---

## 11. Sources And Evidence

Primary or official sources used for this plan:

- [Supermemory LongMemEval research](https://supermemory.ai/research/longmembench)
- [Supermemory source repository](https://github.com/supermemoryai/supermemory)
- [LongMemEval original repository](https://github.com/xiaowu0162/LongMemEval)
- [LongMemEval-V2 repository](https://github.com/xiaowu0162/LongMemEval-V2)
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 technical report](https://arxiv.org/abs/2505.09388)
- [Qwen3 official repository](https://github.com/QwenLM/Qwen3)
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [llama.cpp server](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
- [SimpleMem paper](https://arxiv.org/abs/2601.02553)
- [SimpleMem implementation](https://github.com/aiming-lab/SimpleMem)
- [SwiftMem paper](https://arxiv.org/abs/2601.08160)
- [MAGMA paper](https://arxiv.org/abs/2601.03236)
- [EverMemOS paper](https://arxiv.org/abs/2601.02163)
- [Hindsight paper](https://arxiv.org/abs/2512.12818)
- [Hindsight ACL demonstration](https://aclanthology.org/2026.acl-demo.27/)
- [Zep temporal knowledge graph paper](https://arxiv.org/abs/2501.13956)
- [Graphiti repository](https://github.com/getzep/graphiti)
- [MemoryOS paper](https://arxiv.org/abs/2506.06326)
- [LightMem paper](https://arxiv.org/abs/2510.18866)
- [R3Mem paper](https://arxiv.org/abs/2502.15957)
- [A-MEM paper](https://arxiv.org/abs/2502.12110)
- [BEAM paper](https://arxiv.org/abs/2510.27246)
- [BEAM project page](https://mohammadtavakoli78.github.io/beam-light/)
- [MemoryArena paper](https://arxiv.org/abs/2602.16313)
- [MemoryArena project page](https://memoryarena.github.io/)
- [Mastra Observational Memory research](https://mastra.ai/research/observational-memory)
- [Textual documentation](https://textual.textualize.io/)

The existing ContextMemory synthesis reports remain useful background:

- `reports/research/2026-08-29-frontier-memory-landscape.md`
- `reports/research/2026-08-29-agent-memory-systems-deep-dive.md`
- `reports/architecture/2026-08-29-graph-temporal-core.md`

---

## 12. Definition Of Done

The SOTA OSS milestone is not "the demo looked good". It is:

- the C++ engine passes unit, persistence, isolation, temporal, and budget
  tests;
- the local Qwen extraction path is schema-validated and observable;
- ContextMemory wins a matched quality/token/latency comparison on at least
  one public benchmark;
- the result includes per-question outputs, configs, and an independent
  rerun command;
- ablations show which architectural pieces cause the improvement;
- the TUI demonstrates current-vs-historical truth, provenance, abstention,
  and measured efficiency;
- the project can be installed and run locally without a proprietary database
  or cloud service;
- every unresolved limitation is written down instead of hidden.

That is the path from a cracked hackathon demo to a durable memory brain for
agents everywhere.
