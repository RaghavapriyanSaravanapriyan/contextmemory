# Agent Memory Layer Systems — 2026 Deep Dive

**Date:** 2026-08-29
**Purpose:** Technical deep-dive on the leading agent memory architectures and
the recent (2025-2026) research that matters for ContextMemory. Complements
`2026-08-29-frontier-memory-landscape.md` (strategic landscape) with concrete
architecture, write-path LLM cost, read-path design, latency, and per-system
"what to steal" analysis. Goal: beat Supermemory (supermemoryai) on token
usage, latency, and benchmark scores.

---

## 0. Benchmark context and caveats

All cross-system benchmark comparisons below are NOT apples-to-apples.
Scores vary with: judge model, LongMemEval setting (S ~115k tokens / ~50
sessions vs M ~1.5M tokens / ~500 sessions), answer/backbone model, and
retrieval token budget. MemScore (from Supermemory's MemoryBench) is the
closest thing to a standardized triple: `accuracy% / latencyMs / contextTokens`
(https://github.com/supermemoryai/memorybench). Treat every vendor number as
self-reported unless noted.

Quick map of the field (all 2026):

| System | Architecture | LongMemEval (best published) | Read path has LLM? | Latency signals |
|---|---|---|---|---|
| Zep / Graphiti | Bi-temporal knowledge graph | 71.2% (GPT-4o); managed p95 <200ms | No (retrieval), embedder + cross-encoder | search 200ms-1s OSS; managed sub-200ms |
| Mem0 | Vector + entity/graph, ADD-only extraction | 94.4% (platform, v3) | No (fusion of signals) | search p50 0.15s; total p50 ~0.9s |
| Hindsight | 4-network temporal graph, TEMPR + CARA | **91.4%** (Gemini-3 Pro); 83.6% (OSS-20B) | No (recall); yes (reflect = answer gen) | cross-encoder rerank in recall |
| Letta/MemGPT | OS-tiered blocks, agent-managed | 83.2% (cited) | Yes (agent calls memory tools) | tool-call dependent |
| LangMem | LangGraph namespaces, hot-path + background | no standardized | Yes (agent tools) | search p50 ~18s (cited) |
| Supermemory (target) | Hybrid search + graph traversal + context engine | ~85% production; 95% Recall@15 @ ~720 tokens | Operations tier has query rewriting | search/traversal sub-300ms p50; ~50ms profiles |

---

## 1. Zep / Graphiti — bi-temporal temporal knowledge graph

**Primary sources:** arXiv 2501.13956 (https://arxiv.org/abs/2501.13956),
repo https://github.com/getzep/graphiti, SOTA blog
https://blog.getzep.com/state-of-the-art-agent-memory/, performance docs
https://getzep-graphiti.mintlify.app/advanced/performance-tuning

### Architecture summary
Graph `G = (N, E, φ)` with three hierarchical subgraphs:
- **Episode subgraph** `Ge`: raw input (message/text/JSON) stored non-lossily.
  Episodes carry a reference timestamp `t_ref` so relative dates ("last
  Tuesday") resolve correctly.
- **Semantic entity subgraph** `Gs`: entities extracted from episodes,
  resolved against existing graph entities (dedup), plus fact edges between
  entities.
- **Community subgraph** `Gc`: clusters of strongly connected entities with
  high-level summaries (LightRAG-parallel, optional; Graphiti works without).

### Data model — bi-temporal
Two timelines: `T` (validity / chronological fact truth) and `T'`
(transactional / ingestion order). Every edge tracks four timestamps:
`t'created`, `t'expired` (transactional) and `t_valid`, `t_invalid`
(validity). Edge invalidation: when a new fact contradicts a related existing
edge, the LLM sets the old edge's `t_invalid` to the new edge's `t_valid`.
New info always wins (transactional timeline priority). Superseded facts are
closed, not deleted — auditable history, current-state query is "open windows".

### Write path — the expensive part
Per-episode pipeline (each step a separate prompt, some parallelizable):
1. Entity extraction (with last `n=4` messages of context) + reflection pass
   to reduce hallucination + entity summary. 2-3 LLM calls.
2. Embed each entity name (1024-dim) + fulltext search for candidates.
3. Entity resolution: LLM duplicate check against candidates. 1-2 calls.
4. Fact/edge extraction from episode. 2-3 calls.
5. Edge dedup, batch temporal extraction, contradiction/invalidation. 1-2 calls.

Official perf docs: **"Actual concurrent LLM requests = SEMAPHORE_LIMIT × 6-10"**.
Cost telemetry (issue #467): average **~$0.018 per episode** with default
OpenAI models; per-entity attribute extraction is **74.7%** of episode cost,
entity dedup 20%. Issue #1516: a 5KB doc (~30 entities, ~50 edges) triggers
**~160 serial LLM calls** (30 dedup + 30 attribute + 100 edge/classify).
Throughput guidance: "cap planning at ~1 entity + 1 edge per second per
coroutine". A `skip_extraction` flag was added to bypass LLM extraction
(30 min → 0.03s). Chunk size 4000 tokens reduces calls.

**Cost reduction trick worth stealing:** keep extraction prompts small-model
friendly (GPT-4o-mini default); don't feed the whole graph back into the LLM;
cap dedup context via hybrid candidate search (prompt size does not grow with
graph size). "Add 2000 PDF pages for about a dollar" (their estimate).

### Read path — deterministic (no text LLM)
Three search functions fused + reranked:
- `φcos` cosine semantic similarity (embeddings)
- `φbm25` Okapi BM25 fulltext (Neo4j Lucene)
- `φbfs` breadth-first n-hop graph traversal (can seed from recent episodes)
Then **MMR reranker** + graph-based episode-mentions reranker. No text-to-text
LLM in retrieval — just embedder + cross-encoder. Their claim: >50% of read
latency is the embedding API; "results back in a few hundred ms at negligible
costs" vs GraphRAG's per-query LLM summarization.

### Measured latency / tokens
- OSS perf table: Add episode (short) 2-5s / p95 8s; (long, chunked) 8-15s /
  p95 25s. Search (5 results) 200-500ms / p95 1s; (20 results, reranked)
  500ms-1s / p95 2s.
- LongMemEval-S evaluation (blog): Zep + GPT-4o median 2.58s end-to-end with
  **1.6k context tokens** vs full-context 28.9s / 115k tokens — **~90%
  latency reduction, <2% of tokens**. Accuracy 71.2% vs 60.2% full-context;
  DMR 94.8% vs MemGPT 93.4%. Weak spot: single-session-assistant recall
  (regressed vs full-context).
- Managed Zep (proprietary "Konig" Context Graph Engine, millions of small
  mostly-cold per-user graphs): **sub-200ms p95 retrieval at scale**.

### Storage backends
Graphiti drivers: Neo4j (default, Bolt, Lucene fulltext), FalkorDB (Redis
module, `@` fulltext syntax), Kuzu (embedded, **deprecated** — upstream
archived Oct 2025), Amazon Neptune. Managed Zep uses Konig, a proprietary
graph database service. FalkorDBLite gives an embedded option.

### What to steal
1. **Bi-temporal validity windows + LLM-judged edge invalidation** — the
   cleanest known answer to knowledge-update/temporal-reasoning questions.
2. **Non-lossy episode subgraph under derived entity graph** — provenance
   and replay without bloating retrieval.
3. **Deterministic read path** (semantic + BM25 + BFS + MMR, no generative
   LLM) — the right latency/cost posture.
4. **Anti-pattern to avoid:** per-entity/per-edge LLM calls on write
   (~$0.018/episode, ~160 calls per 5KB doc) — too expensive.

---

## 2. Mem0 — vector + entity linking, ADD-only extraction

**Primary sources:** arXiv 2504.19413 (https://arxiv.org/abs/2504.19413),
repo https://github.com/mem0ai/mem0, docs/how-it-works, architecture.md,
graph-memory.mdx (docs in repo), research page https://mem0.ai/research

### Architecture summary
Dual/triple store:
- **Vector DB**: memory text + embeddings + metadata (timestamps, hash,
  categories, attributed_to) — primary semantic retrieval.
- **Entity/graph store**: entities + entity embeddings + linked memory IDs —
  entity-based retrieval boost. OSS: entity store; Platform: full graph
  memory with nodes/edges (Neo4j, Memgraph, Neptune, Kuzu, Apache AGE).
- **SQL DB**: history log (ADD events) + rolling message window — dedup
  context + audit.

v3 is **ADD-only**: nothing overwritten/deleted; facts accumulate. Hash-based
(MD5) dedup prevents exact duplicates. Claimed +42.1 on temporal reasoning
vs their old UPDATE/DELETE algorithm.

### Write path (`add_memories`) — cheap single-pass
5 stages:
1. Store new memories (async, after agent responds).
2. Context lookup — find related existing memories (to avoid dup storage).
3. **Distill memories — single-pass LLM extraction** producing ADD-only facts
   from input + context. One LLM call for extraction.
4. Deduplicate (hash) + embed (batch).
5. **Entity linking** — extraction LLM identifies entities (proper nouns,
   quoted text, compound noun phrases), relationships, timestamps; entities
   embedded and linked across memories (nodes/edges into graph backend).

Key point: **the entire write path is ~1 LLM extraction call + embedding**,
plus a lookup search. This is orders of magnitude cheaper than Graphiti's
write path. Async add returns in **<50ms**; the LLM work is off the request
path.

### Read path — deterministic fusion, no LLM
Multi-signal parallel scoring + fusion:
- Semantic (vector similarity)
- Keyword (BM25 with verb-form lemmatization)
- Entity (boost memories sharing entities with query; Platform = graph)
- Temporal (time metadata extracted at write vs query temporal intent)
Optional `rerank=True`. Single-pass retrieval (no agentic loops), top_200
budget in their evals.

### Measured latency / tokens
- Paper (LOCOMO): Mem0 search p50 0.148s / p95 0.200s; total p50 0.708s /
  p95 1.44s. Mem0g (graph) search p50 0.476s, total p50 1.091s, highest
  score 68.44% vs 61% best-RAG. 91% lower p95 latency + >90% token savings
  vs full-context.
- Platform v3 (2026): hybrid search ~100-150ms, +rerank +150-200ms, add async
  <50ms. New-algorithm benchmark numbers: LoCoMo **92.5**, LongMemEval
  **94.4** (98.2 assistant recall), BEAM 1M **64.1**, BEAM 10M **48.6**,
  at ~6.7-6.9k tokens/query, latency p50 0.88-1.09s.

### What to steal
1. **Single-pass ADD-only extraction** — cheapest correct write path; keeps
   temporal history; defers consolidation decisions.
2. **Entity linking as a retrieval signal (boost), not a retrieval path** —
   cheap graph, high recall payoff on multi-hop/temporal (+23.1/+29.6 gains).
3. **Async add with <50ms response** — write cost invisible to the user.
4. Anti-pattern to avoid: per-entity extraction calls (their own docs note
   the platform does proprietary optimizations beyond OSS).

---

## 3. Hindsight — 4-network temporal memory (TEMPR + CARA)

**Primary sources:** arXiv 2512.12818 "Hindsight is 20/20"
(https://arxiv.org/abs/2512.12818), ACL demo
(https://aclanthology.org/2026.acl-demo.27.pdf), repo
https://github.com/vectorize-io/hindsight, docs at hindsight.vectorize.io

### Architecture summary
Memory bank with **four logical networks**:
- **World**: objective facts about the external world.
- **Experience**: the agent's own experiences, first person.
- **Opinion**: subjective beliefs with confidence scores, updated by evidence.
- **Observation**: preference-neutral entity summaries, consolidated from
  underlying facts.
Three operations: **retain / recall / reflect**. Two components: **TEMPR**
(Temporal Entity Memory Priming Retrieval = retain+recall) and **CARA**
(Coherent Adaptive Reasoning Agents = reflect). Storage: **PostgreSQL +
pgvector** (HNSW partial indexes per fact type, GIN-indexed tsvector for BM25).

### Write path (retain)
LLM extracts **2-5 coarse narrative facts per conversation** (each fact covers
an entire exchange, not per-utterance — makes retrieval less sensitive to
segmentation). Each fact: classified into a network, timestamped with
occurrence interval (τs, τe) + mention time τm, entity-resolved (string
similarity + co-occurrence), and linked via **temporal, semantic, entity, and
causal graph edges**. Observation-level consolidation refines entity summaries
as evidence arrives (strengthen/weaken trends).

### Read path (recall)
**Four retrieval strategies in parallel**: semantic (HNSW vector), keyword
(BM25), graph (spreading activation over entity/temporal/causal links),
temporal (occurrence-interval filtering). Merge via **Reciprocal Rank Fusion
+ cross-encoder rerank**. **Token-budget-constrained**: `Recall(B, Q, k)`
returns the ranked facts whose combined tokens ≤ k — predictable context cost
regardless of memory size. No generative LLM in recall. Reflect (answer
generation) is an LLM conditioned on profile + disposition traits
(skepticism/literalism/empathy 1-5), which shapes opinions via reinforcement
(supporting evidence ↑ confidence, contradicting ↓).

### Benchmarks (best published, self-reported)
- LongMemEval: **91.4%** (Gemini-3 Pro), 89.0% (OSS-120B), **83.6%**
  (OSS-20B; +44.6 pts over full-context same model = 39.0%, and beats
  full-context GPT-4o 60.2%). Beats Supermemory+GPT-4o (81.6%) and
  Zep+GPT-4o (71.2%).
- LoCoMo: 89.61% (Gemini-3), 85.67% (OSS-20B) vs 75.8% Memobase / 75.1% Zep.
- Largest gains on multi-session (21.1→79.7) and temporal reasoning
  (31.6→79.7) with OSS-20B — the structured time/entity graph, not the
  reranker, drives the lift (their ablation claim).

### What to steal
1. **Fact-belief separation (world vs opinion) with confidence + evidence
   reinforcement** — the most credible mechanism for preference/opinion
   evolution questions (weakest category for most systems).
2. **Coarse narrative facts (2-5 per conversation) instead of per-utterance
   facts** — cuts write LLM cost and noise.
3. **Token-budget-constrained retrieval** — direct lever to beat token-usage
   metrics while keeping precision.
4. **Four-link graph (temporal, semantic, entity, causal)** + spreading
   activation for multi-hop.
5. Anti-pattern to avoid: cross-encoder rerank on huge candidate sets is
   where read latency accumulates; pair with candidate narrowing (SwiftMem).

---

## 4. Letta / MemGPT and LangMem — tiered / self-managing memory

**Primary sources:** https://arxiv.org/abs/2310.08560 (MemGPT), Letta docs
memory-blocks (https://docs.letta.com/guides/core-concepts/memory/),
Letta leaderboard (https://leaderboard.letta.com/), LangMem docs
(https://langchain-ai.github.io/langmem/), langchain-ai/langmem repo.

### MemGPT / Letta
OS-inspired hierarchy: **core memory** = in-context blocks (persona, human,
custom; character-limited ~2-5k chars; always visible; agent-editable via
`memory_insert`/`memory_replace`/`memory_rethink`); **external memory** =
archival (semantic search via `archival_memory_search`) + conversation
history (hybrid search) + filesystem. Agent pages its own memory via tools.
**Sleep-time compute**: background/sleep-time agents consolidate and rewrite
blocks during idle periods — memory management off the interactive path.
Shared blocks give multi-agent memory. Letta Leaderboard / Context-Bench now
measure *models'* context-engineering ability (GPT-5.2 Codex #1 filesystem
93.0), not memory-layer retrieval quality — a different axis from the
LongMemEval family.

### LangMem
Three memory types: **semantic** (facts; collections for unbounded knowledge
or profiles for schema'd current-state), **episodic** (past experiences /
few-shot), **procedural** (system instructions; prompt optimization).
Formation: **hot path** (agent tools `create_manage_memory_tool` /
`create_search_memory_tool`) or **background** (`ReflectionExecutor`,
`create_memory_store_manager` for batch extraction/consolidation, with
`enable_inserts`/`enable_deletes` toggles). Storage on LangGraph `BaseStore`
(namespace/key JSON docs, semantic search). Each memory op = "accept
conversation + current state → LLM → updated state". No standardized
LongMemEval scores; cited p50 search ~18s makes it impractical for interactive
use — this is the cost of LLM-managed memory.

### What to steal
1. **Tiered memory with an always-visible core budget + retrieved archive** —
   an explicit, controllable token budget per turn.
2. **Sleep-time/background consolidation** — move LLM work off the request
   path (Letta proved this design).
3. Anti-pattern to avoid: letting the agent drive memory via tool calls
   (slow, nondeterministic, token-hungry on read).

---

## 5. Embedded / graph databases for a memory layer

**Primary sources:** Kuzu study (https://github.com/prrao87/kuzudb-study),
community benchmark (https://github.com/Sunny-sketchs/graph-db-benchmark),
AIMultiple benchmark (https://aimultiple.com/graph-databases), Kuzu archival
analysis (https://gdotv.com/blog/kuzu-legacy-embedded-graph-database-landscape/),
Graphiti driver docs, licensing analysis (dreaming.press and till-freitag posts).

| Engine | Deployment | Storage | License | Vector/fulltext | Notes |
|---|---|---|---|---|---|
| **Kuzu** | **Embedded (in-process)** | columnar, disk | MIT | HNSW + fulltext built-in | **Archived Oct 10 2025** (Apple acqui-hired team). Forks: LadybugDB (Kineviz), Lance Graph, TuringDB, FalkorDBLite. Dead as a dependency. |
| **FalkorDB** | Server (Redis module) / **FalkorDBLite embedded** | GraphBLAS sparse matrices in Redis | **SSPLv1** (source-available; restrictive for SaaS) | RedisSearch fulltext; vector via separate index | Multi-tenant native: each graph = a Redis key, thousands of small per-user graphs per instance — ideal for agent memory shape. |
| **Memgraph** | Server | in-memory + WAL | BSL 1.1 → Apache 2.0 after 4 yrs | HNSW | RAM-bound working set; MAGE algorithm library; Kafka streaming; Bolt-compatible. |
| **Neo4j** | Server | native graph, disk | GPLv3 Community / Commercial | native HNSW + Lucene | Ecosystem king; JVM-heavy (~2.7GB heap for 381k/804k benchmark graph); overkill <5M nodes. |

Measured signals (not vendor-controlled):
- Kuzu vs Neo4j (prrao87): ingestion 53x faster (0.58s vs 30.64s), multi-hop
  queries 2-374x faster; embedded = zero transport overhead.
- Community 0.5vCPU/256MB benchmark: FalkorDB fastest traversals (1-hop p50
  1.19ms, 2-hop p50 3.58ms) and mixed throughput (1,338 req/s); Kuzu fastest
  point lookups (0.56ms) and aggregation (26ms); Memgraph/ArangoDB OOM'd
  under the cap.
- AIMultiple (381k nodes/804k edges): FalkorDB won 11/12 queries, 6,693 QPS
  @ 8 threads (6.7x Neo4j), cold start 1.1ms, point lookups 0.044ms.

**Takeaway:** today there is **no permissively-licensed embedded graph
engine** (Kuzu is dead). Options: FalkorDBLite (embedded, SSPL), Postgres +
pgvector + a link table (Hindsight's choice — pragmatic, boring, and it
achieved the best published LongMemEval), or a purpose-built columnar store.
FalkorDB's per-graph = Redis-key model is the right multi-tenancy shape for
per-user memory.

---

## 6. Supermemory — the target

**Primary sources:** repo https://github.com/supermemoryai/supermemory,
research https://supermemory.ai/research, MemoryBench
https://github.com/supermemoryai/memorybench.

### What they claim
- **#1 on LongMemEval, LoCoMo, ConvoMem** (self-reported).
- LongMemEval: **95% Recall@15 adding only ~720 tokens of context = 99.4%
  context reduction** (99.6% @10, 99.8% @5). Recall by category: knowledge
  updates 99%, assistant 100%, user 97%, multi-session 93%, temporal 91%,
  preference 90%. Production LongMemEval ~85% (their research page).
- **~50ms user profiles**; **search & traversal sub-300ms p50** (hybrid search
  + graph traversal); operations tier (re-ranking, aggregation, query
  rewriting) $0.10/1k.
- Architecture: hybrid search + graph traversal + "context engine"; a
  **query-rewriting/operations tier exists on the read path** (billed
  separately, so it is not LLM-free).
- **Experimental ASMR** (not production): ditches vector DB; parallel LLM
  "reader" agents (6, Gemini 2.0 Flash) extract 6 categories during ingest;
  3 "search" agents + 8-12 answer variants → **98.6% LongMemEval_s** (12-agent
  consensus version 97.2%). Cost: ~12+ frontier API calls per query — the
  anti-thesis of token/latency efficiency. This is the benchmark-chasing
  ceiling, not a production baseline.

### What beating it requires (targets)
Their published operating point: **~85% LongMemEval, ~720-1500 tokens of
context, sub-300ms search, ~50ms profiles**. To beat on all three axes we
need: accuracy ≥ 85% (ideally ≥ 91% territory that Hindsight proved reachable
with a strong backbone), context < ~700 tokens, and p50 search well under
100ms with a deterministic read path. The ASMR result shows raw accuracy
headroom exists but only with prohibitive cost — it is not the real
competitor on our chosen axes.

---

## 7. Recent research: consolidation, forgetting, indexing, write-cost

### SwiftMem — query-aware indexing (arXiv 2601.08160, Huawei, Jan 2026)
https://arxiv.org/abs/2601.08160 · https://github.com/EdwardTex/SwiftMem
- **Problem:** every query scans the whole memory (O(N_mem)) even with HNSW;
  ANN reduces scan cost but not *which region* to search.
- **Solution:** three-tier query-aware index. (1) **Temporal index**:
  per-user sorted timelines + global episode lookup → O(log N) range queries.
  (2) **Semantic DAG-Tag index**: LLM generates 3-8 tags per episode; tags
  form a DAG; a query-tag router maps queries to tags via embedding alignment
  + top-k + hierarchical expansion → O(k·(log|V|+Dmax)). (3) **Embedding
  index with co-consolidation**: periodically reorganize vector layout so
  same-tag-cluster embeddings are physically adjacent (cache locality).
- **Numbers:** **10.8-12.7 ms/query search** on LoCoMo (GPT-4.1-mini);
  **47x faster than Zep (522ms)** and 76x vs Nemori (835ms); total e2e 1,289ms
  vs full-context 5,806ms. Temporal index alone cut search 35% (11.1→7.2ms);
  co-consolidation improved LLM-judge 64.3→78.6 and latency 10.2→7.4ms.
- **Steal:** route-then-search before any dense/rerank step; temporal index;
  physical layout co-consolidation. Caveat: tag routing can drop relevant
  candidates for vague queries; needs a fallback path.

### FadeMem — biologically-inspired forgetting (arXiv 2601.18642, 2026)
https://arxiv.org/html/2601.18642v2
- Dual-layer memory with **adaptive exponential decay** (Ebbinghaus curve)
  modulated by semantic relevance, access frequency, temporal patterns;
  LLM-guided conflict resolution and memory fusion (subsumes/subsumed merges);
  fused memories decay slower (consolidated importance). **45% storage
  reduction** with better multi-hop/retrieval on Multi-Session Chat, LoCoMo,
  LTI-Bench.
- **Steal:** selective forgetting as an explicit, scheduled operation — the
  field keeps everything and pays retrieval noise.

### SleepGate — sleep-cycle KV cache consolidation (arXiv 2603.14517, Mar 2026)
https://arxiv.org/html/2603.14517v1
- Targets **proactive interference** (stale superseding values poison
  retrieval, degrades log-linearly regardless of context length). Learned
  sleep micro-cycles over the KV cache: conflict-aware temporal tagger +
  forgetting gate + consolidation. Theory: interference horizon O(n)→O(log n).
  Proof-of-concept: 99.5% retrieval at PI depth 5 vs <18% for all baselines.
- **Steal:** the framing — consolidating/evicting superseded traces is the
  answer to knowledge-update questions; relevant to our temporal store design.

### Nemori — prediction-error distillation (arXiv 2508.03341, 2025)
https://arxiv.org/html/2508.03341v4
- What deserves memory? **Prediction error**: what existing knowledge can
  already predict is redundant; what it fails to predict deserves storage.
  Training-free; Episodic Memory Integration → Semantic Knowledge Distillation.
  Temporal reasoning 77.3 LLM-judge (gpt-4.1-mini), +15.9% over A-MEM,
  +14.8% over Zep. Data-driven alternative to importance-score heuristics.
- **Steal:** cheap distillation-time importance signal — skip extraction for
  predictable/redundant content → write-cost and storage reduction.

### Memory management empirics (arXiv 2505.16067, 2025)
https://arxiv.org/abs/2505.16067v1
- "Experience-following" property (similar input → similar output), **error
  propagation** (bad memories beget bad behavior), **misaligned experience
  replay**. Selective addition (quality-gate before storing) + combined
  deletion (redundant/outdated/never-retrieved) → **+10% absolute gain**.
- **Steal:** quality-gate the write path; delete on evidence, not on age.

### Memini / multi-timescale memory (arXiv 2605.05097, 2026)
- Benna-Fusi-style coupled **fast + slow variables** per associative edge:
  fast = episodic (decays without reinforcement), slow = consolidated
  (strengthens with repetition). Emergent selective forgetting from the
  dynamics rather than hard-coded rules.

### CMA — Continuum Memory Architecture (arXiv 2601.09913, 2026)
- A requirements checklist for "real" memory vs RAG: persistence, **selective
  retention**, **retrieval-driven mutation** (access strengthens/stabilizes),
  associative routing (spreading activation), temporal continuity,
  consolidation/abstraction. Useful spec for what ContextMemory's semantics
  must include.

---

## 8. HyDE / query expansion / hybrid retrieval for sub-100ms

**Primary sources:** HyDE paper (Gao et al., 2022), BEIR results; Gemma RAG
vs HyDE study (arXiv 2506.21568); practical guides (zeroentropy.dev,
cfgnotes.com, atomic-rag, dev.to retrieval checklist).

### Facts
- **HyDE** (embed a generated hypothetical answer instead of the query):
  beats DPR zero-shot on 11/11 BEIR; MS-MARCO nDCG@10 56.6 vs 43.3. Cost:
  **+1 LLM call per query**; +25-60% latency on 1-4B local models; in a
  personal-data assistant study it had **100% hallucination rate** and was
  slower (13.2s vs 7.9s RAG). It is a *recall lever, not a precision lever* —
  valuable only when the stack is recall-bound.
- **For sub-100ms retrieval, an LLM on the read path is disqualifying.**
  Query rewriting/HyDE LLM calls (even small models) add 100ms-1s+.
- Best-practice deterministic recipe (consensus across 2024-2026 sources):
  1. **Hybrid = BM25 + dense, fused with RRF** — beats either alone on BEIR/
     MTEB; "dense-only lost that argument."
  2. Retrieve broad (~20-50), **rerank with a small cross-encoder to ~5**,
     send 3-5. Reranker only helps what the first pass surfaced — measure
     recall@K, not just NDCG@10.
  3. If you use HyDE/expansion at all: use a tiny/fast model, **cache
     hypotheses**, run it in **parallel** with first-pass retrieval or
     **conditionally** (a cheap classifier for ambiguous queries), and fall
     back to the raw query when hypothesis↔query cosine < ~0.5.
  4. Sub-15ms is proven possible deterministically (SwiftMem) with routing +
     HNSW + BM25 + small reranker.

---

## 9. Synthesis — what to build into ContextMemory

Cross-system, the strongest composite design for the goal (beat Supermemory
on tokens, latency, benchmarks):

**Write path (LLM, but cheap and off-request):**
- Single-pass ADD-only extraction of **coarse narrative facts** (Hindsight
  style: 2-5 per conversation, not per utterance; Mem0's one-call extraction
  validates this). Batch timestamp extraction in one call (Graphiti's
  `extract_timestamps_batch`). NO per-entity/per-edge LLM calls (Graphiti
  anti-pattern).
- Quality-gate storage (memory-management study: selective addition).
- Async: return <50ms, extract/embed/consolidate in background (Mem0/Letta).
- Entity + temporal + causal metadata extracted in the same pass, not
  separately.

**Temporal store (bi-temporal, the core differentiator):**
- Graphiti-style validity windows (t_valid/t_invalid) + LLM-judged edge
  invalidation for knowledge updates; provenance to episodes.
- SwiftMem-style temporal index (per-user sorted timelines, O(log N)) so
  temporal queries never scan.
- Opinion network with confidence + evidence reinforcement (Hindsight) for
  preference/opinion evolution.
- Selective forgetting as a scheduled consolidation job (FadeMem decay /
  SleepGate eviction), off the read path.

**Read path (deterministic, sub-100ms, near-zero tokens of overhead):**
- Route-then-search: query-tag routing + temporal check first (SwiftMem),
  then hybrid HNSW + BM25 + entity/graph boost + temporal channel (Hindsight
  four-way but only on the narrowed candidate set), RRF fuse, small
  cross-encoder rerank on ~20-50 candidates. No generative LLM.
- Token-budget-constrained assembly (Hindsight) to hit <700 tokens/query.
- Plan: LongMemEval-S ~85%+ (Zep 71.2%, Hindsight-OSS-20B 83.6% prove the
  architecture carries most of the accuracy), p50 search target ~10-30ms
  (SwiftMem proves 11ms), context ~500-700 tokens vs Supermemory's 720.

**Storage:**
- No permissively-licensed embedded graph engine exists (Kuzu dead).
  Postgres + pgvector + link/timeline tables (Hindsight's boring, benchmark-
  winning choice) or FalkorDBLite (embedded, per-graph multi-tenancy, SSPL)
  are the practical options. Measure before choosing.

---

## 10. Open questions for ContextMemory

- Standard rig (judge, slice S vs M, backbone, token budget) to make any
  claim vs Supermemory comparable — per landscape report, build the harness
  first.
- Whether 91%+ LongMemEval is reachable without an LLM reflect layer (that
  number uses an answer LLM; retrieval-only ceiling is lower).
- Cost of cross-encoder rerank on the narrowed candidate set vs accuracy
  delta — measure with local CPU-friendly models (no GPU available).

## Sources

- Zep: https://arxiv.org/abs/2501.13956 · https://github.com/getzep/graphiti ·
  https://blog.getzep.com/state-of-the-art-agent-memory/ ·
  https://getzep-graphiti.mintlify.app/advanced/performance-tuning ·
  GitHub issues #200, #467, #1516
- Mem0: https://arxiv.org/abs/2504.19413 · https://github.com/mem0ai/mem0 ·
  https://mem0.ai/research
- Hindsight: https://arxiv.org/abs/2512.12818 ·
  https://aclanthology.org/2026.acl-demo.27.pdf ·
  https://github.com/vectorize-io/hindsight
- Letta/MemGPT: https://arxiv.org/abs/2310.08560 ·
  https://docs.letta.com/guides/core-concepts/memory/ ·
  https://leaderboard.letta.com/
- LangMem: https://langchain-ai.github.io/langmem/ ·
  https://github.com/langchain-ai/langmem
- Graph DBs: https://github.com/prrao87/kuzudb-study ·
  https://github.com/Sunny-sketchs/graph-db-benchmark ·
  https://aimultiple.com/graph-databases ·
  https://gdotv.com/blog/kuzu-legacy-embedded-graph-database-landscape/
- Supermemory: https://github.com/supermemoryai/supermemory ·
  https://supermemory.ai/research ·
  https://github.com/supermemoryai/memorybench
- Research: SwiftMem https://arxiv.org/abs/2601.08160 · FadeMem
  https://arxiv.org/html/2601.18642v2 · SleepGate
  https://arxiv.org/html/2603.14517v1 · Nemori
  https://arxiv.org/html/2508.03341v4 · memory mgmt
  https://arxiv.org/abs/2505.16067v1 · Memini https://arxiv.org/abs/2605.05097 ·
  CMA https://arxiv.org/pdf/2601.09913v1
- HyDE/retrieval: HyDE (Gao et al. 2022) · arXiv 2506.21568 ·
  https://zeroentropy.dev/concepts/query-expansion/ ·
  https://cfgnotes.com/rag/hyde-query-expansion/