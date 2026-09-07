# Frontier Memory Benchmarks (2026 synthesis)

Durable reference for which instrument measures what. Synthesized from
primary sources (linked); vendor self-reports are flagged, not facts.

## The three strict-memory benchmarks

| Benchmark | Scale & shape | Measures | Status 2026 |
| --- | --- | --- | --- |
| LoCoMo (ACL 2024) | ~35 sessions, ~300 turns, ~9K tokens, one narrative; full release 50 conv / 7,512 Q | single/multi-hop, temporal, open-domain recall | Near-saturated (managed systems ~92). Regression test, not a differentiator. Caveat: "LoCoMo" names two tests — full release vs ~10-conv vendor subset. Ask which. |
| LongMemEval (ICLR 2025) | 500 curated Q, 5–6 abilities; S ~115K tokens/40 sessions, M ~500 sessions | extraction, multi-session + temporal reasoning, knowledge update, abstention | Still discriminative. Assistants drop ~30% as histories grow. The honest conversational bar. |
| BEAM (arXiv 2025, 2510.27246) | 100 convos up to 10M tokens, 2,000 Q, 10 abilities | + contradiction resolution, event ordering, instruction following, summarization | The frontier. 1M-context models degrade with length even with retrieval. Cite as arXiv (venue unconfirmed). |

## Newer / adjacent instruments

- **LongMemEval-V2 (2026):** 451 Q over web-agent trajectories (up to 115M
  tokens). Tests experience memory (workflows, gotchas, state tracking),
  not chit-chat recall. Best reported ~72% (coding-agent memory).
- **MemoryBench (Supermemory, open source):** standardized head-to-head
  harness for memory providers (Supermemory, Mem0, Zep, …). The right tool
  for provider comparisons.
- **Long-context attention benches** (NIAH, RULER, InfiniteBench,
  LongBench): measure the substrate (attention over one fixed input), NOT
  cross-session memory. Do not cite as memory scores.

## What no public benchmark measures (our edge)

Write precision, forgetting/eviction/consolidation dynamics, per-user
isolation, and deterministic retrieval latency. ContextMemory's `dims`
(write-precision / evolution / forgetting) and `bench` (null-reader
p50/p95) harnesses exist precisely for these gaps.

## Published reference numbers (vendor-reported, dated 2026)

- Supermemory: #1 LongMemEval, LoCoMo, ConvoMem; 95% Recall@15, 99.4%
  context reduction, ~50 ms profiles (supermemory.ai, self-report).
- Mem0: LoCoMo 92.5, LongMemEval 94.4, BEAM-1M 64.1, BEAM-10M 48.6,
  ~6.9K tokens/query, p50 ≤ 1.1 s (mem0.ai, self-report, Apr–Jun 2026).
- BEAM paper LIGHT: +3.5–12.7% over long-context baselines on average,
  +100%+ at the 10M extreme (Tavakoli et al.).

Rule: never present these beside our numbers as head-to-head. Different
harnesses, judges, and dates. Compare head-to-head on one rig
(`benchmarks/run_official.py`) or say nothing.

## Sources

- LoCoMo: Maharana et al., arXiv:2402.17753
- LongMemEval: Wu et al., ICLR 2025, arXiv:2410.10813
- BEAM/LIGHT: Tavakoli et al., arXiv:2510.27246
- LongMemEval-V2: REAL-Lab-NU/LongMemEval-V2 (GitHub)
- Mem0 2026 numbers: mem0.ai/blog (state-of-memory, benchmarks)
- Supermemory: supermemory.ai/docs, github.com/supermemoryai/supermemory
- Landscape: dreaming.press LoCoMo-vs-LongMemEval-vs-BEAM (2026-06),
  mnemoverse.com evaluation guide, Evanyuan-builder/memory-core-eval,
  MemTensor/OmniMemEval
