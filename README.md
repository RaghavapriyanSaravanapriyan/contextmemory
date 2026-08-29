# contextmemory

> a better persistent memory and context optimization layer for modern agentic systems.

ContextMemory is an engineering effort to build a memory layer for agentic
systems that measurably outperforms the 2026 frontier (Mem0, Zep/Graphiti,
Letta, LangMem, and the LongMemEval-S cluster) on both the standard memory
benchmarks and the dimensions the field does not measure: write precision,
temporal evolution, forgetting, and read-path latency.

The guiding principle: you cannot beat what you cannot measure. The
evaluation harness therefore comes first, and every claim is backed by a
reproducible run.

See `tasks/active/T001-beat-frontier-memory-layers.md` for the mission and
milestones, and `reports/research/2026-08-29-frontier-memory-landscape.md`
for the landscape analysis.

## Installation

```bash
uv sync --extra dev
```

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Usage

The evaluation harness replays benchmark histories through a memory system
and answers questions with a model-agnostic reader client (any
OpenAI-compatible endpoint: frontier APIs, vLLM, Ollama, LM Studio).

```bash
uv run contextmemory \
  --data benchmarks/data/longmemeval_oracle.json \
  --system full-history \
  --reader-api-base https://api.openai.com/v1 \
  --reader-api-key $OPENAI_API_KEY \
  --reader-model gpt-4o-mini \
  --out reports/runs/run.jsonl
```

Download the LongMemEval data first:

```bash
mkdir -p benchmarks/data
cd benchmarks/data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
```

## Repository structure

```text
src/contextmemory/eval/   evaluation harness (protocol, data, runner, scoring, baselines)
benchmarks/data/          downloaded benchmark datasets (gitignored)
reports/research/         research investigations
reports/runs/             experiment records
docs/architecture/        current system architecture
tasks/active/             active tasks
```

## Development

```bash
scripts/verify.sh     # run the full verification (tests + lint)
uv run pytest         # test suite
uv run ruff check src tests
```