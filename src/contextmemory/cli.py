"""CLI for running memory evaluations.

Provides a minimal, reproducible entry point for experiments:

    cm eval --data benchmarks/data/longmemeval_oracle.json \\
        --system full-history --reader-api-base https://api.openai.com/v1 \\
        --reader-api-key $OPENAI_API_KEY --reader-model gpt-4o-mini \\
        --out reports/runs/run.jsonl

Deterministic scoring is used for iteration; pass ``--judge-model`` to also
run the official-style LLM judge.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from contextmemory.eval import (
    FullHistorySystem,
    MemorySystem,
    OpenAICompatClient,
    RecencyWindowSystem,
    load_longmemeval,
    replay,
    score_deterministic,
)
from contextmemory.eval.runner import ReplayResult

_SYSTEMS: dict[str, Callable[[OpenAICompatClient], MemorySystem]] = {
    "full-history": lambda reader: FullHistorySystem(reader),
    "recency-2": lambda reader: RecencyWindowSystem(reader, window=2),
    "recency-10": lambda reader: RecencyWindowSystem(reader, window=10),
}


def _write_results(path: str, results: list[ReplayResult]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "question_id": r.question_id,
                        "question_type": r.question_type,
                        "hypothesis": r.hypothesis,
                        "judged": r.judged,
                        "answer_time_s": r.timing.answer_s,
                        "ingest_time_s": r.timing.ingest_s,
                    }
                )
                + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a memory evaluation.")
    parser.add_argument("--data", required=True, help="LongMemEval JSON data file")
    parser.add_argument(
        "--system",
        required=True,
        choices=sorted(_SYSTEMS),
        help="memory system under test",
    )
    parser.add_argument("--reader-api-base", required=True)
    parser.add_argument("--reader-api-key", default="EMPTY")
    parser.add_argument("--reader-model", required=True)
    parser.add_argument(
        "--max-instances", type=int, default=0, help="limit instances (0 = all)"
    )
    parser.add_argument("--out", required=True, help="path for hypothesis JSONL")
    args = parser.parse_args(argv)

    instances = load_longmemeval(args.data)
    if args.max_instances:
        instances = instances[: args.max_instances]

    reader = OpenAICompatClient(
        base_url=args.reader_api_base,
        api_key=args.reader_api_key,
        model=args.reader_model,
    )
    factory = lambda: _SYSTEMS[args.system](reader)  # noqa: E731
    results = replay(instances, factory)
    report = score_deterministic(results)
    _write_results(args.out, results)

    print(f"system:        {args.system}")
    print(f"instances:     {report.n}")
    print(f"deterministic overall: {report.overall:.4f}")
    for qtype in sorted(report.per_type):
        print(f"  {qtype}: {report.per_type[qtype]:.4f} ({report.counts[qtype]})")
    print(f"results:       {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())