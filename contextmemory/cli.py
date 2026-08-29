"""CLI for running memory evaluations.

Subcommands:

    cm eval  --data benchmarks/data/longmemeval_oracle.json \
        --system full-history --reader-api-base https://api.openai.com/v1 \
        --reader-api-key $OPENAI_API_KEY --reader-model gpt-4o-mini \
        --out reports/runs/run.jsonl
    cm dims  --system full-history --reader-api-base ... --reader-model ...
    cm bench --system full-history

``eval`` replays a LongMemEval dataset; ``dims`` runs the custom-dimension
scenarios (write precision, evolution, forgetting); ``bench`` measures
deterministic ingest/answer latency with a null reader. Deterministic scoring
is used for iteration; pass ``--judge-model`` to ``eval`` for official-style
LLM-judged numbers.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

from contextmemory.engine.embedder import DeterministicHashEmbedder
from contextmemory.engine.extractor import LLMExtractor
from contextmemory.eval import (
    CoreMemorySystem,
    FullHistorySystem,
    MemorySystem,
    OpenAICompatClient,
    RecencyWindowSystem,
    load_longmemeval,
    replay,
    score_deterministic,
)
from contextmemory.eval.dimensions import default_scenarios, run_dimensions
from contextmemory.eval.latency import NullReader, bench_latency
from contextmemory.eval.runner import ReplayResult
from contextmemory.eval.scoring import judge_results

_SYSTEMS: dict[str, Callable[[OpenAICompatClient], MemorySystem]] = {
    "full-history": lambda reader: FullHistorySystem(reader),
    "recency-2": lambda reader: RecencyWindowSystem(reader, window=2),
    "recency-10": lambda reader: RecencyWindowSystem(reader, window=10),
    "contextmemory": lambda reader: CoreMemorySystem(
        reader,
        extractor=LLMExtractor(reader),
        embedder=DeterministicHashEmbedder(),
    ),
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


def _print_report(
    prefix: str,
    overall: float,
    per_type: dict[str, float],
    counts: dict[str, int],
) -> None:
    print(f"{prefix} {overall:.4f}")
    for qtype in sorted(per_type):
        print(f"  {qtype}: {per_type[qtype]:.4f} ({counts[qtype]})")


def _cmd_eval(args: argparse.Namespace) -> int:
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

    if args.judge_model:
        judge = OpenAICompatClient(
            base_url=args.judge_api_base or args.reader_api_base,
            api_key=args.judge_api_key or args.reader_api_key,
            model=args.judge_model,
        )
        judged_report, labeled = judge_results(results, judge)
        _write_results(args.out, labeled)
        print(f"judge model:   {args.judge_model}")
        _print_report(
            "judged overall:",
            judged_report.overall,
            judged_report.per_type,
            judged_report.counts,
        )

    print(f"results:       {args.out}")
    return 0


def _cmd_dims(args: argparse.Namespace) -> int:
    reader = OpenAICompatClient(
        base_url=args.reader_api_base,
        api_key=args.reader_api_key,
        model=args.reader_model,
    )
    factory = lambda: _SYSTEMS[args.system](reader)  # noqa: E731
    reports = run_dimensions(default_scenarios(), factory)
    print(f"system:        {args.system}")
    print(f"dimensions:    {sum(r.n_probes for r in reports)} probes")
    for report in reports:
        label = (
            f"{report.dimension} overall: {report.overall:.4f}"
            f" ({report.n_probes} probes)"
        )
        print(label)
        for result in report.scenario_results:
            for probe in result.probe_results:
                mark = "ok  " if probe.correct else "FAIL"
                abstain = "(abstain)" if probe.probe.is_abstention_probe else ""
                print(f"  [{mark}] {probe.probe.question} {abstain}")
                print(f"       hyp: {probe.hypothesis[:120]}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    factory = lambda: _SYSTEMS[args.system](NullReader())  # noqa: E731
    report = bench_latency(
        factory,
        n_sessions=args.sessions,
        probes=args.probes,
        seed=args.seed,
    )
    print(f"system:        {args.system}")
    print(f"workload:      {report.n_sessions} sessions, {report.n_probes} probes")
    print(report.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run memory evaluations.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_eval = sub.add_parser("eval", help="replay a LongMemEval dataset")
    p_eval.add_argument("--data", required=True, help="LongMemEval JSON data file")
    p_eval.add_argument("--system", required=True, choices=sorted(_SYSTEMS))
    p_eval.add_argument("--reader-api-base", required=True)
    p_eval.add_argument("--reader-api-key", default="EMPTY")
    p_eval.add_argument("--reader-model", required=True)
    p_eval.add_argument(
        "--judge-model", default=None, help="official-style LLM judge model"
    )
    p_eval.add_argument("--judge-api-base", default=None)
    p_eval.add_argument("--judge-api-key", default=None)
    p_eval.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="limit instances (0 = all)",
    )
    p_eval.add_argument("--out", required=True, help="path for hypothesis JSONL")
    p_eval.set_defaults(func=_cmd_eval)

    p_dims = sub.add_parser("dims", help="run custom-dimension scenarios")
    p_dims.add_argument("--system", required=True, choices=sorted(_SYSTEMS))
    p_dims.add_argument("--reader-api-base", required=True)
    p_dims.add_argument("--reader-api-key", default="EMPTY")
    p_dims.add_argument("--reader-model", required=True)
    p_dims.set_defaults(func=_cmd_dims)

    p_bench = sub.add_parser(
        "bench", help="measure deterministic ingest/answer latency"
    )
    p_bench.add_argument("--system", required=True, choices=sorted(_SYSTEMS))
    p_bench.add_argument("--sessions", type=int, default=200)
    p_bench.add_argument("--probes", nargs="*", default=None)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.set_defaults(func=_cmd_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())