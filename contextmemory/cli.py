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
import sys
from collections.abc import Callable
from datetime import datetime

import httpx

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


def make_reader(
    base_url: str,
    api_key: str,
    model: str,
    *,
    timeout: float = 240.0,
):
    """Build a reader for ``base_url``.

    Detects a local Ollama server (``/api/tags`` responds) and uses the native
    Ollama client, which disables Qwen3-family thinking mode and constrains
    extraction to valid JSON. The OpenAI-compatible layer cannot do either, so
    thinking models (qwen3, ...) silently burn their token budget on a hidden
    reasoning trace and return nothing. Everything else uses the standard
    OpenAI-compatible client.
    """
    from contextmemory.engine.ollama import OllamaChatClient

    root = base_url.rstrip("/")
    try:
        resp = httpx.get(f"{root}/api/tags", timeout=1.0)
        if resp.status_code == 200:
            return OllamaChatClient(root, model, timeout=timeout)
    except httpx.HTTPError:
        pass
    return OpenAICompatClient(root, api_key, model, timeout=timeout)


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

    reader = make_reader(
        args.reader_api_base, args.reader_api_key, args.reader_model
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
        judge = make_reader(
            args.judge_api_base or args.reader_api_base,
            args.judge_api_key or args.reader_api_key,
            args.judge_model,
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
    reader = make_reader(
        args.reader_api_base, args.reader_api_key, args.reader_model
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


def _cmd_demo(args: argparse.Namespace) -> int:
    from contextmemory.engine.ollama import DEFAULT_BASE_URL
    from contextmemory.tui import run_tui

    run_tui(
        offline=not args.live,
        container_tag=args.container,
        base_url=args.url or DEFAULT_BASE_URL,
        model=args.model,
        auto_launch=args.auto_launch,
    )
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from contextmemory.api import MemoryClient
    from contextmemory.tui.scenarios import seed_cells

    client = MemoryClient(args.container, embedder=None)
    for cell in seed_cells():
        client.engine.store.reconcile(cell)
    reader = make_reader(
        args.reader_api_base, args.reader_api_key, args.reader_model
    )
    answer, report = client.ask(args.question, reader, question_date=None)
    print(f"question: {args.question}")
    print(f"route:    {report.time_mode_name}")
    print(f"tokens:   {report.tokens}")
    print(f"answer:   {answer}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from contextmemory.api import MemoryClient
    from contextmemory.eval.protocol import Session, Turn

    client = MemoryClient(args.container, embedder=None)
    turns = [
        Turn(role=role, content=content)
        for role, content in (part.split(":", 1) for part in args.turn)
    ]
    session = Session(session_id=args.session_id, timestamp=datetime.now(),
                      turns=turns)
    rep = client.session(session)
    print(f"cells:   {rep.cells} (new {rep.new_cells}, dup {rep.dup_cells})")
    print(f"capture: {rep.capture_ms:.3f}ms  extract: {rep.extract_ms:.3f}ms")
    return 0


def _cmd_mcp(args: argparse.Namespace) -> int:
    from contextmemory.mcp import main as mcp_main

    return mcp_main(["--container", args.container])


def _cmd_chat(args: argparse.Namespace) -> int:
    """Chat with Ollama while ContextMemory tools are available in-process."""
    from contextmemory.engine.ollama import OllamaManager
    from contextmemory.mcp import _TOOLS, MCPServer

    manager = OllamaManager(args.url)
    if not manager.start_managed():
        print(f"Ollama unavailable: {manager.last_error}", file=sys.stderr)
        return 1
    model = args.model or (manager.list_models(refresh=True) or [None])[0]
    if not model:
        print("No Ollama models found. Pull one with: ollama pull qwen3:4b",
              file=sys.stderr)
        return 1

    reader = manager.reader(model, max_tokens=args.max_tokens)
    memory = MCPServer(container=args.container)
    tools = [
        {"type": "function", "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["inputSchema"],
        }}
        for tool in _TOOLS
    ]
    system = (
        "You are an assistant with persistent ContextMemory. Always call "
        "the memory tool for durable user facts and call recall before "
        "answering questions about the user or prior conversations. Do not "
        "claim to remember anything unless the tool returned it. Answer "
        "directly and briefly."
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    print(f"ContextMemory chat | Ollama: {model} | MCP: connected")
    print("Type /exit to quit.\n")
    try:
        while True:
            try:
                prompt = input("you> ").strip()
            except EOFError:
                break
            if not prompt:
                continue
            if prompt.lower() in {"/exit", "/quit"}:
                break
            is_question = prompt.rstrip().endswith("?")
            if not is_question and (
                prompt.lower().startswith(("i ", "i'm ", "my ", "remember "))
            ):
                stored = memory._dispatch("memory", {"content": prompt})
                prompt += f"\n\n[ContextMemory write result: {stored}]"
            if is_question:
                retrieved = memory._dispatch("recall", {"query": prompt})
                prompt += f"\n\n[ContextMemory retrieved memory:\n{retrieved}]"
            messages.append({"role": "user", "content": prompt})
            for _ in range(4):
                message = reader.chat_with_tools(messages, tools,
                                                  max_tokens=args.max_tokens)
                messages.append(message)
                calls = message.get("tool_calls") or []
                if not calls:
                    print(f"ollama> {(message.get('content') or '').strip()}\n")
                    break
                for call in calls:
                    fn = call.get("function", {})
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", {}) or {}
                    call_args = (json.loads(raw_args) if isinstance(raw_args, str)
                                 else raw_args)
                    result = memory._dispatch(name, call_args)
                    messages.append({
                        "role": "tool", "content": result,
                    })
            else:
                print("ollama> Tool loop limit reached; please try again.\n")
    finally:
        reader.close()
        manager.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="ContextMemory — memory infrastructure for AI systems.",
        prog="contextmemory",
    )
    # No subcommand → the TUI, which is the product surface.
    sub = parser.add_subparsers(dest="command")

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

    p_demo = sub.add_parser(
        "demo", help="launch the TUI brain (offline scripted demo by default)"
    )
    p_demo.add_argument("--live", action="store_true",
                        help="use a local Ollama model for extraction/answers")
    p_demo.add_argument("--container", default="brain")
    p_demo.add_argument(
        "--url", default=None,
        help="Ollama base URL (default http://localhost:11434)",
    )
    p_demo.add_argument(
        "--model", default=None,
        help="Ollama model to use (e.g. qwen3:8b); default = first pulled model",
    )
    p_demo.add_argument(
        "--auto-launch", action="store_true",
        help="launch `ollama serve` under the hood if not already running",
    )
    p_demo.set_defaults(func=_cmd_demo)

    p_ask = sub.add_parser("ask", help="ask the memory brain a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--container", default="brain")
    p_ask.add_argument("--reader-api-base", required=True)
    p_ask.add_argument("--reader-api-key", default="EMPTY")
    p_ask.add_argument("--reader-model", required=True)
    p_ask.set_defaults(func=_cmd_ask)

    p_ingest = sub.add_parser(
        "ingest", help="ingest a turn into memory (role:content)"
    )
    p_ingest.add_argument("--container", default="brain")
    p_ingest.add_argument("--session-id", default="s1")
    p_ingest.add_argument("--turn", action="append", required=True)
    p_ingest.set_defaults(func=_cmd_ingest)

    p_mcp = sub.add_parser(
        "mcp", help="run the MCP server (stdio) for AI tool bridges"
    )
    p_mcp.add_argument("--container", default="brain")
    p_mcp.set_defaults(func=_cmd_mcp)

    p_chat = sub.add_parser(
        "chat", help="chat with Ollama using ContextMemory MCP tools"
    )
    p_chat.add_argument("--model", default=None,
                        help="Ollama model (default: first installed model)")
    p_chat.add_argument("--url", default="http://localhost:11434")
    p_chat.add_argument("--container", default="brain")
    p_chat.add_argument("--max-tokens", type=int, default=512)
    p_chat.set_defaults(func=_cmd_chat)

    args = parser.parse_args(argv)
    if args.command is None:
        # `contextmemory` with no subcommand → the TUI product surface.
        return _cmd_demo(argparse.Namespace(
            live=False, container="brain", url=None, model=None,
            auto_launch=False,
        ))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
