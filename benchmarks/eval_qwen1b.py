"""Head-to-head memory-layer eval on the local 1B rig (qwen2.5:1.5b).

Three suites, one shared rig, identical reader model — the "beat SOTA"
requirement is head-to-head on standard benchmarks, not vibes:

* Suite A — LongMemEval-style (oracle subset, stratified by question type):
  knowledge updates, temporal reasoning, user-state recall.
* Suite B — LoCoMo-style (locomo10 convo, multi-session conversational
  recall across ~35 sessions).
* Suite C — BEAM-style: C1 large-scale latency (deterministic ingest +
  search p50/p95 at 100/500/2000 cells) and C2 temporal evolution accuracy.

Systems: contextmemory (C++ ETMC + LLM extraction) vs full-history
(upper-bound: everything in context) vs recency-2 (no-memory floor).

Usage:
    uv run python benchmarks/eval_qwen1b.py [--instances 12] [--locomo-qa 12]

Writes benchmarks/results/qwen1b_<ts>.json and prints a markdown table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextmemory.engine.embedder import DeterministicHashEmbedder
from contextmemory.engine.extractor import LLMExtractor
from contextmemory.engine.ollama import OllamaChatClient
from contextmemory.eval.data import QuestionInstance, load_longmemeval
from contextmemory.eval.dimensions import run_dimensions
from contextmemory.eval.latency import NullReader, bench_latency
from contextmemory.eval.protocol import Session, Turn
from contextmemory.eval.runner import replay
from contextmemory.eval.scoring import deterministic_match, score_deterministic
from contextmemory.eval.systems import (
    CoreMemorySystem,
    FullHistorySystem,
    RecencyWindowSystem,
)

MODEL = "qwen2.5:1.5b"
BASE_URL = "http://localhost:11434"

# LongMemEval oracle -> our deterministic Session mapping needs no change;
# instances already carry haystack sessions + question dates.


def pick_stratified(instances: list[QuestionInstance], n: int) -> list:
    quota = {
        "temporal-reasoning": 3,
        "multi-session": 3,
        "knowledge-update": 3,
        "single-session-user": 1,
        "single-session-assistant": 1,
        "single-session-preference": 1,
    }
    by_type: dict[str, list] = defaultdict(list)
    for i in instances:
        by_type[i.question_type].append(i)
    picked: list = []
    for qtype, k in quota.items():
        # Deterministic spread across the file (no RNG seed games).
        pool = by_type[qtype]
        step = max(1, len(pool) // k)
        picked.extend(pool[j * step] for j in range(min(k, len(pool))))
    return picked[:n]


def load_locomo(path: str, convo_idx: int = 0, n_qa: int = 12):
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    convo = raw[convo_idx]
    sessions: list[Session] = []
    conv = convo["conversation"]
    # session_<k> keys with list values are dialogue; trailing date-only
    # keys (no session text) are skipped. Iterate all keys numerically so a
    # gap never truncates the tail.
    import re as _re

    keys = sorted(
        (int(m.group(1)), k)
        for k in conv
        for m in [_re.fullmatch(r"session_(\d+)", k)]
        if m and isinstance(conv[k], list)
    )
    for n, k in keys:
        turns = [
            Turn(role="user" if t["speaker"] == conv.get("speaker_a",
                                                          "Caroline")
                 else "assistant",
                 content=f"{t['speaker']}: {t['text']}")
            for t in conv[k]
        ]
        sessions.append(Session(session_id=f"locomo-s{n}",
                                timestamp=datetime(2023, 5, 1),
                                turns=turns))
    qa = convo["qa"]
    by_cat: dict[int, list] = defaultdict(list)
    for q in qa:
        by_cat[q.get("category", 0)].append(q)
    picked = []
    cats = sorted(by_cat)
    per = max(1, n_qa // max(1, len(cats)))
    for cat in cats:
        pool = by_cat[cat]
        step = max(1, len(pool) // per)
        picked.extend(pool[j * step] for j in range(min(per, len(pool))))
    # Fill any remainder deterministically across categories.
    if len(picked) < n_qa:
        seen_ids = {id(q) for q in picked}
        for cat in cats:
            for q in by_cat[cat]:
                if len(picked) >= n_qa:
                    break
                if id(q) not in seen_ids:
                    picked.append(q)
                    seen_ids.add(id(q))
            if len(picked) >= n_qa:
                break
    return sessions, picked[:n_qa]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--locomo-qa", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reader = OllamaChatClient(BASE_URL, MODEL, timeout=300.0, max_tokens=256)
    print(f"warming {MODEL} ...", flush=True)
    reader.warm()

    def factory(name: str):
        if name == "contextmemory":
            return CoreMemorySystem(
                reader,
                extractor=LLMExtractor(reader),
                embedder=DeterministicHashEmbedder(),
                container_tag="eval-qwen1b",
            )
        if name == "full-history":
            return FullHistorySystem(reader)
        return RecencyWindowSystem(reader, window=2)

    systems = ["contextmemory", "full-history", "recency-2"]
    report: dict = {
        "model": MODEL,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "suites": {},
    }

    # ---- Suite A: LongMemEval-style -------------------------------------
    oracle = load_longmemeval("benchmarks/data/longmemeval_oracle.json")
    subset = pick_stratified(oracle, args.instances)
    print(f"Suite A: {len(subset)} LongMemEval-oracle instances "
          f"{Counter(i.question_type for i in subset)}", flush=True)
    suite_a: dict = {}
    for name in systems:
        t0 = time.perf_counter()
        results = replay(subset, lambda n=name: factory(n), progress=False)
        wall = time.perf_counter() - t0
        scored = score_deterministic(results)
        suite_a[name] = {
            "overall": round(scored.overall, 4),
            "per_type": {k: round(v, 4) for k, v in scored.per_type.items()},
            "counts": scored.counts,
            "wall_s": round(wall, 1),
            "ingest_s": round(sum(r.timing.ingest_s for r in results), 1),
            "answer_s": round(sum(r.timing.answer_s for r in results), 1),
            "hyps": [
                {"qid": r.question_id, "type": r.question_type,
                 "gold": r.answer, "hyp": r.hypothesis[:300]}
                for r in results
            ],
        }
        print(f"  {name}: {scored.overall:.3f} "
              f"({suite_a[name]['answer_s']}s answer)", flush=True)
    report["suites"]["A_longmemeval"] = suite_a

    # ---- Suite B: LoCoMo-style ------------------------------------------
    sessions, qa = load_locomo(
        "benchmarks/memorybench/data/benchmarks/locomo/locomo10.json",
        n_qa=args.locomo_qa,
    )
    print(f"Suite B: {len(sessions)} sessions, {len(qa)} QA", flush=True)
    suite_b: dict = {}
    for name in systems:
        sys_obj = factory(name)
        t0 = time.perf_counter()
        for s in sessions:
            sys_obj.ingest(s)
        ingest_s = time.perf_counter() - t0
        correct = 0
        t0 = time.perf_counter()
        details = []
        for q in qa:
            hyp = sys_obj.answer(q["question"], datetime(2023, 6, 1))
            ok = deterministic_match(hyp, str(q["answer"]))
            correct += ok
            details.append({"q": q["question"], "gold": str(q["answer"]),
                            "hyp": hyp[:300], "ok": ok,
                            "cat": q.get("category")})
        answer_s = time.perf_counter() - t0
        suite_b[name] = {
            "overall": round(correct / max(1, len(qa)), 4),
            "correct": correct,
            "n": len(qa),
            "ingest_s": round(ingest_s, 1),
            "answer_s": round(answer_s, 1),
            "details": details,
        }
        print(f"  {name}: {correct}/{len(qa)} "
              f"({suite_b[name]['overall']:.3f})", flush=True)
    report["suites"]["B_locomo"] = suite_b

    # ---- Suite C1: BEAM-style scale -------------------------------------
    print("Suite C1: scale latency (deterministic, NullReader)", flush=True)
    from contextmemory.eval.systems import CoreMemorySystem as CMS

    def cm_null():
        return CMS(NullReader(), embedder=DeterministicHashEmbedder(),
                   container_tag="scale")

    lat = bench_latency(cm_null, n_sessions=200,
                        probes=["Where does the user live?",
                                "What is the user's job?",
                                "Summarize weekend plans.",
                                "What was budgeted?"])
    # Extra: 2000-cell search sweep on one store.
    from contextmemory.engine.memory import MemoryEngine

    eng = MemoryEngine("sweep", embedder=DeterministicHashEmbedder())
    base = datetime(2024, 1, 1)
    for i in range(2000):
        eng.ingest(Session(session_id=f"w{i}", timestamp=base,
                           turns=[Turn(role="user",
                                       content=f"Session {i}: user fact "
                                               f"alpha-{i % 50} beta-{i}.")]))
    import statistics as _st

    sweep = {}
    for label, qs in [("hit", "What is alpha-7 beta status?"),
                      ("miss", "What is the user's pet name?")]:
        samples = []
        for _ in range(25):
            s = time.perf_counter()
            eng.recall(qs, base, top_k=8)
            samples.append((time.perf_counter() - s) * 1000)
        samples.sort()
        sweep[label] = {
            "p50_ms": round(_st.median(samples), 3),
            "p95_ms": round(samples[int(0.95 * len(samples))], 3),
            "n_cells": eng.store.cell_count,
        }
    report["suites"]["C_beam_scale"] = {
        "bench200": {"ingest_p50_ms": round(lat.ingest.p50_ms, 3),
                     "ingest_p95_ms": round(lat.ingest.p95_ms, 3),
                     "answer_p50_ms": round(lat.answer.p50_ms, 3),
                     "answer_p95_ms": round(lat.answer.p95_ms, 3)},
        "sweep2000": sweep,
    }
    print(f"  bench200 answer p50 {lat.answer.p50_ms:.3f}ms; "
          f"sweep2000 hit p50 {sweep['hit']['p50_ms']}ms", flush=True)

    # ---- Suite C2: BEAM-style temporal ----------------------------------
    print("Suite C2: temporal evolution (dims, qwen reader)", flush=True)
    from contextmemory.eval.dimensions import default_scenarios

    suite_c2: dict = {}
    for name in systems:
        reps = run_dimensions(default_scenarios(),
                              lambda n=name: factory(n))
        suite_c2[name] = {
            r.dimension: {"overall": round(r.overall, 4), "n": r.n_probes}
            for r in reps
        }
        print(f"  {name}: {suite_c2[name]}", flush=True)
    report["suites"]["C_temporal"] = suite_c2

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or f"benchmarks/results/qwen1b_{ts}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")

    # Markdown table for the chat summary.
    print("\n## Suite A — LongMemEval-style (qwen2.5:1.5b, n=12)")
    print("| system | overall | ingest_s | answer_s |")
    print("|---|---|---|---|")
    for name in systems:
        r = suite_a[name]
        print(f"| {name} | {r['overall']:.3f} | {r['ingest_s']} | "
              f"{r['answer_s']} |")
    print("\n## Suite B — LoCoMo-style (35 sessions, 12 QA)")
    print("| system | acc | correct | ingest_s | answer_s |")
    print("|---|---|---|---|---|")
    for name in systems:
        r = suite_b[name]
        print(f"| {name} | {r['overall']:.3f} | {r['correct']}/{r['n']} | "
              f"{r['ingest_s']} | {r['answer_s']} |")
    print("\n## Suite C — BEAM-style")
    b = report["suites"]["C_beam_scale"]
    print(f"deterministic answer p50 {b['bench200']['answer_p50_ms']}ms "
          f"(200 sessions); 2000-cell sweep hit p50 "
          f"{b['sweep2000']['hit']['p50_ms']}ms / miss p50 "
          f"{b['sweep2000']['miss']['p50_ms']}ms")
    print("temporal dims:", suite_c2)
    reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
