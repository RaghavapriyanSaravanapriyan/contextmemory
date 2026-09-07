"""Official-protocol benchmark runs on the local 1B rig (qwen2.5:1.5b).

Suites (all head-to-head on one rig, identical reader model):

* ``beam`` — official BEAM-100K conversations (Mohammadta/BEAM, ICLR 2026),
  all 10 ability categories, official unified nugget-judge prompt replicated
  verbatim (judge model: qwen2.5:1.5b, disclosed substitution for GPT).
* ``longmemeval`` — official oracle instances, official answer-check judge
  prompts (see ``contextmemory.eval.scoring``), deterministic + judged scores.
* ``locomo`` — official locomo10.json conversations, full QA sets.

Systems: ``contextmemory`` (C++ ETMC + LLM extraction) vs ``full-history``
vs ``recency`` baselines. At BEAM scale (130K tokens/convo) full-history
cannot fit a 1B local reader, so BEAM uses a ``recency-8k`` baseline instead
(disclosed below and in the report).

Checkpoints are appended as JSONL per answer, so aborted runs keep evidence.

Usage:
    uv run python benchmarks/run_official.py beam --convos 0 1
    uv run python benchmarks/run_official.py longmemeval --n 30
    uv run python benchmarks/run_official.py locomo --convos 0 1 2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextmemory.engine.embedder import DeterministicHashEmbedder
from contextmemory.engine.extractor import LLMExtractor
from contextmemory.engine.ollama import OllamaChatClient
from contextmemory.eval.data import load_longmemeval
from contextmemory.eval.dimensions import is_abstention
from contextmemory.eval.protocol import Session, Turn
from contextmemory.eval.runner import replay
from contextmemory.eval.scoring import (
    deterministic_match,
    judge_results,
    score_deterministic,
)
from contextmemory.eval.systems import (
    CoreMemorySystem,
    FullHistorySystem,
    RecencyCharsSystem,
    RecencyWindowSystem,
)

MODEL = "qwen2.5:1.5b"
BASE_URL = "http://localhost:11434"
RESULTS = Path("benchmarks/results")

def make_third_party(name: str, reader, container_tag: str):
    """Build a third-party contender or exit (fail closed, never scored).

    Third parties run through the SAME replay runner, SAME reader model,
    and SAME judge as every local system. Only retrieval/storage is theirs.
    Import is lazy so the stdlib path never hard-depends on vendor SDKs.
    """
    if name == "supermemory":
        from benchmarks.adapters import SkipError, SupermemorySystem

        try:
            return SupermemorySystem(reader, container_tag=container_tag)
        except SkipError as exc:
            raise SystemExit(
                f"supermemory skipped (fail closed, not scored): {exc}"
            ) from exc
    raise SystemExit(f"unknown third-party system {name!r} "
                     f"(known: supermemory)")


# Explicit lineup resolution. Unknown names exit LOUDLY: the old code
# silently mapped any typo to a recency baseline, which could print a
# misleading system label next to recency numbers. No silent fallbacks.
BASELINES = {"full-history", "recency", "recency-2", "recency-8k"}


def resolve_baseline(name: str, reader, *, chars: bool):
    if name == "full-history":
        return FullHistorySystem(reader)
    if name in ("recency", "recency-8k"):
        return RecencyCharsSystem(reader, max_chars=8000)
    if name == "recency-2":
        return RecencyWindowSystem(reader, window=2)
    raise SystemExit(
        f"unknown system {name!r} "
        f"(known: contextmemory, supermemory, {sorted(BASELINES)})")

# --- official BEAM unified judge prompt (replicated from the BEAM repo's
# src/prompts.py; only the model is substituted: qwen2.5:1.5b for GPT). ---
BEAM_JUDGE_PROMPT = """You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): <rubric_item>
- QUESTION: <question>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms. Treat them as equal when numerically equivalent.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure.
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
- **0.5 (Partial Compliance)**: Partially complies (minor inaccuracies/incomplete execution).
- **0.0 (No Compliance)**: Fails to comply (required element missing/incorrect, prohibited element present, or non-responsive).

Respond with ONLY a JSON object in exactly this shape:
{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation]"
}

NOTE: ONLY output the json object, without any explanation before or after that"""


def beam_judge_score(
    reader: OllamaChatClient,
    question: str,
    rubric_item: str,
    response: str,
) -> float:
    """Score one rubric item 0/0.5/1 with the official BEAM judge prompt."""
    prompt = (
        BEAM_JUDGE_PROMPT.replace("<rubric_item>", rubric_item)
        .replace("<question>", question)
        .replace("<llm_response>", response[:2000])
    )
    try:
        raw = reader.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            json_mode=True,
        )
        data = json.loads(raw)
        score = float(data.get("score", 0.0))
    except Exception:
        score = 0.0
        try:
            m = re.search(r"([01](?:\.0|\.5)?)", raw)
            score = float(m.group(1)) if m else 0.0
        except Exception:
            score = 0.0
    return min(1.0, max(0.0, score))


def _parse_anchor(value: str, fallback: datetime) -> datetime:
    try:
        return datetime.strptime(value.strip(), "%B-%d-%Y")
    except (ValueError, AttributeError):
        return fallback


def load_beam_convo(bucket_file: str, idx: int):
    """Load one BEAM conversation: (sessions, probes).

    Sessions chunk the time-ordered chat into ~10-turn blocks so LLM
    extraction runs bounded per call; probes cover all 10 abilities.
    """
    import pandas as pd

    df = pd.read_parquet(bucket_file)
    row = df.iloc[idx]
    batches = list(row["chat"])
    turns = [t for b in batches for t in (list(b) if not isinstance(b, dict) else [b])]
    base = datetime(2024, 3, 1)
    sessions: list[Session] = []
    for s, i in enumerate(range(0, len(turns), 10)):
        block = turns[i : i + 10]
        anchor = base
        for t in block:
            if isinstance(t, dict) and t.get("time_anchor"):
                anchor = _parse_anchor(t["time_anchor"], anchor)
                break
        sessions.append(
            Session(
                session_id=f"beam-{row['conversation_id']}-s{s}",
                timestamp=anchor + timedelta(hours=s),
                turns=[
                    Turn(
                        role="user" if t.get("role") == "user" else "assistant",
                        content=str(t.get("content", ""))[:2000],
                    )
                    for t in block
                    if isinstance(t, dict) and t.get("content")
                ],
            )
        )
    pq = row["probing_questions"]
    if isinstance(pq, str):
        import ast as _ast

        pq = _ast.literal_eval(pq)
    probes: list[dict] = []
    for ability, items in pq.items():
        for it in items:
            gold = (
                it.get("answer") or it.get("ideal_answer")
                or it.get("ideal_response") or it.get("ideal_summary")
                or it.get("expected_compliance") or ""
            )
            probes.append({
                "ability": ability,
                "question": it.get("question", ""),
                "gold": str(gold),
                "rubric": list(it.get("rubric", []) or []),
            })
    qdate = base + timedelta(days=120)
    return sessions, probes, qdate, str(row["conversation_id"])


def make_reader(num_ctx: int = 2048) -> OllamaChatClient:
    r = OllamaChatClient(BASE_URL, MODEL, timeout=300.0, max_tokens=256,
                         num_ctx=num_ctx)
    r.warm()
    return r


def ckpt(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")


def cmd_beam(args: argparse.Namespace) -> int:
    bucket = f"benchmarks/data/beam/data/{args.bucket}-00000-of-00001.parquet"
    reader = make_reader()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"beam100k_{ts}.jsonl"
    systems = args.systems
    summary: dict = {}

    for ci in args.convos:
        sessions, probes, qdate, cid = load_beam_convo(bucket, ci)
        print(f"convo {cid}: {len(sessions)} sessions, {len(probes)} probes",
              flush=True)
        for name in systems:
            if name == "contextmemory":
                sys_obj = CoreMemorySystem(
                    reader, extractor=LLMExtractor(reader),
                    embedder=DeterministicHashEmbedder(),
                    container_tag=f"beam-{cid}",
                )
            elif name == "supermemory":
                sys_obj = make_third_party(name, reader, f"sm-beam-{cid}")
            else:
                sys_obj = resolve_baseline(name, reader, chars=True)
            t0 = time.perf_counter()
            for s in sessions:
                sys_obj.ingest(s)
            ingest_s = time.perf_counter() - t0
            scores: list[float] = []
            for p in probes:
                t1 = time.perf_counter()
                hyp = sys_obj.answer(p["question"], qdate)
                ans_s = time.perf_counter() - t1
                item_scores = [
                    beam_judge_score(reader, p["question"], rub, hyp)
                    for rub in p["rubric"]
                ] if p["rubric"] else [0.0]
                mean_s = sum(item_scores) / max(1, len(item_scores))
                det = deterministic_match(hyp, p["gold"][:500])
                scores.append(mean_s)
                ckpt(out, {"convo": cid, "system": name,
                           "ability": p["ability"], "q": p["question"],
                           "gold": p["gold"][:500], "hyp": hyp[:800],
                           "judge": round(mean_s, 3), "det": det,
                           "ingest_s": round(ingest_s, 1),
                           "answer_s": round(ans_s, 2)})
            avg = sum(scores) / max(1, len(scores))
            summary[f"{cid}/{name}"] = round(avg, 4)
            print(f"  {name}: judge {avg:.3f} (ingest {ingest_s:.0f}s)",
                  flush=True)
    print(json.dumps(summary, indent=1))
    print(f"checkpoint: {out}")
    reader.close()
    return 0


def cmd_longmemeval(args: argparse.Namespace) -> int:
    instances = load_longmemeval("benchmarks/data/longmemeval_oracle.json")
    # Stratified spread across question types (deterministic, no RNG).
    by_type: dict[str, list] = {}
    for i in instances:
        by_type.setdefault(i.question_type, []).append(i)
    subset = []
    per = max(1, args.n // max(1, len(by_type)))
    for qtype in sorted(by_type):
        pool = by_type[qtype]
        step = max(1, len(pool) // per)
        subset.extend(pool[j * step] for j in range(min(per, len(pool))))
    subset = subset[: args.n]
    print(f"{len(subset)} instances", flush=True)

    reader = make_reader(num_ctx=8192)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"longmemeval_{ts}.json"

    def factory(name: str):
        if name == "contextmemory":
            return CoreMemorySystem(
                reader, extractor=LLMExtractor(reader),
                embedder=DeterministicHashEmbedder(),
                container_tag="eval-official")
        if name == "supermemory":
            return make_third_party(name, reader, "sm-eval-official")
        return resolve_baseline(name, reader, chars=False)

    report: dict = {}
    for name in args.systems:
        sys_ckpt = RESULTS / f"longmemeval_{ts}_{name}.jsonl"
        results = replay(subset, lambda n=name: factory(n), progress=False)
        # Checkpoint raw replay evidence BEFORE scoring so a later crash
        # (judge, scoring) never loses the LLM work.
        for r in results:
            ckpt(sys_ckpt, {"qid": r.question_id, "type": r.question_type,
                            "q": r.question, "gold": str(r.answer),
                            "hyp": r.hypothesis,
                            "ingest_s": round(r.timing.ingest_s, 2),
                            "answer_s": round(r.timing.answer_s, 2)})
        det = score_deterministic(results)
        entry: dict = {
            "det_overall": round(det.overall, 4),
            "det_per_type": {k: round(v, 4) for k, v in det.per_type.items()},
            "answer_s": round(sum(r.timing.answer_s for r in results), 1),
            "ingest_s": round(sum(r.timing.ingest_s for r in results), 1),
        }
        if args.judge:
            judged, labeled = judge_results(results, reader)
            entry["judge_overall"] = round(judged.overall, 4)
            entry["judge_per_type"] = {
                k: round(v, 4) for k, v in judged.per_type.items()}
            entry["hyps"] = [
                {"qid": r.question_id, "type": r.question_type,
                 "gold": r.answer, "hyp": r.hypothesis[:600],
                 "judged": r.judged} for r in labeled
            ]
        report[name] = entry
        print(f"  {name}: det {det.overall:.3f} " +
              (f"judge {entry.get('judge_overall')}" if args.judge else ""),
              flush=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")
    reader.close()
    return 0


def cmd_locomo(args: argparse.Namespace) -> int:
    with open("benchmarks/memorybench/data/benchmarks/locomo/"
              "locomo10.json", encoding="utf-8") as fh:
        data = json.load(fh)
    # num_ctx 12K so the full-history baseline sees the whole ~9K-token convo.
    reader = make_reader(num_ctx=12288)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"locomo_{ts}.jsonl"
    summary: dict = {}

    for ci in args.convos:
        convo = data[ci]
        conv = convo["conversation"]
        import re as _re

        keys = sorted(
            (int(m.group(1)), k) for k in conv
            for m in [_re.fullmatch(r"session_(\d+)", k)]
            if m and isinstance(conv[k], list))
        sessions = [
            Session(
                session_id=f"locomo{ci}-s{n}", timestamp=datetime(2023, 5, 1),
                turns=[Turn(
                    role="user" if t["speaker"] == conv.get("speaker_a")
                    else "assistant",
                    content=f"{t['speaker']}: {t['text']}") for t in conv[k]])
            for n, k in keys
        ]
        for name in args.systems:
            if name == "contextmemory":
                sys_obj = CoreMemorySystem(
                    reader, extractor=LLMExtractor(reader),
                    embedder=DeterministicHashEmbedder(),
                    container_tag=f"locomo-{ci}")
            elif name == "supermemory":
                sys_obj = make_third_party(name, reader, f"sm-locomo-{ci}")
            else:
                sys_obj = resolve_baseline(name, reader, chars=False)
            t0 = time.perf_counter()
            for s in sessions:
                sys_obj.ingest(s)
            ingest_s = time.perf_counter() - t0
            correct, total = 0, 0
            for q in convo["qa"]:
                hyp = sys_obj.answer(q["question"], datetime(2023, 6, 1))
                # Category 5 is adversarial: most items carry only an
                # `adversarial_answer` (the plausible trap) and no gold
                # answer — the correct behavior is abstention. Items with a
                # gold `answer` are scored by containment.
                if "answer" in q:
                    ok = deterministic_match(hyp, str(q["answer"]))
                    gold = str(q["answer"])
                else:
                    ok = is_abstention(hyp)
                    gold = f"<abstain; trap={q.get('adversarial_answer')}>"
                correct += ok
                total += 1
                ckpt(out, {"convo": ci, "system": name, "cat": q.get("category"),
                           "q": q["question"], "gold": gold,
                           "hyp": hyp[:500], "ok": ok})
            acc = correct / max(1, total)
            summary[f"{ci}/{name}"] = f"{correct}/{total}={acc:.3f}"
            print(f"  convo {ci} {name}: {correct}/{total}={acc:.3f} "
                  f"(ingest {ingest_s:.0f}s)", flush=True)
    print(json.dumps(summary, indent=1))
    print(f"checkpoint: {out}")
    reader.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Official benchmark runs (1B rig)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("beam", help="BEAM-100K subset, official judge prompt")
    b.add_argument("--bucket", default="100K")
    b.add_argument("--convos", nargs="+", type=int, default=[0, 1])
    b.add_argument("--systems", nargs="+",
                   default=["contextmemory", "recency"])
    b.set_defaults(func=cmd_beam)

    lv = sub.add_parser("longmemeval", help="official oracle replay + judge")
    lv.add_argument("--n", type=int, default=30)
    lv.add_argument("--systems", nargs="+",
                    default=["contextmemory", "full-history", "recency-2"])
    lv.add_argument("--judge", action="store_true")
    lv.set_defaults(func=cmd_longmemeval)

    m = sub.add_parser("locomo", help="official locomo10 full-QA replay")
    m.add_argument("--convos", nargs="+", type=int, default=[0, 1, 2])
    m.add_argument("--systems", nargs="+",
                   default=["contextmemory", "full-history", "recency-2"])
    m.set_defaults(func=cmd_locomo)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
