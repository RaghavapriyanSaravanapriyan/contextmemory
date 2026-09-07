"""Deterministic memory-layer scale sweep + Ollama streaming check.

* Scale: ingest N synthetic sessions through the real C++ reconcile path
  (deterministic embedder, no LLM), then measure recall p50/p95 for a
  projection-hit query and a miss query at each scale.
* Streaming: time-to-first-token vs full-generation total on qwen2.5:1.5b
  for a short and a ~120-token answer.

Usage: uv run python benchmarks/scale_sweep.py
Writes benchmarks/results/scale_<ts>.json.
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextmemory.engine.embedder import DeterministicHashEmbedder
from contextmemory.engine.memory import MemoryEngine
from contextmemory.eval.protocol import Session, Turn


def sweep(scales: list[int]) -> dict:
    out: dict = {}
    for n in scales:
        eng = MemoryEngine(f"sweep-{n}",
                           embedder=DeterministicHashEmbedder())
        base = datetime(2024, 1, 1)
        t0 = time.perf_counter()
        for i in range(n):
            eng.ingest(Session(
                session_id=f"w{i}", timestamp=base + timedelta(minutes=i),
                turns=[Turn(
                    role="user",
                    content=f"Session {i}: the user fact alpha-{i % 50} "
                            f"beta-{i} gamma-{i % 7}.")] ))
        ingest_ms = (time.perf_counter() - t0) * 1000
        res: dict = {"cells": eng.store.cell_count,
                     "ingest_total_ms": round(ingest_ms, 1),
                     "ingest_per_cell_ms": round(
                         ingest_ms / max(1, eng.store.cell_count), 4)}
        for label, q in [("hit", "What is alpha-7 beta status?"),
                         ("miss", "What is the user's pet name?")]:
            samples = []
            for _ in range(25):
                s = time.perf_counter()
                eng.recall(q, base + timedelta(days=400), top_k=8)
                samples.append((time.perf_counter() - s) * 1000)
            samples.sort()
            res[label] = {"p50_ms": round(st.median(samples), 3),
                          "p95_ms": round(samples[int(0.95 * len(samples))],
                                          3)}
        out[n] = res
        print(f"  n={n}: cells={res['cells']} "
              f"ingest/cell={res['ingest_per_cell_ms']}ms "
              f"hit p50={res['hit']['p50_ms']}ms "
              f"miss p50={res['miss']['p50_ms']}ms", flush=True)
    return out


def streaming() -> dict:
    from contextmemory.engine.ollama import OllamaChatClient

    c = OllamaChatClient("http://localhost:11434", "qwen2.5:1.5b",
                         timeout=180.0, max_tokens=256)
    c.warm()
    res: dict = {}
    cases = {
        "short": "Say the word Seattle and nothing else.",
        "long": "In three sentences, explain why Seattle is known "
                "for rain and coffee.",
    }
    for label, text in cases.items():
        msg = [{"role": "user", "content": text}]
        cap = 16 if label == "short" else 120
        t0 = time.perf_counter()
        full = c.complete(msg, max_tokens=cap)
        total_ns = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        first, chunks = None, []
        for d in c.stream_complete(msg, max_tokens=cap):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            chunks.append(d)
        total_s = (time.perf_counter() - t0) * 1000
        res[label] = {
            "nonstream_total_ms": round(total_ns, 1),
            "stream_ttft_ms": round(first or -1, 1),
            "stream_total_ms": round(total_s, 1),
            "ttft_speedup": round(total_ns / max(1, first or 1), 1),
            "stream_chars": len("".join(chunks)),
            "nonstream_chars": len(full),
        }
        print(f"  {label}: nonstream {total_ns:.0f}ms vs "
              f"TTFT {first:.0f}ms "
              f"({res[label]['ttft_speedup']}x)", flush=True)
    c.close()
    return res


def main() -> int:
    print("scale sweep (deterministic C++ path):", flush=True)
    report = {"ts": datetime.now().isoformat(timespec="seconds"),
              "model": "qwen2.5:1.5b",
              "scale": sweep([100, 500, 2000, 10000])}
    print("streaming check:", flush=True)
    report["streaming"] = streaming()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(f"benchmarks/results/scale_{ts}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
