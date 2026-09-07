#!/usr/bin/env python3
"""cmbench — one-command memory benchmark runner (stdlib only).

Downloads, installs, and runs the memory benchmarks on ANY machine
(Windows / macOS / Linux) with the SAME model for every system, serially,
with live streaming output and a final metrics table.

Single-command usage (from anywhere)::

    Linux/macOS:
      git clone https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git \\
        && cd contextmemory && python3 scripts/cmbench.py --model qwen3:4b

    Windows (PowerShell):
      git clone https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git;
        cd contextmemory; py scripts/cmbench.py --model qwen3:4b

What it does, in order:
  1. ENV    — checks Python >= 3.11, installs ContextMemory (pip -e .,
             builds the C++ core; needs cmake+ninja+compiler with hints).
  2. MODEL  — probes the reader (--base-url, Ollama by default), lists
             models, pulls --model on request.
  3. DATA   — fetches missing official datasets (LongMemEval oracle/S,
             LoCoMo, BEAM parquet) with MB progress bars.
  4. RUN    — runs suites SERIALLY (dims, bench, longmemeval, locomo,
             beam), every system on the SAME model, stdout streamed live.
  5. REPORT — checkpoints JSONL under reports/runs/cmbench-<ts>/ plus a
             final metrics table (accuracy, tokens, latency).

Optional head-to-heads (no key = cleanly skipped, never faked):
  --with-supermemory  clone supermemoryai/supermemory for reference and, if
                      SUPERMEMORY_API_KEY is set, add it to the lineup.
  --with-mem0         add Mem0 to the lineup if `mem0ai` is installed and
                      OPENAI_API_KEY (or MEM0_API_KEY) is set.

Only stdlib is used so the script runs before anything is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = "https://github.com/RaghavapriyanSaravanapriyan/contextmemory.git"
SUPERMEMORY_REPO = "https://github.com/supermemoryai/supermemory.git"

DATASETS = {
    # dest (relative to repo root) -> download URL
    "benchmarks/data/longmemeval_oracle.json": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        "/resolve/main/longmemeval_oracle.json"
    ),
    "benchmarks/data/longmemeval_s_cleaned.json": (
        "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        "/resolve/main/longmemeval_s_cleaned.json"
    ),
    "benchmarks/data/locomo10.json": (
        "https://raw.githubusercontent.com/snap-research/locomo"
        "/main/data/locomo10.json"
    ),
    "benchmarks/data/beam/data/100K-00000-of-00001.parquet": (
        "https://huggingface.co/datasets/Mohammadta/BEAM"
        "/resolve/main/data/100K-00000-of-00001.parquet"
    ),
}

SUITE_NEEDS = {
    "dims": [],
    "bench": [],
    "longmemeval": ["benchmarks/data/longmemeval_oracle.json"],
    "locomo": ["benchmarks/data/locomo10.json"],
    "beam": ["benchmarks/data/beam/data/100K-00000-of-00001.parquet"],
}

SUITE_ORDER = ["dims", "bench", "longmemeval", "locomo", "beam"]


def log(msg: str) -> None:
    print(f"[cmbench] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[cmbench] ERROR: {msg}", file=sys.stderr, flush=True)


def run(cmd: list[str], cwd: Path, stream: bool = True) -> int:
    """Run a command, streaming output live. Returns the exit code."""
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if stream:
            print(line, end="", flush=True)
    return proc.wait()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"download {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "cmbench/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        last = time.monotonic()
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            now = time.monotonic()
            if total and now - last > 1.0:
                log(f"  {got / 1e6:.1f} / {total / 1e6:.1f} MB")
                last = now
    if total and tmp.stat().st_size < total:
        raise RuntimeError(f"truncated download: {dest.name}")
    tmp.replace(dest)
    log(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def phase_env(root: Path, yes: bool) -> None:
    log("== phase 1/5: environment ==")
    if sys.version_info < (3, 11):  # noqa: UP036 - runtime guard for old Pythons
        sys.exit("Python >= 3.11 required, found " + sys.version.split()[0])
    try:
        import contextmemory  # noqa: F401
        log("contextmemory importable")
        return
    except ImportError:
        pass
    log("installing contextmemory (builds the C++ core; first run ~1-3 min)")
    for tool, hint in (
        ("cmake", "install cmake: apt/brew/choco install cmake"),
        ("ninja", "install ninja: apt/brew/choco install ninja"),
    ):
        if shutil.which(tool) is None:
            log(f"WARNING: {tool} not found — {hint}")
    installers = []
    if shutil.which("uv") is not None:
        installers.append(["uv", "pip", "install", "-e", "."])
    installers.append([sys.executable, "-m", "pip", "install", "-e", ".",
                       "--disable-pip-version-check"])
    installers.append([sys.executable, "-m", "pip", "install", "-e", ".",
                       "--disable-pip-version-check", "--break-system-packages"])
    for cmd in installers:
        if run(cmd, root) == 0:
            break
    else:
        sys.exit("pip install failed — install a C++/CMake/Ninja toolchain "
                 "and re-run.")
    # Dev extras for pytest parity are optional; pandas only for BEAM.
    _ = yes


def ensure_pandas(root: Path, yes: bool) -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        pass
    log("BEAM suite needs pandas+pyarrow — installing")
    if not yes and sys.stdin.isatty():
        ans = input("pip install pandas pyarrow? [Y/n]: ").strip().lower()
        if ans not in ("", "y", "yes"):
            return False
    rc = run([sys.executable, "-m", "pip", "install", "pandas", "pyarrow",
              "--disable-pip-version-check"], root)
    return rc == 0


def ollama_models(base_url: str) -> list[str] | None:
    """Model names from an Ollama server, or None when unreachable."""
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags",
                                    timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return None


def phase_model(args: argparse.Namespace) -> None:
    log("== phase 2/5: model ==")
    names = ollama_models(args.base_url)
    if names is None:
        log(f"no Ollama at {args.base_url} — using OpenAI-compatible reader")
        log("set --api-key or OPENAI_API_KEY for hosted endpoints")
        return
    log(f"Ollama reachable ({len(names)} models): " + ", ".join(names[:8]))
    if args.model in names:
        log(f"model ready: {args.model}")
        return
    log(f"model {args.model!r} not pulled.")
    if args.pull or (args.yes is False and sys.stdin.isatty()
                     and input(f"ollama pull {args.model}? [Y/n]: "
                               ).strip().lower() in ("", "y", "yes")):
        rc = run(["ollama", "pull", args.model], Path.cwd())
        if rc != 0:
            sys.exit(f"ollama pull {args.model} failed")
    elif not args.pull and args.yes:
        sys.exit(f"model {args.model!r} missing — pull it or pass --pull")


def phase_data(root: Path, suites: list[str]) -> None:
    log("== phase 3/5: datasets ==")
    needed: list[str] = []
    for s in suites:
        needed.extend(SUITE_NEEDS.get(s, []))
    for rel in dict.fromkeys(needed):
        dest = root / rel
        if dest.exists() and dest.stat().st_size > 0:
            log(f"have {rel}")
            continue
        try:
            download(DATASETS[rel], dest)
        except Exception as exc:
            err(f"cannot fetch {rel}: {exc}")
            err(f"place it at {dest} manually and re-run "
                f"(--suites {','.join(suites)})")
            sys.exit(1)


def suite_cmd(args: argparse.Namespace, root: Path, outdir: Path,
              suite: str) -> list[str]:
    py = sys.executable
    base = ["--reader-api-base", args.base_url,
            "--reader-api-key", args.api_key, "--reader-model", args.model]
    if suite == "dims":
        return [py, "-m", "contextmemory.cli", "dims",
                "--system", args.systems.split(",")[0].strip(),
                *base]
    if suite == "bench":
        return [py, "-m", "contextmemory.cli", "bench",
                "--system", "contextmemory"]
    if suite == "longmemeval":
        systems = [s.strip() for s in args.systems.split(",")]
        # run_official handles multi-system lineups on one rig
        cmd = [py, "benchmarks/run_official.py", "longmemeval",
               "--n", str(args.n), "--systems", *systems]
        if args.judge:
            cmd.append("--judge")
        return cmd
    if suite == "locomo":
        systems = [s.strip() for s in args.systems.split(",")]
        return [py, "benchmarks/run_official.py", "locomo",
                "--convos", *[str(c) for c in args.locomo_convos],
                "--systems", *systems]
    if suite == "beam":
        systems = [s.strip() for s in args.systems.split(",")]
        return [py, "benchmarks/run_official.py", "beam",
                "--convos", *[str(c) for c in args.beam_convos],
                "--systems", *systems]
    raise ValueError(suite)


def parse_summary(suite: str, output: str) -> str:
    """Best-effort one-line summary scraped from live output."""
    pats = {
        "dims": r"(\w[\w-]* overall: [\d.]+ \(\d+ probes\))",
        "bench": r"(ingest\s+p50.*|answer\s+p50.*)",
        "longmemeval": r"(\S+: det [\d.]+.*)",
        "locomo": r"(convo \d+ \S+: \S+)",
        "beam": r"(\S+: judge [\d.]+.*)",
    }
    hits = re.findall(pats.get(suite, r"^$"), output, re.M)
    return " | ".join(h.strip() for h in hits[-4:]) or "(see log)"


def parse_dims_rows(output: str) -> list[tuple[str, str, str]]:
    """[(dimension, score, probes)] from `dims` output."""
    return [(d, s, n) for d, s, n in
            re.findall(r"^(\w[\w-]*)\s+overall:\s+([\d.]+)\s+\((\d+) probes\)",
                       output, re.M)]


def parse_bench_rows(output: str) -> list[tuple[str, str, str, str]]:
    """[(kind, p50, p95, mean)] from `bench` output."""
    rows = []
    for kind, p50, p95, mean in re.findall(
            r"^(ingest|answer)\s+p50\s+([\d.]+)\s*ms\s+p95\s+([\d.]+)\s*ms"
            r"\s+mean\s+([\d.]+)\s*ms", output, re.M):
        rows.append((kind, p50, p95, mean))
    return rows


def parse_longmemeval_rows(output: str) -> list[tuple[str, str, str]]:
    """[(system, det, judge-or-dash)] from official longmemeval output."""
    rows = []
    for line in output.splitlines():
        m = re.match(r"^\s{2}(\S+): det ([\d.]+)\s*(judge ([\d.]+))?", line)
        if m:
            rows.append((m.group(1), m.group(2), m.group(4) or "-"))
    return rows


def parse_convo_rows(output: str) -> list[tuple[str, str, str]]:
    """[(convo, system, score)] from locomo/beam `convo i sys: score` lines."""
    rows = []
    for line in output.splitlines():
        m = re.match(r"^\s*(?:convo\s+(\S+)\s+)?(\S+):\s+"
                     r"(judge\s+[\d.]+\s*(?:\(ingest[^)]*\))?"
                     r"|\d+/\d+=[\d.]+\s*(?:\(ingest[^)]*\))?)",
                     line.strip())
        if m and (m.group(1) or "convo" in line):
            convo = m.group(1) or "?"
            rows.append((convo, m.group(2), m.group(3).strip()))
    return rows


def md_table(headers: list[str], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) if len(lines) > 2 else "_no rows parsed (see log)_"


def phase_run(args: argparse.Namespace, root: Path, outdir: Path,
              suites: list[str]) -> dict[str, dict]:
    log("== phase 4/5: running suites serially, one rig, one model ==")
    log(f"model={args.model} base={args.base_url} systems={args.systems}")
    results: dict[str, dict] = {}
    for i, suite in enumerate(suites, 1):
        log(f"-- suite {i}/{len(suites)}: {suite} --")
        t0 = time.perf_counter()
        cmd = suite_cmd(args, root, outdir, suite)
        # Tee live output to both terminal and per-suite log.
        logfile = outdir / f"{suite}.log"
        proc = subprocess.Popen(
            cmd, cwd=str(root), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        text: list[str] = []
        timed_out = False
        with open(logfile, "w", encoding="utf-8") as fh:
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    fh.write(line)
                    text.append(line)
                rc = proc.wait(timeout=args.timeout or None)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = 124
                timed_out = True
                msg = f"\n[cmbench] suite {suite} exceeded --timeout {args.timeout}s\n"
                print(msg, flush=True)
                fh.write(msg)
        dt = time.perf_counter() - t0
        blob = "".join(text)
        results[suite] = {
            "exit": rc, "seconds": round(dt, 1),
            "summary": parse_summary(suite, blob),
            "timed_out": timed_out,
            "log": str(logfile),
        }
        status = "OK" if rc == 0 else ("TIMEOUT" if timed_out else f"FAILED({rc})")
        log(f"suite {suite}: {status} in {dt:.0f}s — {results[suite]['summary']}")
        if rc != 0 and not args.keep_going:
            sys.exit(f"stopping after {suite} failure (see {logfile}); "
                     f"re-run with --keep-going to continue")
    return results


def dataset_provenance(root: Path, suites: list[str]) -> list[tuple[str, str, str]]:
    rows = []
    for rel in dict.fromkeys(n for s in suites for n in SUITE_NEEDS.get(s, [])):
        p = root / rel
        size = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "missing"
        rows.append((rel, size, DATASETS.get(rel, "bundled/synthetic")))
    if not rows:
        rows.append(("(synthetic)", "-", "generated in-process, no download"))
    return rows


def render_markdown(root: Path, outdir: Path, results: dict[str, dict],
                    args: argparse.Namespace, started: str) -> str:
    """Portable Markdown report: the artifact you paste into a PR or issue."""
    import platform as _plat

    ok = all(r["exit"] == 0 for r in results.values())
    lines = [
        "# cmbench report",
        "",
        f"**{'PASS' if ok else 'FAIL'}** · {started} · "
        f"model `{args.model}` · systems `{args.systems}`",
        "",
        "## Rig",
        "",
        md_table(["Field", "Value"], [
            ("date (UTC)", started),
            ("platform", f"{_plat.system()} {_plat.release()} ({_plat.machine()})"),
            ("python", _plat.python_version()),
            ("model", args.model),
            ("reader base URL", args.base_url),
            ("systems", args.systems),
            ("suites", ", ".join(results) or "-"),
            ("repo", REPO),
        ]),
        "",
        "Reproduce:",
        "",
        f"```bash\npython scripts/cmbench.py --model {args.model} "
        f"--base-url {args.base_url} --systems {args.systems} "
        f"--suites {','.join(results) or 'dims,bench'} --n {args.n}\n```",
        "",
        "## Summary",
        "",
        md_table(["Suite", "Status", "Time (s)", "Result"], [
            (s, "ok" if r["exit"] == 0 else f"exit {r['exit']}",
             r["seconds"], r["summary"].replace(" | ", "; "))
            for s, r in results.items()
        ]),
        "",
    ]
    for suite, r in results.items():
        lines += [f"## {suite}", ""]
        logname = Path(r.get("log", "")).name or f"{suite}.log"
        if suite == "dims":
            lines += [md_table(["Dimension", "Score", "Probes"],
                               parse_dims_rows(_read_log(outdir, suite))), ""]
        elif suite == "bench":
            lines += [md_table(["Kind", "p50 (ms)", "p95 (ms)", "mean (ms)"],
                               parse_bench_rows(_read_log(outdir, suite))), ""]
        elif suite == "longmemeval":
            lines += [md_table(["System", "Deterministic", "Judge"],
                               parse_longmemeval_rows(_read_log(outdir, suite))), ""]
        elif suite in ("locomo", "beam"):
            lines += [md_table(["Convo", "System", "Score"],
                               parse_convo_rows(_read_log(outdir, suite))), ""]
        lines += [f"Full log: `{logname}`", ""]
    lines += [
        "## Datasets",
        "",
        md_table(["File", "Size", "Source"],
                 dataset_provenance(root, list(results))),
        "",
        "## Checkpoints",
        "",
        "- Per-suite logs: `<outdir>/<suite>.log`",
        "- Official-run JSONL: `benchmarks/results/`",
        "- This report: `REPORT.md` (here) + `summary.json`",
        "",
        "## Caveats",
        "",
        "- Same rig, same model for every system in this report; do not "
        "compare these numbers with other harnesses, judges, or dates.",
        "- LLM-judged scores use the configured reader as judge unless the "
        "official GPT judge is used; judge substitution is disclosed per run.",
        "- `dims`/`bench` are ContextMemory harnesses for gaps public "
        "benchmarks don't cover (write precision, evolution, forgetting, "
        "deterministic latency).",
        "",
    ]
    return "\n".join(lines)


def _read_log(outdir: Path, suite: str) -> str:
    try:
        return (outdir / f"{suite}.log").read_text(encoding="utf-8")
    except OSError:
        return ""


def phase_report(root: Path, outdir: Path, results: dict,
                 args: argparse.Namespace, started: str) -> None:
    log("== phase 5/5: report ==")
    report = {
        "model": args.model, "base_url": args.base_url,
        "systems": args.systems, "suites": list(results),
        "started_utc": started,
        "results": results,
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=1))
    md = render_markdown(root, outdir, results, args, started)
    (outdir / "REPORT.md").write_text(md, encoding="utf-8")
    print()
    print("=" * 64)
    print(f"cmbench results  model={args.model}  systems={args.systems}")
    print("=" * 64)
    for suite, r in results.items():
        mark = "ok " if r["exit"] == 0 else "FAIL"
        print(f"[{mark}] {suite:<12} {r['seconds']:>7.0f}s  {r['summary']}")
    print(f"\nmarkdown : {outdir / 'REPORT.md'}")
    print(f"json     : {outdir / 'summary.json'}")
    print("checkpoints from official runs: benchmarks/results/")
    print("=" * 64)
    log("== phase 5/5: report ==")
    report = {
        "model": args.model, "base_url": args.base_url,
        "systems": args.systems, "suites": list(results),
        "results": results,
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=1))
    print()
    print("=" * 64)
    print(f"cmbench results  model={args.model}  systems={args.systems}")
    print("=" * 64)
    for suite, r in results.items():
        mark = "ok " if r["exit"] == 0 else "FAIL"
        print(f"[{mark}] {suite:<12} {r['seconds']:>7.0f}s  {r['summary']}")
    print(f"\nlogs + checkpoints: {outdir}")
    print("checkpoints from official runs: benchmarks/results/")
    print("=" * 64)


def maybe_supermemory(root: Path, args: argparse.Namespace) -> None:
    if not args.with_supermemory:
        return
    dest = root / "benchmarks" / "supermemory"
    if not (dest / "README.md").exists():
        log("cloning supermemoryai/supermemory (reference, ~200MB)…")
        rc = run(["git", "clone", "--depth", "1", SUPERMEMORY_REPO, str(dest)],
                 root)
        if rc != 0:
            err("supermemory clone failed — continuing without it")
            return
    else:
        log("have benchmarks/supermemory")
    if os.environ.get("SUPERMEMORY_API_KEY"):
        log("SUPERMEMORY_API_KEY set — wire it into your harness via "
            "benchmarks/run_official.py --systems (adapter TODO per key)")
    else:
        log("no SUPERMEMORY_API_KEY — cloud comparison skipped (honest: "
            "no key, no claim). Local lineup runs regardless.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-command memory benchmarks: same rig, same model.")
    p.add_argument("--model", default="qwen3:4b", help="reader model for ALL systems")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--systems", default="contextmemory,full-history",
                   help="comma list: contextmemory,full-history,recency-2,recency")
    p.add_argument("--suites", default="dims,bench",
                   help="comma list of dims,bench,longmemeval,locomo,beam (or all)")
    p.add_argument("--n", type=int, default=30, help="longmemeval instances")
    p.add_argument("--judge", action="store_true",
                   help="official-style LLM judge for longmemeval")
    p.add_argument("--locomo-convos", nargs="+", type=int, default=[0, 1])
    p.add_argument("--beam-convos", nargs="+", type=int, default=[0, 1])
    p.add_argument("--pull", action="store_true", help="ollama pull the model")
    p.add_argument("--yes", action="store_true", help="non-interactive")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="per-suite timeout in seconds (0 = none)")
    p.add_argument("--check", action="store_true",
                   help="env+model+data checks only, run nothing")
    p.add_argument("--with-supermemory", action="store_true")
    p.add_argument("--out", default="",
                   help="report dir (default reports/runs/cmbench-<ts>)")
    args = p.parse_args(argv)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    if suites == ["all"]:
        suites = list(SUITE_ORDER)
    for s in suites:
        if s not in SUITE_ORDER:
            sys.exit(f"unknown suite {s!r} (pick from {SUITE_ORDER})")
    args.suites = [s for s in SUITE_ORDER if s in suites]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    outdir = Path(args.out) if args.out else root / "reports" / "runs" / f"cmbench-{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    phase_env(root, args.yes)
    phase_model(args)
    maybe_supermemory(root, args)
    if "beam" in args.suites and not ensure_pandas(root, args.yes):
        log("pandas declined — dropping beam suite")
        args.suites = [s for s in args.suites if s != "beam"]
    phase_data(root, args.suites)
    if args.check:
        log("check mode: env + model + data OK, ran nothing")
        return 0
    results = phase_run(args, root, outdir, args.suites)
    phase_report(root, outdir, results, args, started)
    return 0 if all(r["exit"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
