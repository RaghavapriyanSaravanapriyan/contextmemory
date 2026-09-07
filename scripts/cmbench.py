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
  3. DATA   — fetches missing OFFICIAL datasets (LongMemEval oracle/S,
             LoCoMo, BEAM parquet) with MB progress bars. URLs are pinned
             to upstream repos (see DATASETS) and verified.
  4. RUN    — runs suites SERIALLY in fixed order (contextmemory first,
             then supermemory, then baselines): dims + bench run EVERY
             system separately; official suites share one replay + judge.
             stdout streamed live.
  5. REPORT — checkpoints JSONL under reports/runs/cmbench-<ts>/ plus
             REPORT.md (metrics tables, rig, sources, readiness, bias
             controls) and summary.json.

Bias controls: same model/reader/judge for all, explicit system
resolution (unknown names exit loudly, never silent fallbacks),
third parties fail closed (skipped with reason, never zeros),
--fast (≈10% subsets, default) vs --full.

Optional head-to-heads (no key = cleanly skipped, never faked):
  --with-supermemory  clone supermemoryai/supermemory for reference,
                      install the official SDK, and — when
                      SUPERMEMORY_API_KEY is set — add it to the lineup
                      through benchmarks/adapters (same reader, same
                      prompt shape, async ingest polled to done).

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

# Every dataset comes from its OFFICIAL source. Verified 2026-09-08:
# - LongMemEval files: official README download block (huggingface.co /
#   xiaowu0162/longmemeval-cleaned). Superseded only by longmemeval_m (not
#   used: needs 500-session histories per question).
# - LoCoMo: official repo README points at ./data/locomo10.json
#   (github.com/snap-research/locomo); downloaded 2.8MB, byte-identical path.
# - BEAM 100K: HEAD-checked 200 on huggingface.co/Mohammadta/BEAM
#   (5.4MB, CC-BY-SA-4.0). Splits used per --bucket (default 100K).
SUITE_ORDER = ["dims", "bench", "longmemeval", "locomo", "beam"]

# CLI-validated system names for the built-in dims/bench suites.
LOCAL_SYSTEMS = {"full-history", "recency-2", "recency-10", "contextmemory",
                 "supermemory"}
THIRD_PARTY = {"supermemory"}


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


def phase_env(root: Path, args: argparse.Namespace) -> None:
    log("== phase 1/5: environment ==")
    if sys.version_info < (3, 11):  # noqa: UP036 - runtime guard for old Pythons
        sys.exit("Python >= 3.11 required, found " + sys.version.split()[0])
    try:
        import contextmemory  # noqa: F401
        log("contextmemory already installed — skipping reinstall")
        return
    except ImportError:
        pass
    if args.no_install:
        sys.exit("contextmemory is not installed and --no-install was passed. "
                 "Run without --no-install once, or pip install -e . manually.")
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
    # Re-verify in a FRESH interpreter: the running process never picks up
    # newly written .pth entries, so an in-process import would lie.
    probe = subprocess.run(
        [sys.executable, "-c", "import contextmemory"],
        cwd=str(root), capture_output=True)
    if probe.returncode != 0:
        sys.exit("install reported success but a fresh python still cannot "
                 "import contextmemory — check which python/pip pair ran.")


def ensure_pandas(root: Path, args: argparse.Namespace) -> bool:
    try:
        import pandas  # noqa: F401
        log("pandas already installed — skipping")
        return True
    except ImportError:
        pass
    if args.no_install:
        log("pandas missing and --no-install passed — dropping beam suite")
        return False
    log("BEAM suite needs pandas+pyarrow — installing")
    if not args.yes and sys.stdin.isatty():
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


def ensure_ollama_up(root: Path, base_url: str,
                     timeout_s: float = 60.0) -> bool:
    """One-shot Ollama: start `ollama serve` detached if the port is dead.

    Returns True when /api/tags answers. The server is intentionally left
    running (it is a service; PID + stop command are logged). Only acts on
    the DEFAULT local URL — hosted bases are never auto-started.
    """
    if ollama_models(base_url) is not None:
        return True
    if base_url.rstrip("/") != "http://localhost:11434":
        return False
    if shutil.which("ollama") is None:
        err("no `ollama` binary found — install it from ollama.com, then re-run")
        return False
    log("Ollama is down — starting `ollama serve` in the background…")
    try:
        if sys.platform == "win32":
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP,
                cwd=str(root))
        else:
            proc = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, cwd=str(root))
    except OSError as exc:
        err(f"could not launch ollama: {exc}")
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ollama_models(base_url) is not None:
            log(f"`ollama serve` is up (pid {proc.pid}; stop later with "
                f"`kill {proc.pid}`)")
            return True
        time.sleep(2.0)
    err(f"`ollama serve` did not answer within {timeout_s:.0f}s")
    return False


def ensure_model_pulled(root: Path, args: argparse.Namespace,
                        names: list[str]) -> bool:
    """One-shot model fetch: `ollama pull` when the model is missing.

    The user asked for one-shot: a missing model downloads automatically.
    `--no-pull` opts out (metered connections); then we exit with the fix.
    """
    if args.model in names:
        log(f"model ready: {args.model}")
        return True
    if args.no_pull:
        sys.exit(f"model {args.model!r} not pulled and --no-pull was passed. "
                 f"Run: ollama pull {args.model}")
    if shutil.which("ollama") is None:
        sys.exit(f"model {args.model!r} missing and no `ollama` binary. "
                 f"Install from ollama.com, then: ollama pull {args.model}")
    log(f"model {args.model!r} missing — downloading "
        f"(`ollama pull {args.model}`, one-time, may take a while)…")
    t0 = time.monotonic()
    rc = run(["ollama", "pull", args.model], root)
    dt = time.monotonic() - t0
    if rc != 0:
        sys.exit(f"ollama pull {args.model} failed after {dt:.0f}s")
    log(f"model {args.model} downloaded in {dt:.0f}s")
    return True


LLM_SUITES = ("dims", "longmemeval", "locomo", "beam")


def split_runnable_suites(suites: list[str],
                          reader_ok: bool) -> tuple[list[str], list[str]]:
    """Keep model-free suites when no reader is reachable.

    `bench` uses a null reader (pure latency, no model). Everything else
    needs a live model. Dropping — not crashing — is the Apple behavior:
    the user still gets signal, plus the exact fix.
    """
    if reader_ok:
        return list(suites), []
    kept = [s for s in suites if s not in LLM_SUITES]
    dropped = [s for s in suites if s in LLM_SUITES]
    return kept, dropped


READER_FIX = (
    "no reader model reachable.\n"
    "  fix local:  ollama serve  # new terminal, default port 11434\n"
    "              ollama pull {model}  # then re-run\n"
    "  fix hosted: pass --base-url + --api-key (or set OPENAI_API_KEY)\n"
    "  model-free: --suites bench  # deterministic latency, no model needed"
)


def phase_model(root: Path, args: argparse.Namespace) -> bool:
    log("== phase 2/5: model ==")
    names = ollama_models(args.base_url)
    if names is None:
        if args.api_key and args.api_key != "EMPTY":
            log("using OpenAI-compatible reader "
                f"(key set, base {args.base_url})")
            return True
        # One-shot: bring Ollama up ourselves instead of giving up.
        if not ensure_ollama_up(root, args.base_url):
            log("LLM suites will be skipped; model-free suites still run.\n"
                + READER_FIX.format(model=args.model))
            return False
        names = ollama_models(args.base_url) or []
    log(f"Ollama reachable ({len(names)} models): " + ", ".join(names[:8]))
    # One-shot: a missing model downloads automatically (--no-pull opts out).
    if args.model not in names:
        return ensure_model_pulled(root, args, names)
    log(f"model ready: {args.model}")
    return True


def phase_data(root: Path, suites: list[str]) -> None:
    log("== phase 3/5: datasets ==")
    needed: list[str] = []
    for s in suites:
        needed.extend(SUITE_NEEDS.get(s, []))
    needed = list(dict.fromkeys(needed))
    if not needed:
        log("no downloads needed (synthetic suites run in-process)")
        return
    for rel in needed:
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
              suite: str, system: str = "") -> list[str]:
    py = sys.executable
    base = ["--reader-api-base", args.base_url,
            "--reader-api-key", args.api_key, "--reader-model", args.model]
    if suite == "dims":
        # Zero-bias rule: EVERY system runs the same scenarios, same reader.
        return [py, "-m", "contextmemory.cli", "dims", "--system", system,
                *base]
    if suite == "bench":
        # Deterministic latency, per system (null reader, no model).
        return [py, "-m", "contextmemory.cli", "bench", "--system", system]
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


def sys_metrics() -> list[tuple[str, str]]:
    """Machine context for the report. Stdlib only, degrades to n/a."""
    import platform as _plat

    cpu = str(os.cpu_count() or "n/a")
    ram = "n/a"
    try:
        if sys.platform == "linux":
            with open("/proc/meminfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        ram = f"{int(line.split()[1]) / 1e6:.1f} GB"
                        break
        elif sys.platform == "win32":
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MS()
            st.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            ram = f"{st.ullTotalPhys / 1e9:.1f} GB"
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                 capture_output=True, text=True)
            ram = f"{int(out.stdout.strip()) / 1e9:.1f} GB"
    except Exception:
        ram = "n/a"
    return [("os", f"{_plat.system()} {_plat.release()} ({_plat.machine()})"),
            ("cpu cores", cpu), ("ram", ram)]


def md_table(headers: list[str], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines) if len(lines) > 2 else "_no rows parsed (see log)_"


def order_systems(systems: list[str]) -> list[str]:
    """Run order, fixed for comparability: contextmemory first (full detail),
    then supermemory, then baselines. Order never affects scores — every
    system gets a fresh container and identical inputs."""
    first = [s for s in systems if s == "contextmemory"]
    second = [s for s in systems if s == "supermemory"]
    rest = [s for s in systems if s not in ("contextmemory", "supermemory")]
    return first + second + rest


def phase_run(args: argparse.Namespace, root: Path, outdir: Path,
              suites: list[str], systems: list[str]) -> dict[str, dict]:
    log("== phase 4/5: running suites serially, one rig, one model ==")
    log(f"model={args.model} base={args.base_url} order={order_systems(systems)}")
    results: dict[str, dict] = {}
    for i, suite in enumerate(suites, 1):
        if suite in ("dims", "bench"):
            # Per-system runs: the bias fix. Each contender faces the same
            # scenarios/workload; tables stay side-by-side in REPORT.md.
            for system in order_systems(systems):
                key = f"{suite}:{system}"
                log(f"-- suite {i}/{len(suites)}: {suite} [{system}] --")
                results[key] = _run_one(args, root, outdir, suite, system, key)
                if results[key]["exit"] != 0 and not args.keep_going:
                    sys.exit(_fail_msg(suite, system, outdir))
        else:
            # Official suites run the whole lineup in one process (shared
            # replay, shared judge) via benchmarks/run_official.py.
            log(f"-- suite {i}/{len(suites)}: {suite} "
                f"[{','.join(order_systems(systems))}] --")
            results[suite] = _run_one(args, root, outdir, suite, "", suite)
            if results[suite]["exit"] != 0 and not args.keep_going:
                sys.exit(_fail_msg(suite, "lineup", outdir))
    return results


def _fail_msg(suite: str, who: str, outdir: Path) -> str:
    return (f"stopping after {suite} [{who}] failure "
            f"(see {outdir / (suite + '.log')}); re-run with --keep-going "
            f"to continue")


def _run_one(args: argparse.Namespace, root: Path, outdir: Path,
             suite: str, system: str, key: str) -> dict:
    t0 = time.perf_counter()
    cmd = suite_cmd(args, root, outdir, suite, system)
    logfile = outdir / f"{key.replace(':', '-')}.log"
    # Tee live output to both terminal and per-run log.
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
            msg = f"\n[cmbench] {key} exceeded --timeout {args.timeout}s\n"
            print(msg, flush=True)
            fh.write(msg)
    dt = time.perf_counter() - t0
    blob = "".join(text)
    status = "OK" if rc == 0 else ("TIMEOUT" if timed_out else f"FAILED({rc})")
    summary = parse_summary(suite, blob)
    log(f"run {key}: {status} in {dt:.0f}s — {summary}")
    return {
        "exit": rc, "seconds": round(dt, 1), "summary": summary,
        "timed_out": timed_out, "log": str(logfile),
    }


def dataset_provenance(
    root: Path, suites: list[str]
) -> list[tuple[str, str, str, str]]:
    rows = []
    for rel in dict.fromkeys(n for s in suites for n in SUITE_NEEDS.get(s, [])):
        p = root / rel
        if not p.exists() or p.stat().st_size == 0:
            rows.append((rel, "missing", "-", DATASETS.get(rel, "")))
            continue
        size = f"{p.stat().st_size / 1e6:.1f} MB"
        # Row counts for JSON datasets (cheap except the 277MB S file —
        # count it by top-level split instead of full parse).
        count = "-"
        if p.suffix == ".json" and p.stat().st_size < 100_000_000:
            try:
                count = str(len(json.loads(p.read_text(encoding="utf-8"))))
            except (ValueError, OSError, MemoryError):
                count = "unreadable"
        rows.append((rel, size, count, DATASETS.get(rel, "bundled/synthetic")))
    if not rows:
        rows.append(("(synthetic)", "-", "-", "generated in-process, no download"))
    return rows


def render_markdown(root: Path, outdir: Path, results: dict[str, dict],
                    args: argparse.Namespace, started: str,
                    readiness: list[tuple[str, str, str]],
                    judge_note: str,
                    dropped: list[str] | None = None) -> str:
    """Portable Markdown report: the artifact you paste into a PR or issue."""
    import platform as _plat

    ok = all(r["exit"] == 0 for r in results.values())
    order = ", ".join(order_systems(
        [s.strip() for s in args.systems.split(",")]))
    skip_note = (" · skipped: " + ",".join(dropped)) if dropped else ""
    # Contender banner: who ACTUALLY ran vs who was skipped. A skipped
    # contender must be visible at a glance — never buried in a table.
    ran = sorted({k.split(":")[1] if ":" in k else "lineup"
                  for k in results})
    skipped_tp = [s for s, st, _ in readiness if st != "in lineup"]
    banner = (f"RAN: {', '.join(ran) or 'nothing'}"
              + (f" · SKIPPED: {', '.join(skipped_tp)}" if skipped_tp else "")
              + (" · (all requested contenders ran)" if not skipped_tp else
                 " · (skipped contenders scored NOTHING — see readiness)"))
    lines = [
        "# cmbench report",
        "",
        f"**{'PASS' if ok else 'FAIL'}** · {started} · "
        f"model `{args.model}` · run order `{order}`{skip_note}",
        "",
        f"> {banner}",
        "",
        "## Rig",
        "",
        md_table(["Field", "Value"], [
            ("date (UTC)", started),
            *sys_metrics(),
            ("python", _plat.python_version()),
            ("model (ALL systems)", args.model),
            ("reader base URL", args.base_url),
            ("run order", order),
            ("judge", judge_note),
            ("repo", REPO),
        ]),
        "",
        "Reproduce:",
        "",
        f"```bash\npython scripts/cmbench.py --model {args.model} "
        f"--base-url {args.base_url} --systems {args.systems} "
        f"--suites {','.join(suites_in(results)) or 'dims,bench'} "
        f"--n {args.n}" + (" --judge" if args.judge else "") + "\n```",
        "",
        "## Summary",
        "",
        md_table(["Run", "Status", "Time (s)", "Result"], [
            (k, "ok" if r["exit"] == 0 else f"exit {r['exit']}",
             r["seconds"], r["summary"].replace(" | ", "; "))
            for k, r in results.items()
        ]),
        "",
    ]
    # Per-suite detail: per-system keys (dims:X) get System columns.
    for key, r in results.items():
        suite = key.split(":")[0]
        system = key.split(":")[1] if ":" in key else ""
        title = f"## {suite}" + (f" — {system}" if system else " (lineup)")
        lines += [title, ""]
        logname = Path(r.get("log", "")).name or f"{key}.log"
        blob = _read_log_file(outdir, key)
        if suite == "dims":
            rows = [(system or "?", d, s, n)
                    for d, s, n in parse_dims_rows(blob)]
            lines += [md_table(["System", "Dimension", "Score", "Probes"],
                               rows), ""]
        elif suite == "bench":
            rows = [(system or "?", k, p50, p95, mean)
                    for k, p50, p95, mean in parse_bench_rows(blob)]
            lines += [md_table(["System", "Kind", "p50 (ms)", "p95 (ms)",
                                "mean (ms)"], rows), ""]
        elif suite == "longmemeval":
            lines += [md_table(["System", "Deterministic", "Judge"],
                               parse_longmemeval_rows(blob)), ""]
        elif suite in ("locomo", "beam"):
            lines += [md_table(["Convo", "System", "Score"],
                               parse_convo_rows(blob)), ""]
        lines += [f"Full log: `{logname}`", ""]
    lines += [
        "## Datasets (official sources only)",
        "",
        md_table(["File", "Size", "Items", "Official source"],
                 dataset_provenance(root, suites_in(results))),
        "",
        "## Third-party readiness",
        "",
        md_table(["System", "Status", "Detail"], readiness),
        "",
        "## Bias controls (how this stays honest)",
        "",
        "- One rig, one reader model, serial execution — every system sees "
        "identical sessions in identical order.",
        "- Official suites share one replay runner and one judge; "
        f"judge: `{judge_note}`.",
        "- Third parties use the same reader and the same prompt shape as "
        "the local engine (only retrieval/storage is theirs).",
        "- Skipped contenders are reported as skipped with the reason — "
        "never as zero scores.",
        "- `dims`/`bench` are ContextMemory harnesses for gaps public "
        "benchmarks don't cover; they run for EVERY lineup system, and the "
        "tables above show all of them side-by-side.",
        "- Do not compare these numbers with other harnesses, judges, "
        "or dates.",
        "",
        "## Checkpoints",
        "",
        "- Per-run logs: `<outdir>/<suite>[-<system>].log`",
        "- Official-run JSONL: `benchmarks/results/`",
        "- This report: `REPORT.md` (here) + `summary.json`",
        "",
    ]
    return "\n".join(lines)


def suites_in(results: dict[str, dict]) -> list[str]:
    return list(dict.fromkeys(k.split(":")[0] for k in results))


def _read_log_file(outdir: Path, key: str) -> str:
    try:
        return (outdir / f"{key.replace(':', '-')}.log").read_text(
            encoding="utf-8")
    except OSError:
        return ""


def _read_log(outdir: Path, suite: str) -> str:
    return _read_log_file(outdir, suite)


def phase_report(root: Path, outdir: Path, results: dict,
                 args: argparse.Namespace, started: str,
                 readiness: list[tuple[str, str, str]],
                 dropped: list[str] | None = None) -> None:
    log("== phase 5/5: report ==")
    dropped = dropped or []
    judge_note = (f"{args.model} (reader-as-judge)" if args.judge
                  else "deterministic-only (no LLM judge)")
    report = {
        "model": args.model, "base_url": args.base_url,
        "systems": args.systems, "suites": list(results),
        "skipped_suites": dropped,
        "started_utc": started, "judge": judge_note,
        "third_party": [{"system": s, "status": st, "detail": d}
                        for s, st, d in readiness],
        "results": results,
    }
    (outdir / "summary.json").write_text(json.dumps(report, indent=1))
    md = render_markdown(root, outdir, results, args, started, readiness,
                         judge_note, dropped)
    (outdir / "REPORT.md").write_text(md, encoding="utf-8")
    print()
    print("=" * 64)
    print(f"cmbench results  model={args.model}  systems={args.systems}")
    print("=" * 64)
    for key, r in results.items():
        mark = "ok " if r["exit"] == 0 else "FAIL"
        print(f"[{mark}] {key:<22} {r['seconds']:>7.0f}s  {r['summary']}")
    print(f"\nmarkdown : {outdir / 'REPORT.md'}")
    print(f"json     : {outdir / 'summary.json'}")
    print("checkpoints from official runs: benchmarks/results/")
    print("=" * 64)


def ensure_supermemory_sdk(root: Path, args: argparse.Namespace) -> None:
    """Best-effort `pip install supermemory` (explicit opt-in = consent)."""
    try:
        import supermemory  # noqa: F401
        log("supermemory SDK already installed — skipping")
        return
    except ImportError:
        pass
    if args.no_install:
        log("supermemory SDK missing and --no-install passed — skipping")
        return
    log("installing supermemory SDK (third-party contender)")
    cmd = (["uv", "pip", "install", "supermemory"] if shutil.which("uv")
           else [sys.executable, "-m", "pip", "install", "supermemory",
                 "--disable-pip-version-check"])
    if run(cmd, root) != 0:
        err("supermemory SDK install failed — contender will be skipped")


def check_supermemory_ready() -> tuple[bool, str]:
    """Probe via the repo's own adapter (no side effects, no spend)."""
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "benchmarks"))
    try:
        from adapters import probe_supermemory
    except ImportError as exc:
        return False, f"adapter missing ({exc})"
    ready, detail = probe_supermemory()
    if not ready and "no key" in detail:
        # Their documented self-host port: if something listens there,
        # say so (the binary prints its API key on first boot — paste it
        # into SUPERMEMORY_API_KEY). Port-open is a hint, not a claim.
        import socket as _sock

        try:
            with _sock.create_connection(("127.0.0.1", 6767), timeout=1.0):
                detail += ("; local :6767 is OPEN (their self-host default)"
                           " — use the API key that binary printed")
        except OSError:
            pass
    return ready, detail


def maybe_supermemory(root: Path,
                      args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Download supermemory reference + assess lineup readiness.

    Returns readiness rows for the report. Never fakes: not-ready means
    the run proceeds WITHOUT the contender and says exactly why.
    """
    readiness: list[tuple[str, str, str]] = []
    dest = root / "benchmarks" / "supermemory"
    if (dest / "README.md").exists():
        log("have benchmarks/supermemory (reference clone)")
    elif args.with_supermemory:
        log("cloning supermemoryai/supermemory (reference, ~200MB)…")
        if run(["git", "clone", "--depth", "1", SUPERMEMORY_REPO, str(dest)],
               root) != 0:
            err("supermemory clone failed — reference unavailable")
    else:
        log("supermemory reference not cloned "
            "(pass --with-supermemory to fetch it)")

    if args.with_supermemory:
        ensure_supermemory_sdk(root, args)
    ready, detail = check_supermemory_ready()
    if ready:
        readiness.append(("supermemory", "in lineup", detail))
        if "supermemory" not in args.systems_list:
            args.systems_list.append("supermemory")
            args.systems_list = order_systems(args.systems_list)
            args.systems = ",".join(args.systems_list)
            log("supermemory ready — added to the lineup")
    else:
        readiness.append(("supermemory", "skipped", detail))
        if "supermemory" in args.systems_list:
            log(f"supermemory requested but not ready ({detail}) — "
                f"it will fail closed with instructions, not score zeros")
    return readiness


def cmd_doctor(root: Path, args: argparse.Namespace) -> int:
    """Diagnose the machine for a one-shot run. Read-only, no installs."""
    print("# cmbench doctor")
    print(f"- python: {sys.version.split()[0]} ({sys.executable})")
    for tool in ("uv", "git", "cmake", "ninja", "ollama"):
        print(f"- {tool}: {shutil.which(tool) or 'MISSING'}")
    cxx = shutil.which("cl") or shutil.which("g++") or shutil.which("clang++")
    print(f"- c++ compiler: {cxx or 'MISSING (needed to build the core)'}")
    try:
        total, used, free = shutil.disk_usage(str(root))
        print(f"- disk free: {free / 1e9:.1f} GB of {total / 1e9:.1f} GB")
    except OSError:
        print("- disk free: n/a")
    names = ollama_models(args.base_url)
    if names is None:
        print(f"- ollama {args.base_url}: DOWN "
              f"(cmbench will auto-start `ollama serve`)")
    else:
        have = "yes" if args.model in names else "NO (will auto-pull)"
        print(f"- ollama {args.base_url}: UP ({len(names)} models); "
              f"{args.model}: {have}")
    for rel in dict.fromkeys(n for s in SUITE_ORDER for n in SUITE_NEEDS[s]):
        p = root / rel
        ok = p.exists() and p.stat().st_size > 0
        print(f"- dataset {rel}: {'have' if ok else 'missing (will download)'}")
    for key in ("OPENAI_API_KEY", "SUPERMEMORY_API_KEY"):
        print(f"- {key}: {'set' if os.environ.get(key) else 'not set'}")
    try:
        import supermemory  # noqa: F401
        print("- supermemory SDK: installed")
    except ImportError:
        print("- supermemory SDK: missing (needs --with-supermemory)")
    try:
        import pandas  # noqa: F401
        print("- pandas: installed (BEAM ready)")
    except ImportError:
        print("- pandas: missing (auto-installs when beam runs)")
    try:
        import contextmemory  # noqa: F401
        print("- contextmemory: installed")
    except ImportError:
        print("- contextmemory: missing (will install)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-command memory benchmarks: same rig, same model.")
    p.add_argument("--model", default="qwen3:4b", help="reader model for ALL systems")
    p.add_argument("--base-url", default="http://localhost:11434")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--systems", default="contextmemory,full-history",
                   help="comma list: contextmemory,supermemory,full-history,"
                   "recency-2,recency-10,recency")
    p.add_argument("--suites", default="longmemeval,locomo,beam",
                   help="comma list of dims,bench,longmemeval,locomo,beam (or all)."
                   " Default: the three official benchmarks (no synthetics).")
    p.add_argument("--fast", action="store_true",
                   help="~10% smoke subsets (50 LME / 1 LoCoMo / 1 BEAM convo)")
    p.add_argument("--full", action="store_true",
                   help="full official sets (default when --fast is absent)")
    p.add_argument("--n", type=int, default=0,
                   help="longmemeval instances (0 = preset default)")
    p.add_argument("--judge", action="store_true",
                   help="official-style LLM judge for longmemeval")
    p.add_argument("--locomo-convos", nargs="+", type=int, default=[0, 1])
    p.add_argument("--beam-convos", nargs="+", type=int, default=[0, 1])
    p.add_argument("--pull", action="store_true",
                   help="kept for compatibility: pulls are now automatic")
    p.add_argument("--no-pull", action="store_true",
                   help="never auto-pull a missing model (metered links)")
    p.add_argument("--yes", action="store_true", help="non-interactive")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="per-suite timeout in seconds (0 = none)")
    p.add_argument("--no-install", action="store_true",
                   help="never pip-install anything; fail fast if missing")
    p.add_argument("--check", action="store_true",
                   help="env+model+data checks only, run nothing")
    p.add_argument("--doctor", action="store_true",
                   help="diagnose this machine (read-only) and exit")
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
    # Subset presets: FULL official sets by default; --fast selects the
    # ~10% smoke subsets (50 LME questions, 1 LoCoMo convo, 1 BEAM convo).
    # dims/bench are synthetic and always run whole (they take seconds).
    fast = args.fast and not args.full
    if fast:
        if args.n <= 0:
            args.n = 50
        if args.locomo_convos == [0, 1]:
            args.locomo_convos = [0]
        if args.beam_convos == [0, 1]:
            args.beam_convos = [0]
    else:
        if args.n <= 0:
            args.n = 500  # full LongMemEval question set
        if args.locomo_convos == [0, 1]:
            args.locomo_convos = list(range(10))  # all 10 conversations
        if args.beam_convos == [0, 1]:
            # Full 100K split (20 convos); guarded per-convo downstream.
            args.beam_convos = list(range(20))
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if not systems:
        sys.exit("--systems is empty")
    args.systems_list = order_systems(systems)
    args.systems = ",".join(args.systems_list)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.doctor:
        return cmd_doctor(root, args)
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    started = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    outdir = Path(args.out) if args.out else root / "reports" / "runs" / f"cmbench-{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    phase_env(root, args.yes)
    reader_ok = phase_model(root, args)
    readiness = maybe_supermemory(root, args)
    # Fail fast on typos for the built-in suites: an unknown name must
    # never silently become somebody else's numbers.
    if any(s in args.suites for s in ("dims", "bench")):
        bad = [s for s in args.systems_list if s not in LOCAL_SYSTEMS]
        if bad:
            sys.exit(f"unknown system(s) {bad} for dims/bench "
                     f"(known: {sorted(LOCAL_SYSTEMS)})")
    # No reader, no LLM suites: run what is runnable (bench), or exit
    # with the fix instead of a 19s traceback.
    args.suites, dropped = split_runnable_suites(args.suites, reader_ok)
    if dropped:
        log(f"skipped (no reader): {','.join(dropped)}")
    if not args.suites:
        err("nothing runnable: every requested suite needs a reader model.")
        err(READER_FIX.format(model=args.model))
        return 2
    if "beam" in args.suites and not ensure_pandas(root, args):
        log("pandas declined — dropping beam suite")
        args.suites = [s for s in args.suites if s != "beam"]
    phase_data(root, args.suites)
    if args.check:
        log("check mode: env + model + data + readiness OK, ran nothing")
        for name, status, detail in readiness:
            log(f"third-party {name}: {status} ({detail})")
        return 0
    results = phase_run(args, root, outdir, args.suites, args.systems_list)
    phase_report(root, outdir, results, args, started, readiness, dropped)
    return 0 if all(r["exit"] == 0 for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
