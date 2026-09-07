"""Tests for the one-command benchmark runner (offline, model-free)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_cmbench():
    spec = importlib.util.spec_from_file_location(
        "cmbench", ROOT / "scripts" / "cmbench.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cm():
    return load_cmbench()


def test_parse_args_defaults(cm) -> None:
    # Defaults are FULL official sets — no small/synthetic version
    # unless --fast is passed explicitly.
    args = cm.parse_args(["--model", "qwen3:4b"])
    assert args.model == "qwen3:4b"
    assert args.suites == ["longmemeval", "locomo", "beam"]
    assert args.timeout == 0.0
    assert args.n == 500
    assert args.locomo_convos == list(range(10))
    assert args.beam_convos == list(range(20))
    assert args.systems_list == ["contextmemory", "full-history"]


def test_parse_args_fast_preset(cm) -> None:
    args = cm.parse_args(["--fast"])
    assert args.n == 50
    assert args.locomo_convos == [0]
    assert args.beam_convos == [0]


def test_parse_args_all_and_rejects_unknown(cm) -> None:
    args = cm.parse_args(["--suites", "all"])
    assert args.suites == ["dims", "bench", "longmemeval", "locomo", "beam"]
    with pytest.raises(SystemExit):
        cm.parse_args(["--suites", "nope"])


def test_order_systems(cm) -> None:
    assert cm.order_systems(["full-history", "supermemory", "contextmemory"]) == [
        "contextmemory", "supermemory", "full-history"]


def test_split_runnable_suites(cm) -> None:
    kept, dropped = cm.split_runnable_suites(
        ["dims", "bench", "longmemeval"], True)
    assert (kept, dropped) == (["dims", "bench", "longmemeval"], [])
    kept, dropped = cm.split_runnable_suites(
        ["dims", "bench", "locomo"], False)
    assert kept == ["bench"] and dropped == ["dims", "locomo"]
    kept, dropped = cm.split_runnable_suites(["dims"], False)
    assert kept == [] and dropped == ["dims"]  # caller exits 2 with the fix


def test_preflight_reader_fails_cleanly() -> None:
    from contextmemory.cli import _preflight_reader

    class _Dead:
        def complete(self, messages, temperature=0.0):
            raise ConnectionError("refused")

    import pytest as _pt
    with _pt.raises(SystemExit) as ei:
        _preflight_reader(_Dead(), "qwen3:4b", "http://localhost:11434")
    msg = str(ei.value.code)
    assert "ollama serve" in msg and "ollama pull qwen3:4b" in msg


def test_preflight_reader_passes(fake_reader) -> None:
    from contextmemory.cli import _preflight_reader

    assert _preflight_reader(fake_reader, "m", "u") is None


def test_suite_cmd_shapes(cm, tmp_path) -> None:
    args = cm.parse_args(["--model", "m", "--systems",
                          "contextmemory,full-history", "--n", "5"])
    outdir = tmp_path
    assert "dims" in cm.suite_cmd(args, ROOT, outdir, "dims")
    bench = cm.suite_cmd(args, ROOT, outdir, "bench", "contextmemory")
    assert bench[-1] == "contextmemory"
    lme = cm.suite_cmd(args, ROOT, outdir, "longmemeval")
    assert "--n" in lme and "--systems" in lme
    dims_cm = cm.suite_cmd(args, ROOT, outdir, "dims", "contextmemory")
    dims_fh = cm.suite_cmd(args, ROOT, outdir, "dims", "full-history")
    assert dims_cm != dims_fh  # per-system runs, no shared numbers
    args_j = cm.parse_args(["--judge"])
    assert "--judge" in cm.suite_cmd(args_j, ROOT, outdir, "longmemeval")
    assert "locomo" in cm.suite_cmd(args, ROOT, outdir, "locomo")
    assert "beam" in cm.suite_cmd(args, ROOT, outdir, "beam")
    with pytest.raises(ValueError):
        cm.suite_cmd(args, ROOT, outdir, "nope")


def test_parsers(cm) -> None:
    dims_out = ("system:        contextmemory\n"
                "evolution overall: 0.8000 (5 probes)\n"
                "forgetting overall: 1.0000 (3 probes)\n")
    assert cm.parse_dims_rows(dims_out) == [
        ("evolution", "0.8000", "5"), ("forgetting", "1.0000", "3")]
    bench_out = ("ingest  p50    0.042 ms  p95    0.066 ms  mean    0.049 ms\n"
                 "answer  p50    0.168 ms  p95    0.253 ms  mean    0.168 ms\n")
    assert cm.parse_bench_rows(bench_out) == [
        ("ingest", "0.042", "0.066", "0.049"),
        ("answer", "0.168", "0.253", "0.168")]
    lme_out = "  contextmemory: det 0.500 judge 0.60\n"
    assert cm.parse_longmemeval_rows(lme_out) == [
        ("contextmemory", "0.500", "0.60")]
    loc_out = "  convo 0 contextmemory: 3/5=0.600 (ingest 12s)\n"
    rows = cm.parse_convo_rows(loc_out)
    assert rows and rows[0][0] == "0" and "0.600" in rows[0][2]
    assert cm.md_table(["A"], []) == "_no rows parsed (see log)_"


def test_render_markdown(tmp_path, cm) -> None:
    args = cm.parse_args(["--model", "m", "--suites", "bench"])
    outdir = tmp_path / "rep"
    outdir.mkdir()
    (outdir / "bench-contextmemory.log").write_text(
        "ingest  p50    0.042 ms  p95    0.066 ms  mean    0.049 ms\n",
        encoding="utf-8")
    results = {"bench:contextmemory": {
        "exit": 0, "seconds": 3.0, "summary": "ingest p50 0.042",
        "timed_out": False, "log": str(outdir / "bench-contextmemory.log")}}
    md = cm.render_markdown(ROOT, outdir, results, args, "2026-09-08 00:00 UTC",
                            [("supermemory", "skipped", "no key")],
                            "deterministic-only (no LLM judge)")
    assert "# cmbench report" in md
    assert "0.042" in md
    assert "## Caveats" not in md  # renamed section
    assert "## Bias controls (how this stays honest)" in md
    assert "no key" in md and "deterministic-only" in md
    assert "Reproduce:" in md
