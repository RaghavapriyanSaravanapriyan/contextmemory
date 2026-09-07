# Run Report — One-Command Benchmarks + Markdown Reports (2026-09-08)

## What shipped

`scripts/cmbench.py` (stdlib only): env → model → data → serial runs →
`REPORT.md` + `summary.json` + per-suite logs under
`reports/runs/cmbench-<ts>/`. Same rig, same model, every system.
`--with-supermemory` clones the reference repo and joins the lineup only
with `SUPERMEMORY_API_KEY` (no key = skipped, never faked).

## Proof (this rig, CPU-only)

`python scripts/cmbench.py --suites bench --yes` → PASS, REPORT.md with
bench table (ingest p50 0.042 ms, answer p50 0.176 ms), reproduce command,
dataset provenance, caveats. Full LLM suites need a capable reader model;
`--check` validates env/model/data without running.

## Validation

- 6 new cmbench tests (arg parsing, command shapes, parsers, markdown).
- Full suite: 106 pytest green, ruff clean (incl. scripts/cmbench.py),
  15/15 C++ green.
