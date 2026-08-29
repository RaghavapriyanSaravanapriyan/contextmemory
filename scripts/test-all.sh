#!/usr/bin/env bash
# ContextMemory v1 — full deterministic test suite, one command.
#
#   ./scripts/test-all.sh
#
# Runs everything that needs no model and no API key:
#   1. C++ ETMC core: build + 11 dependency-free test suites
#   2. Python tests + lint (scripts/verify.sh)
#   3. Deterministic latency bench (microsecond read path, no reader)
#
# Model-based testing (dims / eval) is a separate, slower step — see README.

set -euo pipefail

cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$1"; }

step "1/3 C++ core — build + tests"
cmake -S core -B build/core -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build/core -j >/dev/null
./build/core/cmcore_test

step "2/3 Python — test suite + lint"
uv run pytest
uv run ruff check contextmemory tests

step "3/3 Latency bench — deterministic read path (no model)"
uv run contextmemory bench --system contextmemory --sessions 200

step "v1 test suite complete"