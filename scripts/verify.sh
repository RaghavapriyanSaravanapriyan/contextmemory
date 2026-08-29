#!/usr/bin/env bash

set -euo pipefail

echo "==> ContextMemory verification"

echo "--> Running test suite"
uv run pytest

echo "--> Linting with ruff"
uv run ruff check contextmemory tests

echo "==> Verification complete"