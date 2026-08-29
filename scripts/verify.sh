#!/usr/bin/env bash

set -euo pipefail

echo "==> ContextMemory verification"

echo "--> Running test suite"
uv run pytest

echo "--> Linting with ruff"
uv run ruff check src tests

echo "==> Verification complete"