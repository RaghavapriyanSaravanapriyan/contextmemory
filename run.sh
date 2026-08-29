#!/usr/bin/env bash
# ContextMemory one-shot: install everything, ensure Ollama, launch the TUI.
#
#   ./run.sh                  # offline scripted demo (no model needed)
#   ./run.sh --live           # connect to / launch Ollama, pick a model in the TUI
#   ./run.sh --live --model qwen3:8b --url http://localhost:11434
#
# What it does:
#   1. installs `uv` if missing
#   2. ensures the `ollama` binary exists (installs it on Linux/macOS if not)
#   3. installs Python deps and builds the C++ core
#   4. launches the TUI; in --live mode it auto-starts `ollama serve` under
#      the hood when nothing is listening, then lets you pick a model
#      (press O to reconnect / change models at any time).

set -euo pipefail

cd "$(dirname "$0")"

step() { printf '\n==> %s\n' "$1"; }

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
  step "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. ollama binary (needed for live mode + managed launch)
if ! command -v ollama >/dev/null 2>&1; then
  step "installing Ollama"
  if [[ "$(uname)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      brew install ollama
    else
      echo "ERROR: Ollama is required for live mode."
      echo "Install it from https://ollama.com/download, or run this script "
      echo "without --live for the offline demo."
      exit 1
    fi
  else
    curl -fsSL https://ollama.com/install.sh | sh
  fi
fi

# 3. Python deps + C++ core build
step "installing Python dependencies and building the C++ core"
uv sync --extra dev
uv pip install -e . >/dev/null

# 4. launch the TUI
step "launching ContextMemory brain"
if [[ "$*" == *"--live"* ]] && [[ "$*" != *"--auto-launch"* ]]; then
  # live mode implies auto-launching ollama under the hood
  exec uv run contextmemory demo "$@" --auto-launch
fi
exec uv run contextmemory demo "$@"