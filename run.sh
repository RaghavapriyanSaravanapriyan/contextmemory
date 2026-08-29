#!/usr/bin/env bash
# ContextMemory one command: check deps -> install what's missing -> build ->
# launch the TUI. Thin wrapper around run.py (which does the real work).
#
#   ./run.sh                          # offline demo, zero config
#   ./run.sh --live                   # connect to / auto-launch Ollama
#   ./run.sh --live --model qwen3:8b --url http://localhost:11434
#
# On Windows use run.bat (cmd.exe) or run.ps1 (PowerShell) instead.

set -euo pipefail
cd "$(dirname "$0")"

# Find a Python 3 interpreter (run.py works on any modern 3.x and self-heals).
if command -v python3 >/dev/null 2>&1; then
  exec python3 run.py "$@"
elif command -v python >/dev/null 2>&1; then
  exec python run.py "$@"
else
  echo "ERROR: Python 3 is required. Install it from https://www.python.org/downloads/"
  exit 1
fi
