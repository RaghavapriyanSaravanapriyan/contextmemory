#!/usr/bin/env bash

set -euo pipefail

SESSION="contextmemory"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux attach-session -t "$SESSION"
    exit 0
fi

tmux new-session -d -s "$SESSION" -n "lab" -c "$ROOT"
tmux split-window -h -t "$SESSION:0" -c "$ROOT"

tmux send-keys -t "$SESSION:0.0" \
    "clear; echo 'THINK TEAM 🧠 — ARCHITECT + RESEARCHER'; echo; opencode" C-m

tmux send-keys -t "$SESSION:0.1" \
    "clear; echo 'FORGE TEAM ⚙️ — CODER + TESTER'; echo; opencode" C-m

tmux select-pane -t "$SESSION:0.0"

tmux attach-session -t "$SESSION"
