# ContextMemory one command for Windows PowerShell.
#   .\run.ps1                          # offline demo, zero config
#   .\run.ps1 --live                   # connect to / auto-launch Ollama
#   .\run.ps1 --live --model qwen3:8b  # pick the model up front
#
# Tip: run.bat does the same thing from plain cmd.exe with no execution-policy
# hassles. Both just delegate to run.py, which checks every dependency,
# installs whatever is missing, builds the C++ core, and launches the TUI.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (Get-Command py -ErrorAction SilentlyContinue) {
    py run.py $args
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    python run.py $args
}
else {
    Write-Host "ERROR: Python 3 is required. Install it from https://www.python.org/downloads/"
    exit 1
}