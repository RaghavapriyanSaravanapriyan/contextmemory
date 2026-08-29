@echo off
rem ContextMemory one command for Windows (cmd.exe - no PowerShell needed).
rem   run.bat                          offline demo, zero config
rem   run.bat --live                   connect to / auto-launch Ollama
rem   run.bat --live --model qwen3:8b  pick the model up front
rem
rem run.py checks every dependency, installs what's missing (uv, C++ compiler
rem via winget/MSVC, Ollama for --live), builds the C++ core, launches the TUI.

setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
    py run.py %*
    exit /b %errorlevel%
)

where python >nul 2>nul
if %errorlevel%==0 (
    python run.py %*
    exit /b %errorlevel%
)

echo ERROR: Python 3 is required. Install it from https://www.python.org/downloads/
exit /b 1