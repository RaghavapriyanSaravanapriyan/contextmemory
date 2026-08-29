#!/usr/bin/env python3
"""ContextMemory one-command installer + TUI launcher.

One command gets you from a bare machine to the running brain:

    ./run.sh                        # offline demo (no model, zero config)
    ./run.sh --live                 # connect to / auto-launch Ollama
    ./run.sh --live --model qwen3:8b --url http://localhost:11434

What this script does, in order:

  1. checks Python is new enough (3.11+; `uv` fetches a managed one if not)
  2. installs `uv` if missing (the only external dependency)
  3. installs a C++ compiler if missing (Linux: g++/clang++; macOS: Xcode CLT;
     Windows: MSVC Build Tools) -- the C++ core is a real build
  4. installs Ollama only for --live mode
  5. `uv sync` installs Python deps and builds the C++ core
  6. launches the TUI, forwarding all your flags

Everything that can be installed automatically is. If a step really needs
manual action (macOS Xcode dialog, huge Windows toolchain), the script tells
you exactly what to run. Works on Linux, macOS, and Windows.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PY_MIN = (3, 11)

# --live means we need the ollama binary; offline demo needs nothing.
LIVE = "--live" in sys.argv[1:]

STEALTH = {
    "darwin": ["-fs", "-Sl", "https://astral.sh/uv/install.sh"],
    "linux": ["-fsSL", "https://astral.sh/uv/install.sh"],
}
WINDOWS = os.name == "nt"


def step(msg: str) -> None:
    print(f"\n==> {msg}")


def info(msg: str) -> None:
    print(f"    {msg}")


def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    info("$ " + " ".join(cmd))
    return subprocess.run(cmd, **kw)  # type: ignore[arg-type]


def have(name: str) -> bool:
    return shutil.which(name) is not None


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def sh_install_uv() -> None:
    """Install uv via the official one-liner for Unix-like systems."""
    script = STEALTH["darwin" if sys.platform == "darwin" else "linux"]
    if not have("curl"):
        fail(
            "This machine has no `curl` and no `uv`. Install curl first "
            "(e.g. `apt-get install curl`), then re-run."
        )
    step("installing uv (official installer)")
    cmd = "curl {} | sh".format(" ".join(script))
    ok = subprocess.run(["bash", "-c", cmd]).returncode
    if ok != 0:
        fail("Failed to install uv. Re-run manually:")
        info(cmd)


def win_install_uv() -> None:
    step("installing uv (official PowerShell installer)")
    cmd = (
        "powershell -ExecutionPolicy ByPass -c "
        '"irm https://astral.sh/uv/install.ps1 | iex"'
    )
    if run(["cmd", "/c", cmd]).returncode != 0:
        fail(
            "Failed to install uv. Run manually in PowerShell:\n"
            '    irm https://astral.sh/uv/install.ps1 | iex'
        )


def ensure_uv() -> str:
    """Return the uv executable path, installing it if necessary."""
    if have("uv"):
        info("uv: installed")
        return "uv"
    # uv installs to ~/.local/bin on all platforms; make sure PATH finds it.
    local_bin = os.path.expanduser(
        os.path.join("~", ".local", "bin", "uv.exe" if WINDOWS else "uv")
    )
    if os.path.isfile(local_bin):
        info("uv: installed (not on PATH)")
        return local_bin
    if WINDOWS:
        win_install_uv()
    else:
        sh_install_uv()
    if have("uv"):
        return "uv"
    if os.path.isfile(local_bin):
        return local_bin
    fail(
        "uv installed but not found. Add ~/.local/bin to PATH, "
        "or re-run with: export PATH=\"$HOME/.local/bin:$PATH\""
    )
    return local_bin  # unreachable; keeps type-checkers quiet


# --- C++ compiler ---------------------------------------------------------


def ensure_compiler() -> None:
    """Make sure a C++ compiler exists, installing one when possible."""
    if sys.platform == "darwin":
        ensure_macos_compiler()
    elif WINDOWS:
        ensure_windows_compiler()
    else:
        ensure_linux_compiler()


def ensure_macos_compiler() -> None:
    if have("clang++") and have("make"):
        info("compiler: clang++ (Xcode CLT)")
        return
    step("installing Xcode Command Line Tools (C++ compiler)")
    info("Running `xcode-select --install` -- approve the dialog that pops up.")
    run(["xcode-select", "--install"])
    print()
    info("Waiting for the Command Line Tools to finish installing...")
    for _ in range(60):  # up to ~5 minutes
        time.sleep(5)
        if have("clang++") and have("make"):
            info("compiler: clang++ (Xcode CLT)")
            return
    fail(
        "Xcode Command Line Tools are still installing. Once the dialog "
        "completes, re-run this script."
    )


def ensure_linux_compiler() -> None:
    if have("g++") or have("clang++"):
        info("compiler: " + ("g++" if have("g++") else "clang++"))
        return
    step("installing C++ compiler (g++)")
    # Common package managers; try the first one present.
    for pm in (
        ["apt-get", "install", "-y", "--no-install-recommends", "g++"],
        ["dnf", "install", "-y", "gcc-c++"],
        ["pacman", "-S", "--noconfirm", "gcc"],
        ["apk", "add", "g++"],
    ):
        if not have(pm[0]):
            continue
        info(f"trying: {' '.join(pm)}")
        sudo = ["sudo"] if os.geteuid() != 0 else []
        if run([*sudo, *pm]).returncode == 0 and have("g++"):
            info("compiler: g++")
            return
        break  # only try the first available package manager
    fail(
        "No C++ compiler found and none could be auto-installed. "
        "Install g++ (e.g. `sudo apt-get install g++`), then re-run."
    )


def ensure_windows_compiler() -> None:
    # MSVC via Visual Studio / Build Tools, or MinGW as a fallback.
    if have("g++") or have("cl") or _msvc_present():
        info("compiler: " + ("g++ (MinGW)" if have("g++") else "MSVC"))
        return
    step("installing Visual Studio Build Tools (C++ compiler)")
    if have("winget"):
        info("winget: installing MSVC Build Tools (this is a large download)")
        res = run(
            [
                "winget", "install", "--id", "Microsoft.VisualStudio.2022.BuildTools",
                "--silent", "--accept-package-agreements",
                "--accept-source-agreements", "--override",
                "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools "
                "--includeRecommended",
            ]
        )
        if res.returncode == 0 and _msvc_present():
            info("compiler: MSVC Build Tools")
            return
        fail(
            "MSVC Build Tools install did not complete. Re-run this script "
            "after installing them, or install MinGW (`winget install "
            "BrechtSanders.WinLibs.POSIX.UCRT`)."
        )
    fail(
        "No C++ compiler found. Install Visual Studio Build Tools with the "
        "C++ workload (https://visualstudio.microsoft.com/downloads/), "
        "then re-run."
    )


def _msvc_present() -> bool:
    vs = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if not os.path.isfile(vs):
        return False
    res = run([vs, "-latest", "-products", "*", "-requires",
               "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"],
              capture_output=True)
    return res.returncode == 0 and bool(res.stdout.decode().strip())


# --- Ollama (live mode only) ---------------------------------------------


def ensure_ollama() -> None:
    if not LIVE:
        info("ollama: skipped (offline demo needs no model)")
        return
    if have("ollama"):
        info("ollama: installed")
        return
    step("installing Ollama")
    if WINDOWS:
        if have("winget"):
            cmd = ["winget", "install", "-e", "--id", "Ollama.Ollama"]
            if run(cmd).returncode != 0:
                fail("Failed to install Ollama. Install from https://ollama.com")
        else:
            fail("Install Ollama from https://ollama.com, then re-run.")
    elif sys.platform == "darwin" and have("brew"):
        if run(["brew", "install", "ollama"]).returncode != 0:
            fail("Failed to install Ollama. Install from https://ollama.com")
    else:
        cmd = "curl -fsSL https://ollama.com/install.sh | sh"
        if run(["bash", "-c", cmd]).returncode != 0:
            fail("Failed to install Ollama. Install from https://ollama.com")
    if not have("ollama"):
        fail("Ollama installed but not on PATH. Re-open your terminal and re-run.")


# --- build + launch -------------------------------------------------------


def build_and_launch(uv: str, args: list[str]) -> None:
    step("installing Python dependencies and building the C++ core")
    res = run([uv, "sync", "--extra", "dev"], cwd=ROOT)
    if res.returncode != 0:
        fail("`uv sync` failed. Scroll up to see the build error.")

    # Fail fast if the C++ core did not build.
    probe = run([uv, "run", "python", "-c",
                 "from contextmemory.core import MemoryStore; "
                 "MemoryStore('probe')"], cwd=ROOT)
    if probe.returncode != 0:
        fail("The C++ core did not build. Scroll up to see why.")

    demo_args = [a for a in args if a != "--auto-launch"]
    if LIVE and "--auto-launch" not in demo_args:
        demo_args.append("--auto-launch")
    step("launching ContextMemory brain")
    os.execvp(uv, [uv, "run", "contextmemory", "demo", *demo_args])


def main() -> int:
    print("ContextMemory — one-command setup")

    # 1. Python
    v = sys.version_info
    if v < PY_MIN:
        step("checking Python")
        info("found Python {}.{}.{} (need 3.11+)".format(*v[:3]))
        info("`uv` can fetch a managed Python for the project; continuing...")
    else:
        info("Python {}.{}.{}: OK".format(*v[:3]))

    # 2. uv
    step("checking uv")
    uv = ensure_uv()

    # 3. C++ compiler
    step("checking C++ compiler")
    ensure_compiler()

    # 4. Ollama (only for --live)
    step("checking Ollama")
    ensure_ollama()

    # 5-6. build + launch
    build_and_launch(uv, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())