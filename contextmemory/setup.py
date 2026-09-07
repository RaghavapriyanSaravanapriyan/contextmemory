"""Interactive setup wizard for ContextMemory (stdlib only, no bloat).

``contextmemory setup`` walks a new user through the three questions that
matter — what they are building, which model provider to use, and which
memory container to keep — probes the provider for reachability, and writes
the result to the platform config dir via :class:`AppConfig`.

Re-running setup is safe: existing values become the defaults.
"""

from __future__ import annotations

import sys

from .config import BUILDING_CHOICES, DEFAULT_BUILDING, AppConfig

_OLLAMA_DEFAULT = "http://localhost:11434"


def _prompt(text: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    try:
        raw = input(f"{text}{hint}: ").strip()
    except EOFError:
        return default
    return raw or default


def _choose_numbered(title: str, options: list[str], default_idx: int = 0) -> int:
    print(f"\n{title}")
    for i, opt in enumerate(options):
        mark = " (current)" if i == default_idx else ""
        print(f"  {i + 1}) {opt}{mark}")
    while True:
        raw = _prompt(f"Pick 1-{len(options)}", str(default_idx + 1))
        try:
            idx = int(raw) - 1
        except ValueError:
            print("Enter a number.", file=sys.stderr)
            continue
        if 0 <= idx < len(options):
            return idx
        print(f"Enter 1-{len(options)}.", file=sys.stderr)


def _probe_ollama(base_url: str, timeout: float = 2.0) -> list[str]:
    """Return model names from an Ollama server, or [] when unreachable."""
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(
            base_url.rstrip("/") + "/api/tags", timeout=timeout
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    models = []
    for m in data.get("models", []) or []:
        name = m.get("name") or m.get("model")
        if name:
            models.append(name)
    return models


def run_setup() -> int:
    print("ContextMemory setup — local-first memory for your model.\n")
    cfg = AppConfig.load()

    # 1. What are you building?
    keys = list(BUILDING_CHOICES)
    labels = list(BUILDING_CHOICES.values())
    try:
        cur_building = keys.index(cfg.building)
    except ValueError:
        cur_building = 0
    building = keys[
        _choose_numbered("What are you building?", labels, cur_building)
    ]

    # 2. Provider.
    providers = ["ollama (local, recommended)", "offline (no model)"]
    cur_prov = 0 if cfg.provider in ("", "ollama") else 1
    provider_choice = _choose_numbered("Model provider?", providers, cur_prov)

    base_url = cfg.base_url or _OLLAMA_DEFAULT
    model = cfg.model
    if provider_choice == 0:
        base_url = _prompt("Ollama base URL", base_url)
        print("Probing Ollama…")
        found = _probe_ollama(base_url)
        if found:
            print(f"Ollama reachable ({len(found)} model(s)).")
            default_idx = found.index(model) if model in found else 0
            model = found[
                _choose_numbered("Which model?", found, default_idx)
            ]
            provider = "ollama"
        else:
            print(
                f"Ollama not reachable at {base_url}. "
                "Setup continues; start it later with `ollama serve` "
                "and re-run `contextmemory setup` to pick a model.",
                file=sys.stderr,
            )
            model = _prompt(
                "Model name (e.g. qwen3:4b, empty = auto-detect later)", model
            )
            provider = "ollama"
    else:
        provider = ""
        model = ""

    # 3. Container.
    container = _prompt("Memory container tag", cfg.container or "brain")

    cfg.complete_onboarding(
        building=building or DEFAULT_BUILDING,
        provider=provider,
        model=model,
        base_url=base_url,
        container=container or "brain",
    )
    print("\nSaved configuration:")
    print(f"  building:  {cfg.resolve_building_label()}")
    print(f"  provider:  {provider or 'offline'}")
    print(f"  model:     {model or '(automatic)'}")
    print(f"  container: {cfg.container}")
    print("\nNext steps:")
    print("  contextmemory demo            # watch the brain work")
    if provider == "ollama":
        print(
            f"  contextmemory chat --model {model or '<model>'}  # talk with memory"
        )
    print("  contextmemory mcp --container brain  # MCP bridge for agents")
    return 0


def show_config() -> int:
    cfg = AppConfig.load()
    print(f"onboarded: {cfg.onboarded}")
    print(f"building:  {cfg.resolve_building_label()} ({cfg.building})")
    print(f"provider:  {cfg.provider or 'offline'}")
    print(f"model:     {cfg.model or '(automatic)'}")
    print(f"base_url:  {cfg.base_url}")
    print(f"container: {cfg.container}")
    return 0
