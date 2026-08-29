"""ContextMemory product configuration.

A tiny local-first config that answers three questions:

* has onboarding been completed? (first-run detection)
* what is the user building? (an optional profile hint)
* which provider + model should the live brain use?

Stored as JSON under the platform config dir. No secrets beyond what the user
types; Ollama defaults are local by design.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT = "contextmemory"


def _config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    elif platform.system() == "Darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / PROJECT


def _config_path() -> Path:
    return _config_dir() / "config.json"


DEFAULT_BUILDING = "ai-agent"
BUILDING_CHOICES = {
    "ai-agent": "AI Agent",
    "ai-assistant": "AI Assistant",
    "coding-agent": "Coding Agent",
    "application": "Application",
    "research": "Research Project",
    "custom": "Custom Setup",
}


@dataclass
class AppConfig:
    """Persisted product-level configuration."""

    onboarded: bool = False
    building: str = DEFAULT_BUILDING
    provider: str = ""            # "ollama" | "" (offline)
    model: str = ""               # selected model name ("" = automatic)
    base_url: str = "http://localhost:11434"
    strategy: str = "automatic"   # automatic | single | advanced
    container: str = "brain"

    def resolve_building_label(self) -> str:
        return BUILDING_CHOICES.get(self.building, self.building)

    @classmethod
    def load(cls) -> AppConfig:
        path = _config_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f: getattr(cls(), f) for f in cls.__dataclass_fields__}
        known.update(data)
        return cls(**{k: v for k, v in known.items()})

    def save(self) -> None:
        path = _config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    def complete_onboarding(
        self, *, building: str, provider: str, model: str,
        base_url: str, container: str = "brain",
    ) -> None:
        self.onboarded = True
        self.building = building
        self.provider = provider
        self.model = model
        self.base_url = base_url
        self.container = container
        self.save()


__all__ = ["AppConfig"]