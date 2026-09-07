"""Third-party memory adapters for head-to-head evaluation.

Each adapter implements the same ``MemorySystem`` protocol as the local
systems (``ingest`` + ``answer``) so every contender runs through the SAME
replay runner, SAME reader model, and SAME judge. No home advantage.

Adapters fail CLOSED: missing package, key, or backend produces a
``SkipError`` with the exact fix, never silent zeros. A skipped contender
is reported as skipped — not as a score.
"""

from __future__ import annotations

from .supermemory_adapter import (
    SkipError,
    SupermemorySystem,
    probe_supermemory,
)

__all__ = ["SkipError", "SupermemorySystem", "probe_supermemory"]
