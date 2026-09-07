"""Shared test fixtures: a deterministic fake reader."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_dirs(tmp_path, monkeypatch):
    """Keep tests off the real user journals and config.

    MemoryClient defaults to persistent per-container journals under the
    platform data dir; without isolation every test run writes into
    ~/.local/share/contextmemory. Point both dirs at tmp per test.
    """
    (tmp_path / "config").mkdir()
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # Windows/macOS branches read other vars; belt and suspenders.
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path / "config"))
    yield


class FakeReader:
    """Deterministic reader for tests.

    ``behaviors`` maps a substring of the prompt to a canned response;
    unmatched prompts fall back to ``default``. Lets tests assert both
    prompt construction and system behavior without a model.
    """

    def __init__(
        self,
        behaviors: dict[str, str] | None = None,
        default: str = "42",
    ) -> None:
        self._behaviors = behaviors or {}
        self._default = default
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.0) -> str:
        self.calls.append(messages)
        content = messages[-1]["content"]
        for needle, response in self._behaviors.items():
            if needle in content:
                return response
        return self._default


@pytest.fixture
def fake_reader() -> FakeReader:
    return FakeReader()