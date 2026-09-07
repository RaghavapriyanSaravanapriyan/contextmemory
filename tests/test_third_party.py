"""Tests for third-party adapters: fail-closed + fairness.

No keys, no network, no vendor packages: the missing-dependency paths and
the prompt/ordering contracts are verified with stubs.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from adapters import SkipError, SupermemorySystem, probe_supermemory  # noqa: E402


def test_probe_fails_closed_without_package(monkeypatch) -> None:
    monkeypatch.delitem(sys.modules, "supermemory", raising=False)
    monkeypatch.setattr("builtins.__import__", _no_supermemory_import)
    ready, reason = probe_supermemory()
    assert ready is False
    assert "pip install supermemory" in reason


def _no_supermemory_import(name, *args, **kwargs):
    if name == "supermemory" or name.startswith("supermemory."):
        raise ImportError("No module named 'supermemory'")
    return _real_import(name, *args, **kwargs)


_real_import = __import__


def test_constructor_fails_closed_without_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delitem(sys.modules, "supermemory", raising=False)
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "")
    # Env without key must fail even before import matters.
    with pytest.raises(SkipError, match="no key"):
        SupermemorySystem(reader=None)


class _FakeDoc:
    def __init__(self, id="d1", status="done"):
        self.id = id
        self.status = status


class _FakeSearch:
    def __init__(self, texts):
        self._texts = texts
        self.seen = []

    def memories(self, **kwargs):
        self.seen.append(kwargs)
        return types.SimpleNamespace(
            results=[types.SimpleNamespace(memory=t) for t in self._texts])


class _FakeClient:
    def __init__(self, texts=("User lives in Seattle.",)):
        self.added = []
        self.search = _FakeSearch(list(texts))
        self.documents = self

    def add(self, **kwargs):
        self.added.append(kwargs)
        return _FakeDoc()

    def get(self, doc_id):
        return _FakeDoc(id=doc_id, status="done")


class _Reader:
    def __init__(self):
        self.prompts = []

    def complete(self, messages, temperature=0.0):
        self.prompts.append(messages[-1]["content"])
        return "Seattle"


def _install_fake(monkeypatch, client):
    mod = types.ModuleType("supermemory")
    mod.Supermemory = lambda *a, **k: client  # noqa: E731
    monkeypatch.setitem(sys.modules, "supermemory", mod)
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")


def test_ingest_polls_to_done_and_scopes_container(monkeypatch) -> None:
    client = _FakeClient()
    _install_fake(monkeypatch, client)
    reader = _Reader()
    sys_ = SupermemorySystem(reader, container_tag="sm-probe-9")
    from contextmemory.eval.protocol import Session, Turn

    sys_.ingest(Session(session_id="s9", timestamp=datetime.now(UTC),
                        turns=[Turn(role="user", content="I live in Seattle.")]))
    assert client.added[0]["container_tag"] == "sm-probe-9"
    assert client.added[0]["customId"] == "s9"
    assert "user: I live in Seattle." in client.added[0]["content"]


def test_answer_uses_same_reader_and_abstains(monkeypatch) -> None:
    client = _FakeClient()
    _install_fake(monkeypatch, client)
    reader = _Reader()
    sys_ = SupermemorySystem(reader, container_tag="sm-probe-9")
    out = sys_.answer("Where do I live?", datetime.now(UTC))
    assert out == "Seattle"  # reader's answer, not the vendor's
    prompt = reader.prompts[0]
    assert "ONLY the memories below" in prompt
    assert "say so explicitly" in prompt  # abstain instruction, like ours
    assert client.search.seen[0]["container_tag"] == "sm-probe-9"
    assert client.search.seen[0]["search_mode"] == "memories"

    empty = _FakeClient(texts=())
    _install_fake(monkeypatch, empty)
    sys2 = SupermemorySystem(reader, container_tag="sm-probe-9")
    assert "don't have enough information" in sys2.answer("Anything?",
                                                          datetime.now(UTC))


def test_ingest_timeout_fails_closed(monkeypatch) -> None:
    class _Slow(_FakeClient):
        def get(self, doc_id):
            return _FakeDoc(id=doc_id, status="dreaming")

    _install_fake(monkeypatch, _Slow())
    sys_ = SupermemorySystem(reader=_Reader(), container_tag="t",
                             ingest_timeout_s=0.01, poll_interval_s=0.005)
    from contextmemory.eval.protocol import Session, Turn

    with pytest.raises(SkipError, match="timed out"):
        sys_.ingest(Session(session_id="s", timestamp=datetime.now(UTC),
                            turns=[Turn(role="user", content="hi")]))
