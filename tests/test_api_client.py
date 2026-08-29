"""Tests for the public MemoryClient API surface."""

from __future__ import annotations

from contextmemory.api import MemoryClient
from contextmemory.eval.protocol import OpenAICompatClient


class _Extractor:
    """Records whether it was called; returns no cells."""

    def __init__(self) -> None:
        self.called = False

    def extract(self, session) -> list:
        self.called = True
        return []


def test_set_extractor_keeps_store() -> None:
    client = MemoryClient("api-test")
    client.add(
        "The user lives in Berlin",
        subject="user",
        predicate="location",
        object="Berlin",
    )
    assert client.engine.store.cell_count == 1

    swapped = _Extractor()
    client.set_extractor(swapped)
    assert client.engine.store.cell_count == 1  # store survives the swap
    # a fresh client would have an empty store — proves the extractor change
    # did not rebuild the engine
    assert client.projection("user", "location") is not None


def test_swapped_extractor_is_used_on_ingest() -> None:
    from datetime import UTC, datetime

    from contextmemory.eval.protocol import Session, Turn

    client = MemoryClient("api-test2")
    extractor = _Extractor()
    client.set_extractor(extractor)
    client.session(
        Session(
            session_id="s1",
            timestamp=datetime.now(UTC),
            turns=[Turn(role="user", content="hello")],
        )
    )
    assert extractor.called


def test_openai_compat_client_normalizes_v1_suffix() -> None:
    root = OpenAICompatClient(
        "http://localhost:11434", "ollama", "qwen3:8b"
    )
    suffixed = OpenAICompatClient(
        "http://localhost:11434/v1", "ollama", "qwen3:8b"
    )
    assert root._client.base_url == "http://localhost:11434"  # noqa: SLF001
    assert suffixed._client.base_url == "http://localhost:11434"  # noqa: SLF001


def test_recall_defaults_to_now_not_epoch() -> None:
    client = MemoryClient("asof-test")
    client.add(
        "The user lives in Seattle.",
        subject="user",
        predicate="location",
        object="Seattle",
    )
    client.add(
        "The user works at Globex.",
        subject="user",
        predicate="employer",
        object="Globex",
    )
    # No question_date: must evaluate "as of now", not the epoch. An epoch
    # "current state" query sees cells valid_from=now as inactive at 1970 and
    # abstains, which is a silent, hard-to-spot failure.
    report = client.recall("Where does the user live?")
    assert report.sufficient
    assert any("Seattle" in h.text for h in report.hits)