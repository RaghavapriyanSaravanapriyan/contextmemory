"""Regression tests for the speed + correctness pass.

Locks in the bugs fixed while making streaming and the C++ core faster:
tag-index population, forget semantics, entity seeding, streaming payload
shape, and incremental thinking suppression.
"""

from __future__ import annotations

from datetime import UTC, datetime

from contextmemory.core import CellInput, MemoryStore, to_ms
from contextmemory.engine.ollama import (
    _strip_think_incremental,
    strip_thinking,
)

T0 = to_ms(datetime(2024, 1, 1, tzinfo=UTC))
DAY = 86_400_000


def _cell(store, text, subject="", predicate="", obj="", ts=T0, tags=None,
          entities=None):
    return store.reconcile(
        CellInput(
            text=text,
            subject=subject,
            predicate=predicate,
            object=obj,
            observed_at=ts,
            valid_from=ts,
            tags=tags or [],
            entities=entities or [],
        )
    )


def test_tags_are_indexed_before_search() -> None:
    store = MemoryStore("tag-reg")
    _cell(store, "User lives in Seattle.", "user", "location", "Seattle",
          tags=["location"], entities=["Seattle"])
    plan = store.compile("What is the location status?", T0 + DAY)
    assert "location" in plan.tags


def test_forget_excludes_cell_from_search() -> None:
    store = MemoryStore("forget-reg")
    cid = _cell(store, "User lives in Seattle.", "user", "location",
                "Seattle")
    assert store.forget(cid) is True
    assert store.forget(cid) is False  # already forgotten
    assert store.forget(999999) is False  # unknown id
    plan = store.compile("Where does the user live?", T0 + DAY)
    assert store.search(plan) == []


def test_forget_beats_projection_bypass() -> None:
    # The old search allowed projection hits to bypass the active check,
    # so forgotten cells still surfaced as "current truth".
    store = MemoryStore("forget-proj")
    cid = _cell(store, "User prefers vim.", "user", "preference", "vim")
    assert store.projection("user", "preference") is not None
    store.forget(cid)
    plan = store.compile("What does the user prefer?", T0 + DAY)
    assert all(h.cell_id != cid for h in store.search(plan))


def test_entity_ngram_seeding_finds_multitoken_entity() -> None:
    store = MemoryStore("entity-reg")
    _cell(store, "User visited New York last week.", entities=["New York"])
    plan = store.compile("Tell me about New York.", T0 + DAY)
    assert "New York" in plan.entity_seeds


def test_mcp_forget_actually_forgets() -> None:
    from contextmemory.mcp import MCPServer

    srv = MCPServer(container="forget-mcp-test")
    r1 = srv._dispatch("memory", {"content": "I live in Seattle."})
    assert "cell(s)" in r1
    before = srv._dispatch("recall", {"query": "where does the user live"})
    assert "Seattle" in before
    client = srv._client("")
    store = client.engine.store
    # Forget every cell the recall surfaced.
    for h in client.recall("where does the user live").hits:
        assert store.forget(h.cell_id) is True
    msg = srv._dispatch("forget", {"id": 1})
    assert isinstance(msg, str)
    after = srv._dispatch("recall", {"query": "where does the user live"})
    assert "Seattle" not in after


def test_stream_payload_uses_stream_keepalive_and_ctx() -> None:
    import httpx

    from contextmemory.engine.ollama import OllamaChatClient

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        seen.update(_json.loads(request.content.decode()))
        # Minimal NDJSON stream: one content chunk + done.
        body = (
            '{"message": {"content": "hi"}, "done": false}\n'
            '{"message": {"content": ""}, "done": true}\n'
        )
        return httpx.Response(200, text=body)

    client = OllamaChatClient("http://localhost:11434", "qwen2.5:1.5b")
    client._client = httpx.Client(  # noqa: SLF001
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    assert list(client.stream_complete([{"role": "user",
                                         "content": "hi"}])) == ["hi"]
    assert seen["stream"] is True
    assert seen["keep_alive"] == "10m"
    assert seen["options"]["num_ctx"] == 2048


def test_incremental_think_suppression() -> None:
    out1, in_think = _strip_think_incremental("hello <think>secret", False)
    assert out1 == "hello "
    assert in_think is True
    out2, in_think2 = _strip_think_incremental("still secret</think> world",
                                               in_think)
    assert out2 == " world"
    assert in_think2 is False
    assert strip_thinking("<think>x</think>ok") == "ok"
