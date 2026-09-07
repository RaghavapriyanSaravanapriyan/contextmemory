"""Regression tests for the production-hardening pass.

Covers: in-memory client isolation, config forward-compat, cross-session
dup accounting, judge word-boundary parsing, MCP forget validation,
observed_at plumbing, out-of-order projection stability, and FFI enum
validation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from contextmemory import config as cfgmod
from contextmemory.api import MemoryClient
from contextmemory.config import AppConfig
from contextmemory.eval.protocol import Session, Turn


def _session(sid: str, content: str) -> Session:
    return Session(
        session_id=sid,
        timestamp=datetime.now(UTC),
        turns=[Turn(role="user", content=content)],
    )


def test_client_none_journal_is_in_memory(tmp_path) -> None:
    client = MemoryClient("iso-test", journal_path=None)
    client.add("The user lives in Seattle.", subject="user",
               predicate="location", object="Seattle")
    assert client.engine.store.cell_count == 1
    # No journal file anywhere under the isolated data dir.
    leftovers = list(tmp_path.rglob("*.cm.bin"))
    assert leftovers == []


def test_client_default_journal_persists(tmp_path) -> None:
    first = MemoryClient("persist-test")
    first.add("The user lives in Seattle.", subject="user",
              predicate="location", object="Seattle")
    first.engine.persist()
    assert list(tmp_path.rglob("persist-test.cm.bin"))

    second = MemoryClient("persist-test")
    assert second.engine.store.cell_count == 1
    report = second.recall("Where does the user live?")
    assert any("Seattle" in h.text for h in report.hits)


def test_config_ignores_unknown_keys(tmp_path) -> None:
    cfg = AppConfig.load()
    cfg.building = "research"
    cfg.save()
    # Simulate a newer binary / hand edit adding a key.
    path = cfgmod._config_path()  # noqa: SLF001
    data = json.loads(path.read_text(encoding="utf-8"))
    data["future_key"] = "hello"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = AppConfig.load()  # must not raise TypeError
    assert loaded.building == "research"


def test_cross_session_dup_accounting() -> None:
    client = MemoryClient("dup-test", journal_path=None)
    r1 = client.session(_session("s1", "I live in Seattle and prefer Vim."))
    assert r1.new_cells >= 1
    total_after_first = client.engine.cells_ingested
    r2 = client.session(_session("s2", "I live in Seattle and prefer Vim."))
    # Same content re-ingested: dups, not new cells.
    assert r2.new_cells == 0
    assert r2.dup_cells == r2.cells
    assert client.engine.cells_ingested == total_after_first


def test_judge_yes_word_boundary() -> None:
    from contextmemory.eval.scoring import _judge_yes  # noqa: SLF001

    assert _judge_yes("yes")
    assert _judge_yes("Yes, the response is correct.")
    assert _judge_yes("The answer is yes because it matches.")
    assert not _judge_yes("no")
    assert not _judge_yes("No, it is wrong.")
    assert not _judge_yes("eyes yesterday")  # substring trap
    assert not _judge_yes("I don't have enough information.")


def test_judge_failure_does_not_abort_run(fake_reader) -> None:
    from contextmemory.eval.runner import ReplayResult, Timing
    from contextmemory.eval.scoring import judge_results

    class _Flaky:
        def complete(self, messages, temperature=0.0):
            raise RuntimeError("judge exploded")

    results = [
        ReplayResult(
            question_id="q1", question_type="t", question="q",
            answer="a", hypothesis="h", is_abstention=False,
            timing=Timing(),
        )
    ]
    report, labeled = judge_results(results, _Flaky())
    assert report.n == 0  # excluded, not crashed
    assert labeled[0].judged is None


def test_mcp_forget_rejects_bad_ids() -> None:
    from contextmemory.mcp import MCPServer

    server = MCPServer(container="forget-hard-test")
    assert "integer" in server._dispatch("forget", {"id": "abc"})
    assert "positive" in server._dispatch("forget", {"id": -3})
    assert "positive" in server._dispatch("forget", {})
    assert "not found" in server._dispatch("forget", {"id": 999999})


def test_search_hit_carries_observed_at() -> None:
    client = MemoryClient("obs-test", journal_path=None)
    ts = datetime(2024, 5, 1, tzinfo=UTC)
    client.add("The user lives in Seattle.", subject="user",
               predicate="location", object="Seattle", ts=ts)
    report = client.recall("Where does the user live?")
    assert report.hits
    assert report.hits[0].observed_at == int(ts.timestamp() * 1000)


def test_out_of_order_never_rewinds_projection() -> None:
    client = MemoryClient("rewind-test", journal_path=None)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    t1 = datetime(2024, 6, 1, tzinfo=UTC)
    client.add("User lives in New York.", subject="user",
               predicate="location", object="New York", ts=t0)
    client.add("User moved to Seattle.", subject="user",
               predicate="location", object="Seattle", ts=t1)
    # Late-arriving older event.
    client.add("User lived in New York before.", subject="user",
               predicate="location", object="New York", ts=t0)
    proj = client.projection("user", "location")
    assert proj is not None
    active = next(
        h for h in client.recall("Where does the user live?").hits
        if h.cell_id == proj.active_cell
    )
    assert active.object == "Seattle"


def test_reconcile_rejects_bad_kind() -> None:
    from contextmemory.core import MemoryStore

    store = MemoryStore("kind-test")
    with pytest.raises(ValueError):
        store._store.reconcile(  # noqa: SLF001
            "", "", "", "bad kind cell", 255, 0, 0, 1.0, 0.5, "", 0, 0,
            [], [],
        )


def test_word_boundary_routing() -> None:
    from contextmemory.core import MemoryStore

    store = MemoryStore("route-test")
    plan = store.compile("Where should I wash the gold dishes?", 0)
    assert plan.time_mode != 1  # not Historical ("was"/"old" traps)
    plan2 = store.compile("Why is the light blinking?", 0)
    assert plan2.relation_mode != 1  # not MultiHop ("link" trap)
