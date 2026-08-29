"""Tests for the Python wrapper around the C++ core engine."""

from __future__ import annotations

from datetime import UTC, datetime

from contextmemory.core import PREFERENCE, WORLD, MemoryStore

T0 = int(datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1000)
DAY = 86_400_000


def test_add_and_search_roundtrip() -> None:
    store = MemoryStore("test")
    fid = store.add_fact(
        "User prefers TypeScript over Python",
        kind=WORLD,
        is_static=True,
        ts=T0,
        entities=["User"],
    )
    assert store.fact_count == 1
    hits = store.search("programming language TypeScript", at_time=T0 + DAY)
    assert hits and hits[0].fact_id == fid


def test_update_versions_and_invalidates() -> None:
    store = MemoryStore("test")
    fid = store.add_fact("User lives in New York", is_static=True, ts=T0)
    new_id = store.update_fact(fid, "User lives in San Francisco", ts=T0 + 30 * DAY)
    assert new_id != fid
    hits = store.search("where does user live", at_time=T0 + 31 * DAY)
    assert any("San Francisco" in h.text for h in hits)
    assert all("New York" not in h.text for h in hits)

    expired = store.search(
        "where does user live", at_time=T0 + 31 * DAY, include_expired=True
    )
    assert any("New York" in h.text for h in expired)


def test_temporal_abstention() -> None:
    store = MemoryStore("test")
    store.add_fact("User has a statistics exam tomorrow", kind=3, ts=T0)
    assert store.search("statistics exam", at_time=T0).__len__() == 1
    store.expire(1, T0 + DAY)
    assert store.search("statistics exam", at_time=T0 + 2 * DAY) == []


def test_forget_removes_from_retrieval() -> None:
    store = MemoryStore("test")
    store.add_fact("User dislikes cilantro", kind=PREFERENCE, ts=T0)
    store.forget(1, T0 + DAY)
    assert store.search("cilantro", at_time=T0 + 2 * DAY) == []


def test_container_isolation() -> None:
    a = MemoryStore("user_a")
    b = MemoryStore("user_b")
    a.add_fact("User has a pet turtle", ts=T0)
    b.add_fact("User rides a motorcycle", ts=T0)
    assert a.search("turtle", at_time=T0 + DAY)
    assert not a.search("motorcycle", at_time=T0 + DAY)
    assert not b.search("turtle", at_time=T0 + DAY)
    assert b.search("motorcycle", at_time=T0 + DAY)


def test_profile_static_and_dynamic() -> None:
    store = MemoryStore("test")
    store.add_fact("User is a senior engineer", is_static=True, ts=T0)
    store.add_fact("User prefers vim", is_static=True, ts=T0)
    store.add_fact("User asked about graph databases", kind=3, ts=T0)
    prof = store.profile(at_time=T0 + DAY)
    assert [f.text for f in prof.static_facts] == [
        "User is a senior engineer",
        "User prefers vim",
    ]
    assert len(prof.dynamic_facts) == 1


def test_token_budget_truncates() -> None:
    store = MemoryStore("test")
    for i in range(10):
        store.add_fact(f"User prefers the color blue variant number {i}", ts=T0)
    hits = store.search(
        "blue color variant", at_time=T0 + DAY, token_budget=30, top_k=10
    )
    assert 0 < len(hits) < 10


def test_save_load_roundtrip(tmp_path) -> None:
    path = str(tmp_path / "mem.bin")
    store = MemoryStore("test")
    fid = store.add_fact("User lives in Berlin", ts=T0, entities=["Berlin"])
    store.add_embedding(fid, [1.0, 0.0, 0.0, 0.0])
    store.update_fact(fid, "User lives in Prague", ts=T0 + 30 * DAY)
    store.save(path)

    loaded = MemoryStore("test")
    loaded.load(path)
    assert loaded.fact_count == store.fact_count == 2
    assert loaded.edge_count == 1
    assert loaded.entity_count == 1
    hits = loaded.search("where does user live", at_time=T0 + 60 * DAY)
    assert any("Prague" in h.text for h in hits)
    assert all("Berlin" not in h.text for h in hits)


def test_entity_channel_surfaces_linked_facts() -> None:
    store = MemoryStore("test")
    store.add_fact("Wrote the v1 of the billing service", ts=T0, entities=["Acme Corp"])
    store.add_fact("Prefers dark mode", ts=T0)
    hits = store.search(
        "What did the user build?", query_entities=["Acme Corp"], at_time=T0 + DAY
    )
    assert hits and hits[0].fact_id == 1