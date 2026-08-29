"""Tests for the Python facade over the C++ ETMC core engine."""

from __future__ import annotations

from datetime import UTC, datetime

from contextmemory.core import (
    EXPERIENCE,
    PREFERENCE,
    T_CURRENT,
    T_HISTORICAL,
    WORLD,
    CellInput,
    MemoryStore,
    to_ms,
)

T0 = to_ms(datetime(2023, 1, 1, tzinfo=UTC))
DAY = 86_400_000


def _cell(store, text, subject="", predicate="", obj="", ts=T0, kind=WORLD,
          tags=None, entities=None):
    return store.reconcile(
        CellInput(
            text=text,
            subject=subject,
            predicate=predicate,
            object=obj,
            kind=kind,
            observed_at=ts,
            valid_from=ts,
            tags=tags or [],
            entities=entities or [],
        )
    )


def test_add_search_and_projection_roundtrip() -> None:
    store = MemoryStore("test")
    fid = _cell(
        store,
        "User prefers TypeScript over Python",
        subject="user",
        predicate="preference",
        obj="TypeScript",
        kind=WORLD,
        entities=["user"],
    )
    assert store.cell_count == 1
    proj = store.projection("user", "preference")
    assert proj is not None
    assert proj.active_cell == fid
    plan = store.compile("What does the user prefer?", T0 + DAY)
    assert plan.predicate_hint == "preference"
    hits = store.search(plan)
    assert hits and hits[0].cell_id == fid


def test_update_versions_and_invalidates() -> None:
    store = MemoryStore("test")
    fid = _cell(store, "User lives in New York", "user", "location",
                "New York", T0)
    new_id = _cell(store, "User lives in San Francisco", "user", "location",
                   "San Francisco", T0 + 30 * DAY)
    assert new_id != fid
    assert store.cell_count == 2
    assert store.edge_count == 1
    proj = store.projection("user", "location")
    assert proj and proj.active_cell == new_id and proj.version_count == 2

    plan = store.compile("Where does the user live?", T0 + 31 * DAY)
    assert plan.time_mode == T_CURRENT
    hits = store.search(plan)
    assert any("San Francisco" in h.text for h in hits)
    assert all("New York" not in h.text for h in hits)

    hplan = store.compile("Where did the user live before?", T0 + 31 * DAY)
    assert hplan.time_mode == T_HISTORICAL
    hhits = store.search(hplan)
    assert any("New York" in h.text for h in hhits)


def test_temporal_abstention() -> None:
    store = MemoryStore("test")
    _cell(store, "User has a statistics exam tomorrow", kind=EXPERIENCE, ts=T0)
    plan = store.compile("When is the statistics exam?", T0)
    assert store.search(plan)
    # Expire it: supersede with an explicit "none" state.
    _cell(store, "User no longer has a statistics exam", ts=T0 + DAY)
    plan2 = store.compile("When is the statistics exam?", T0 + 2 * DAY)
    hits = store.search(plan2)
    pack = store.pack(plan2, hits)
    assert pack.sufficient  # current truth = no exam


def test_exact_dedup() -> None:
    store = MemoryStore("test")
    a = _cell(store, "User dislikes cilantro", "user", "preference",
              "cilantro", kind=PREFERENCE)
    b = _cell(store, "User dislikes cilantro", "user", "preference",
              "cilantro", kind=PREFERENCE)
    assert a == b
    assert store.cell_count == 1


def test_late_arriving_event() -> None:
    store = MemoryStore("test")
    _cell(store, "User lives in New York", "user", "location", "New York", T0)
    # Event happened on day 30, observed on day 60.
    store.reconcile(
        CellInput(
            text="User moved to Seattle",
            subject="user",
            predicate="location",
            object="Seattle",
            observed_at=T0 + 60 * DAY,
            valid_from=T0 + 30 * DAY,
        )
    )
    plan = store.compile("Where does the user live?", T0 + 35 * DAY)
    hits = store.search(plan)
    # Agent had not yet learned the move: current truth is New York.
    assert any("New York" in h.text for h in hits)
    assert all("Seattle" not in h.text for h in hits)


def test_container_isolation() -> None:
    a = MemoryStore("user_a")
    b = MemoryStore("user_b")
    _cell(a, "User has a pet turtle", "user", "pet", "turtle")
    _cell(b, "User rides a motorcycle", "user", "transport", "motorcycle")
    pa = a.compile("What is the user's pet?", T0 + DAY)
    pb = b.compile("What is the user's pet?", T0 + DAY)
    assert a.search(pa)
    assert not b.search(pb)


def test_profile_static_and_dynamic() -> None:
    store = MemoryStore("test")
    _cell(store, "User is a senior engineer", "user", "role", "engineer",
          kind=WORLD, ts=T0)
    _cell(store, "User prefers vim", "user", "preference", "vim",
          kind=PREFERENCE, ts=T0)
    _cell(store, "User asked about graph databases", kind=EXPERIENCE, ts=T0)
    prof = store.profile(T0 + DAY)
    assert len(prof.static_facts) == 2
    assert len(prof.dynamic_facts) == 1


def test_token_budget_truncates() -> None:
    from dataclasses import replace

    store = MemoryStore("test")
    for i in range(10):
        _cell(store, f"User prefers the color blue variant number {i}",
              "user", "color", f"blue {i}")
    plan = replace(store.compile("What is the user's favorite color?",
                                 T0 + DAY), token_budget=40)
    hits = store.search(plan)
    pack = store.pack(plan, hits)
    assert pack.tokens <= 40
    assert 0 < len(pack.items) < 10


def test_save_load_roundtrip(tmp_path) -> None:
    path = str(tmp_path / "mem.bin")
    store = MemoryStore("test")
    _cell(store, "User lives in Berlin", "user", "location", "Berlin",
          ts=T0, entities=["Berlin"])
    _cell(store, "User lives in Prague", "user", "location", "Prague",
          ts=T0 + 30 * DAY)
    store.save(path)

    loaded = MemoryStore("test")
    loaded.load(path)
    assert loaded.cell_count == store.cell_count == 2
    assert loaded.edge_count == 1
    assert loaded.projection_count == 1
    plan = loaded.compile("Where does the user live?", T0 + 60 * DAY)
    hits = loaded.search(plan)
    assert any("Prague" in h.text for h in hits)
    assert all("Berlin" not in h.text for h in hits)


def test_episode_capture() -> None:
    store = MemoryStore("test")
    eid = store.capture_episode("I live in New York and work at Acme.",
                                observed_at=T0, session_id=7)
    assert eid != 0
    assert store.episode_count == 1


def test_bump_access_heat() -> None:
    store = MemoryStore("test")
    fid = _cell(store, "User prefers dark mode", "user", "preference", "dark")
    store.bump_access(fid)
    store.bump_access(fid)
    hits = store.search(store.compile("dark mode", T0 + DAY))
    assert hits and hits[0].access_heat >= 2