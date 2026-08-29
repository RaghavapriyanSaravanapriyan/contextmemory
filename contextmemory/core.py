"""Python wrapper around the C++ core engine.

The C++ core (``contextmemory._core``) is the entire memory engine: the
bi-temporal store, BM25 + vector + entity indexes, hybrid fusion retrieval,
and snapshot persistence. This module gives it a typed, ergonomic Python
surface and keeps the fact-kind / edge-type constants in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ._core import Store as _Store

# Fact kinds (must match cmcore::FactKind).
WORLD = 0
OPINION = 1
PREFERENCE = 2
EPISODE = 3

# Edge types (must match cmcore::EdgeType).
UPDATES = 0
EXTENDS = 1
DERIVES = 2
RELATED = 3
CAUSAL = 4


def to_ms(dt: datetime) -> int:
    """Convert a datetime to unix-epoch milliseconds (the core's timestamp)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


@dataclass(frozen=True)
class SearchHit:
    """A single retrieved fact."""

    fact_id: int
    text: str
    score: float
    kind: int
    is_static: bool
    confidence: float
    valid_from: int
    invalid_at: int
    source_ref: str
    root_id: int


@dataclass(frozen=True)
class Profile:
    """Static (durable) and dynamic (recent) memory profile."""

    static_facts: list[SearchHit]
    dynamic_facts: list[SearchHit]


class MemoryStore:
    """Typed facade over the C++ core store for a single container tag."""

    def __init__(self, container_tag: str = "") -> None:
        self._store = _Store(container_tag)
        self._container_tag = container_tag

    @property
    def container_tag(self) -> str:
        return self._container_tag

    # --- write path --------------------------------------------------------

    def add_fact(
        self,
        text: str,
        *,
        kind: int = WORLD,
        is_static: bool = False,
        confidence: float = 1.0,
        ts: int = 0,
        ref: str = "",
        entities: list[str] | None = None,
    ) -> int:
        """Store a fact; returns its id."""
        return self._store.add_fact(
            text,
            kind=kind,
            is_static=is_static,
            confidence=confidence,
            ts=ts,
            ref=ref,
            entities=list(entities or []),
        )

    def update_fact(
        self,
        fact_id: int,
        text: str,
        ts: int,
        entities: list[str] | None = None,
    ) -> int:
        """Version a fact (supersedes it); returns the new fact id."""
        return self._store.update_fact(fact_id, text, ts, list(entities or []))

    def link(self, edge_type: int, from_id: int, to_id: int, ts: int = 0) -> None:
        self._store.link(edge_type, from_id, to_id, ts)

    def expire(self, fact_id: int, ts: int) -> None:
        """Close a fact's validity window (it is no longer true)."""
        self._store.expire(fact_id, ts)

    def forget(self, fact_id: int, ts: int) -> None:
        """Transactionally remove a fact from retrieval."""
        self._store.forget(fact_id, ts)

    def set_confidence(self, fact_id: int, value: float) -> None:
        self._store.set_confidence(fact_id, value)

    def add_embedding(self, fact_id: int, vector: list[float]) -> None:
        self._store.add_embedding(fact_id, vector)

    # --- read path ---------------------------------------------------------

    def search(
        self,
        text: str,
        *,
        query_vec: list[float] | None = None,
        query_entities: list[str] | None = None,
        at_time: int = 0,
        top_k: int = 15,
        token_budget: int = 700,
        include_expired: bool = False,
        expand_depth: int = 0,
    ) -> list[SearchHit]:
        results = self._store.search(
            text,
            list(query_vec or []),
            list(query_entities or []),
            at_time=at_time,
            top_k=top_k,
            token_budget=token_budget,
            include_expired=include_expired,
            expand_depth=expand_depth,
        )
        return [SearchHit(**r) for r in results]

    def profile(self, at_time: int = 0, top_k: int = 20) -> Profile:
        p = self._store.profile(at_time=at_time, top_k=top_k)
        return Profile(
            static_facts=[SearchHit(**r) for r in p["static_facts"]],
            dynamic_facts=[SearchHit(**r) for r in p["dynamic_facts"]],
        )

    # --- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        self._store.save(path)

    def load(self, path: str) -> None:
        self._store.load(path)

    # --- introspection -----------------------------------------------------

    @property
    def fact_count(self) -> int:
        return self._store.fact_count

    @property
    def edge_count(self) -> int:
        return self._store.edge_count

    @property
    def entity_count(self) -> int:
        return self._store.entity_count