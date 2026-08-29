"""Python facade over the C++ ETMC core engine.

The C++ core (``contextmemory._core``) is the entire memory engine: immutable
episodes, bi-temporal memory cells, state projections, the deterministic query
compiler, hybrid retrieval, minimum-sufficient evidence packing, and snapshot
persistence. This module gives it a typed, ergonomic Python surface and keeps
the CellKind / CellStatus / TimeMode / RelationMode constants in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ._core import Store as _Store

# Cell kinds (must match cmcore::CellKind).
WORLD = 0
PREFERENCE = 1
OPINION = 2
EXPERIENCE = 3
PROCEDURE = 4

# Cell statuses (must match cmcore::CellStatus).
ACTIVE = 0
SUPERSEDED = 1
EXPIRED = 2
FORGOTTEN = 3
DISPUTED = 4

# Edge types (must match cmcore::EdgeType).
UPDATES = 0
EXTENDS = 1
DERIVES = 2
CAUSAL = 3
RELATED = 4

# Time modes (must match cmcore::TimeMode).
T_CURRENT = 0
T_HISTORICAL = 1
T_INTERVAL = 2
T_RELATIVE = 3
T_NONE = 4

# Relation modes (must match cmcore::RelationMode).
R_DIRECT = 0
R_MULTIHOP = 1
R_CAUSAL = 2
R_NONE = 3

_KIND_NAMES = {
    WORLD: "world",
    PREFERENCE: "preference",
    OPINION: "opinion",
    EXPERIENCE: "experience",
    PROCEDURE: "procedure",
}
_STATUS_NAMES = {
    ACTIVE: "active",
    SUPERSEDED: "superseded",
    EXPIRED: "expired",
    FORGOTTEN: "forgotten",
    DISPUTED: "disputed",
}
_TIME_NAMES = {
    T_CURRENT: "current",
    T_HISTORICAL: "historical",
    T_INTERVAL: "interval",
    T_RELATIVE: "relative",
    T_NONE: "none",
}


def to_ms(dt: datetime) -> int:
    """Convert a datetime to unix-epoch milliseconds (the core's timestamp)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def kind_name(kind: int) -> str:
    return _KIND_NAMES.get(kind, "fact")


def status_name(status: int) -> str:
    return _STATUS_NAMES.get(status, "unknown")


@dataclass(frozen=True)
class CellInput:
    """A compact cell produced by extraction, ready for deterministic reconcile."""

    text: str
    subject: str = ""
    predicate: str = ""
    object: str = ""
    kind: int = WORLD
    observed_at: int = 0  # system time (ms)
    valid_from: int = 0   # event time (ms)
    confidence: float = 1.0
    salience: float = 0.5
    source_ref: str = ""
    source_begin: int = 0
    source_end: int = 0
    tags: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SearchHit:
    """A single retrieved memory cell."""

    cell_id: int
    text: str
    subject: str
    predicate: str
    object: str
    score: float
    kind: int
    status: int
    confidence: float
    salience: float
    access_heat: int
    valid_from: int
    valid_until: int
    root_id: int
    parent_id: int
    source_ref: str
    tags: list[str]
    projection_hit: bool

    @property
    def is_current(self) -> bool:
        return self.status == ACTIVE


@dataclass(frozen=True)
class QueryPlan:
    """A compiled, bounded retrieval plan."""

    text: str
    time_mode: int = T_NONE
    time_start: int = 0
    time_end: int = 0
    entity_seeds: list[str] = field(default_factory=list)
    relation_mode: int = R_NONE
    kind_mask: int = 0xFFFFFFFF
    tags: list[str] = field(default_factory=list)
    candidate_cap: int = 16
    expansion_cap: int = 1
    token_budget: int = 512
    fallback: bool = True
    subject_hint: str = ""
    predicate_hint: str = ""
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceItem:
    cell: SearchHit
    covers_current: bool = False
    covers_historical: bool = False
    covers_relation: bool = False


@dataclass(frozen=True)
class EvidencePack:
    """Minimum-sufficient evidence selected under a token budget."""

    items: list[EvidenceItem]
    tokens: int
    budget: int
    sufficient: bool
    used_fallback: bool = False


@dataclass(frozen=True)
class Profile:
    """Static (durable) and dynamic (recent) memory profile."""

    static_facts: list[SearchHit]
    dynamic_facts: list[SearchHit]


@dataclass(frozen=True)
class StateProjection:
    """The current answer for (subject, predicate)."""

    subject: str
    predicate: str
    active_cell: int
    root_id: int
    version_count: int
    updated_at: int


def _hit_from_dict(d: dict[str, Any]) -> SearchHit:
    return SearchHit(
        cell_id=int(d["cell_id"]),
        text=d["text"],
        subject=d.get("subject", ""),
        predicate=d.get("predicate", ""),
        object=d.get("object", ""),
        score=float(d["score"]),
        kind=int(d["kind"]),
        status=int(d["status"]),
        confidence=float(d["confidence"]),
        salience=float(d["salience"]),
        access_heat=int(d.get("access_heat", 0)),
        valid_from=int(d["valid_from"]),
        valid_until=int(d["valid_until"]),
        root_id=int(d["root_id"]),
        parent_id=int(d["parent_id"]),
        source_ref=d.get("source_ref", ""),
        tags=list(d.get("tags", [])),
        projection_hit=bool(d.get("projection_hit", False)),
    )


def _hit_to_dict(h: SearchHit) -> dict[str, Any]:
    return {
        "cell_id": h.cell_id,
        "text": h.text,
        "subject": h.subject,
        "predicate": h.predicate,
        "object": h.object,
        "score": h.score,
        "kind": h.kind,
        "status": h.status,
        "confidence": h.confidence,
        "salience": h.salience,
        "access_heat": h.access_heat,
        "valid_from": h.valid_from,
        "valid_until": h.valid_until,
        "root_id": h.root_id,
        "parent_id": h.parent_id,
        "source_ref": h.source_ref,
        "tags": list(h.tags),
        "projection_hit": h.projection_hit,
    }


class MemoryStore:
    """Typed facade over the C++ ETMC core for a single container tag."""

    def __init__(self, container_tag: str = "") -> None:
        self._store = _Store(container_tag)
        self._container_tag = container_tag

    @property
    def container_tag(self) -> str:
        return self._container_tag

    # --- capture (immutable evidence, no LLM) -------------------------------

    def capture_episode(self, content: str, *, role: str = "user",
                        observed_at: int = 0, session_id: int = 0) -> int:
        return self._store.capture_episode(role, content, observed_at, session_id)

    # --- reconcile (deterministic first) ------------------------------------

    def reconcile(self, inp: CellInput) -> int:
        return self._store.reconcile(
            inp.subject,
            inp.predicate,
            inp.object,
            inp.text,
            inp.kind,
            inp.observed_at,
            inp.valid_from,
            inp.confidence,
            inp.salience,
            inp.source_ref,
            inp.source_begin,
            inp.source_end,
            list(inp.tags),
            list(inp.entities),
        )

    # --- projections --------------------------------------------------------

    def projection(self, subject: str, predicate: str) -> StateProjection | None:
        d = self._store.projection(subject, predicate)
        if d is None:
            return None
        return StateProjection(
            subject=d["subject"],
            predicate=d["predicate"],
            active_cell=int(d["active_cell"]),
            root_id=int(d["root_id"]),
            version_count=int(d["version_count"]),
            updated_at=int(d["updated_at"]),
        )

    def bump_access(self, cell_id: int) -> None:
        self._store.bump_access(cell_id)

    def add_embedding(self, cell_id: int, vector: list[float]) -> None:
        self._store.add_embedding(cell_id, vector)

    # --- query compilation --------------------------------------------------

    def compile(self, question: str, at_time: int = 0) -> QueryPlan:
        d = self._store.compile(question, at_time)
        return QueryPlan(
            text=d["text"],
            time_mode=int(d["time_mode"]),
            time_start=int(d["time_start"]),
            time_end=int(d["time_end"]),
            entity_seeds=list(d["entity_seeds"]),
            relation_mode=int(d["relation_mode"]),
            kind_mask=int(d["kind_mask"]),
            tags=list(d["tags"]),
            candidate_cap=int(d["candidate_cap"]),
            expansion_cap=int(d["expansion_cap"]),
            token_budget=int(d["token_budget"]),
            fallback=bool(d["fallback"]),
            subject_hint=d["subject_hint"],
            predicate_hint=d["predicate_hint"],
            trace=dict(d["trace"]),
        )

    # --- read path ----------------------------------------------------------

    def search(
        self,
        plan: QueryPlan,
        query_vec: list[float] | None = None,
    ) -> list[SearchHit]:
        raw = self._store.search(
            plan.text,
            plan.time_end,
            plan.time_mode,
            plan.time_start,
            plan.time_end,
            list(plan.entity_seeds),
            plan.relation_mode,
            plan.kind_mask,
            list(plan.tags),
            plan.candidate_cap,
            plan.expansion_cap,
            plan.token_budget,
            plan.fallback,
            plan.subject_hint,
            plan.predicate_hint,
            list(query_vec or []),
        )
        return [_hit_from_dict(d) for d in raw]

    def pack(self, plan: QueryPlan, hits: list[SearchHit]) -> EvidencePack:
        hit_dicts = [_hit_to_dict(h) for h in hits]
        d = self._store.pack_hits(
            hit_dicts,
            plan.time_end,
            plan.time_mode,
            plan.time_start,
            plan.time_end,
            plan.candidate_cap,
            plan.expansion_cap,
            plan.token_budget,
            plan.relation_mode,
        )
        items = []
        for it in d["items"]:
            items.append(
                EvidenceItem(
                    cell=_hit_from_dict(it["cell"]),
                    covers_current=bool(it["covers_current"]),
                    covers_historical=bool(it["covers_historical"]),
                    covers_relation=bool(it["covers_relation"]),
                )
            )
        return EvidencePack(
            items=items,
            tokens=int(d["tokens"]),
            budget=int(d["budget"]),
            sufficient=bool(d["sufficient"]),
            used_fallback=bool(d["used_fallback"]),
        )

    def profile(self, at_time: int = 0, top_k: int = 20) -> Profile:
        p = self._store.profile(at_time, top_k)
        return Profile(
            static_facts=[_hit_from_dict(r) for r in p["static_facts"]],
            dynamic_facts=[_hit_from_dict(r) for r in p["dynamic_facts"]],
        )

    # --- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        self._store.save(path)

    def load(self, path: str) -> None:
        self._store.load(path)

    # --- introspection ------------------------------------------------------

    @property
    def cell_count(self) -> int:
        return self._store.cell_count

    @property
    def edge_count(self) -> int:
        return self._store.edge_count

    @property
    def entity_count(self) -> int:
        return self._store.entity_count

    @property
    def episode_count(self) -> int:
        return self._store.episode_count

    @property
    def projection_count(self) -> int:
        return self._store.projection_count