"""Memory engine — ETMC orchestration.

Write path: capture immutable episodes (cheap, no LLM), extract structured
cells (single LLM call per session, pluggable), reconcile deterministically in
the C++ core (dedup / version / project), embed. Read path: compile the query
into a bounded plan (no LLM), search, pack the minimum-sufficient evidence
under a token budget, and optionally generate an answer with a reader model.

Telemetry separates capture / extraction / reconcile / retrieval / packing /
answer so latency and token claims are measurable per stage.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from ..core import (
    CellInput,
    EvidencePack,
    MemoryStore,
    Profile,
    QueryPlan,
    SearchHit,
    StateProjection,
    to_ms,
)
from ..eval.protocol import ReaderClient, Session
from .embedder import Embedder
from .extractor import Extractor, NullExtractor

DEFAULT_TOKEN_BUDGET = 512
DEFAULT_TOP_K = 8


def _asof(question_date: datetime | None) -> int:
    """Question reference time in epoch ms; ``None`` means right now.

    A missing date must not mean the epoch (1970): "current state" queries
    would then evaluate validity windows at the beginning of time and miss
    every cell. The intuitive contract is "as of now".

    Uses naive ``datetime.now()`` to stay consistent with the write path,
    which timestamps sessions with ``datetime.now()`` and lets ``to_ms`` treat
    naive values as UTC. Mixing aware-UTC "now" with naive-UTC stored
    timestamps shifts the reference by the local UTC offset and can push it
    before a just-stored cell's validity window.
    """
    return to_ms(question_date) if question_date is not None else to_ms(
        datetime.now()
    )


@dataclass
class IngestReport:
    episode_id: int = 0
    cells: int = 0
    new_cells: int = 0
    dup_cells: int = 0
    capture_ms: float = 0.0
    extract_ms: float = 0.0
    reconcile_ms: float = 0.0
    embed_ms: float = 0.0
    extract_prompt_tokens: int = 0
    extract_output_tokens: int = 0


@dataclass
class RecallReport:
    plan: QueryPlan = None  # type: ignore[assignment]
    hits: list[SearchHit] = field(default_factory=list)
    pack: EvidencePack = None  # type: ignore[assignment]
    compile_ms: float = 0.0
    search_ms: float = 0.0
    pack_ms: float = 0.0
    embed_ms: float = 0.0

    @property
    def tokens(self) -> int:
        return self.pack.tokens if self.pack else 0

    @property
    def sufficient(self) -> bool:
        return bool(self.pack and self.pack.sufficient)

    @property
    def time_mode_name(self) -> str:
        return _time_name(self.plan.time_mode) if self.plan else "none"


def _time_name(mode: int) -> str:
    return {0: "current", 1: "historical", 2: "interval", 3: "relative",
            4: "none"}.get(mode, "none")


class MemoryEngine:
    """Ties an extractor, embedder, and the C++ ETMC store into one system."""

    def __init__(
        self,
        container_tag: str = "",
        extractor: Extractor | None = None,
        embedder: Embedder | None = None,
        *,
        journal_path: str | Path | None = None,
    ) -> None:
        self._store = MemoryStore(container_tag)
        self._extractor: Extractor = extractor or NullExtractor()
        self._embedder = embedder
        self._cells_ingested = 0
        self._extract_failures = 0
        self._fallback_count = 0
        self._journal: str | None = str(journal_path) if journal_path else None
        if self._journal:
            with contextlib.suppress(OSError, RuntimeError, ValueError):
                self._store.load(self._journal)

    def persist(self) -> None:
        """Write the memory journal to disk (no-op without a journal path)."""
        if self._journal:
            Path(self._journal).parent.mkdir(parents=True, exist_ok=True)
            self._store.save(self._journal)

    @property
    def store(self) -> MemoryStore:
        return self._store

    def set_extractor(self, extractor: Extractor | None) -> None:
        """Swap the write-path extractor, keeping the same store.

        Lets a live session adopt LLM extraction without losing the cells the
        deterministic path already wrote to this engine.
        """
        self._extractor = extractor or NullExtractor()

    @property
    def cells_ingested(self) -> int:
        return self._cells_ingested

    @property
    def extract_failures(self) -> int:
        return self._extract_failures

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    # --- write path ---------------------------------------------------------

    def ingest(self, session: Session) -> IngestReport:
        report = IngestReport()

        transcript = "\n".join(
            f"{turn.role}: {turn.content}" for turn in session.turns
        )
        t0 = time.perf_counter()
        report.episode_id = self._store.capture_episode(
            transcript,
            observed_at=to_ms(session.timestamp),
            session_id=_stable_session_id(session.session_id),
        )
        report.capture_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        try:
            cells = self._extractor.extract(session)
        except Exception:  # noqa: BLE001 - extraction must never kill ingest
            self._extract_failures += 1
            cells = NullExtractor().extract(session)
        report.extract_ms = (time.perf_counter() - t0) * 1000
        report.extract_prompt_tokens = _estimate_tokens(transcript)
        report.extract_output_tokens = _estimate_tokens("".join(c.text for c in cells))

        t0 = time.perf_counter()
        seen: set[int] = set()
        created: list[tuple[CellInput, int]] = []
        for cell in cells:
            cell_id = self._store.reconcile(cell)
            if cell_id in seen:
                report.dup_cells += 1
            else:
                seen.add(cell_id)
                report.new_cells += 1
                created.append((cell, cell_id))
        report.reconcile_ms = (time.perf_counter() - t0) * 1000
        report.cells = len(cells)
        self._cells_ingested += report.new_cells

        if self._embedder is not None and created:
            t0 = time.perf_counter()
            vectors = self._embedder.embed([c.text for c, _ in created])
            for (_, cell_id), vec in zip(created, vectors, strict=True):
                self._store.add_embedding(cell_id, vec)
            report.embed_ms = (time.perf_counter() - t0) * 1000

        with contextlib.suppress(OSError, RuntimeError, ValueError):
            self.persist()

        return report

    # --- read path ----------------------------------------------------------

    def recall(
        self,
        question: str,
        question_date: datetime | None = None,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        top_k: int = DEFAULT_TOP_K,
        embed_query: bool = True,
    ) -> RecallReport:
        at = _asof(question_date)
        report = RecallReport()

        t0 = time.perf_counter()
        plan = replace(
            self._store.compile(question, at),
            candidate_cap=top_k,
            token_budget=token_budget,
        )
        report.compile_ms = (time.perf_counter() - t0) * 1000
        report.plan = plan

        query_vec: list[float] | None = None
        if embed_query and self._embedder is not None:
            t0 = time.perf_counter()
            query_vec = self._embedder.embed([question])[0]
            report.embed_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        report.hits = self._store.search(plan, query_vec)
        report.search_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        report.pack = self._store.pack(plan, report.hits)
        report.pack_ms = (time.perf_counter() - t0) * 1000
        if report.pack.used_fallback:
            self._fallback_count += 1
        return report

    def answer(
        self,
        question: str,
        question_date: datetime | None,
        reader: ReaderClient,
        *,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> tuple[str, RecallReport]:
        report = self.recall(question, question_date, token_budget=token_budget)
        if not report.pack or not report.pack.items:
            return _abstain(question), report
        context = format_evidence(report.pack)
        if not report.pack.sufficient:
            context += (
                "\n\nNote: the retrieved evidence may be insufficient to "
                "answer. If so, say you do not have enough information."
            )
        prompt = (
            "You are a helpful assistant with access to long-term memories "
            "about the user, retrieved from a memory store. Answer the "
            "question using ONLY the memories below. Cite memory ids in "
            "brackets when you use them. If the memories do not contain "
            "enough information to answer, say so explicitly.\n"
            "Answer directly in one or two sentences. No reasoning, no "
            "preamble, no self-talk, no markdown.\n\n"
            f"<memories>\n{context}\n</memories>\n\n"
            f"Question (date: {question_date}): {question}"
        )
        answer = reader.complete([{"role": "user", "content": prompt}],
                                 temperature=0.0)
        return answer, report

    def profile(self, question_date: datetime | None = None) -> Profile:
        at = _asof(question_date)
        return self._store.profile(at_time=at)

    def projection(self, subject: str, predicate: str) -> StateProjection | None:
        return self._store.projection(subject, predicate)

    # --- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        self._store.save(path)

    def load(self, path: str) -> None:
        self._store.load(path)


def format_evidence(pack: EvidencePack) -> str:
    """Token-optimized evidence block with stable IDs and provenance."""
    lines: list[str] = []
    for item in pack.items:
        c = item.cell
        status = _status_label(c.status)
        lines.append(f"[M{c.cell_id} | {status} | {c.subject}/{c.predicate}]")
        lines.append(c.text)
    return "\n".join(lines)


def _abstain(question: str) -> str:
    return (
        "I don't have enough information in memory to answer this question. "
        f"The memories I have do not cover: {question}"
    )


def _status_label(status: int) -> str:
    return {
        0: "current",
        1: "superseded",
        2: "expired",
        3: "forgotten",
        4: "disputed",
    }.get(status, "unknown")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _stable_session_id(session_id: str) -> int:
    # deterministic session id for grouping episodes
    return sum(ord(ch) for ch in session_id) % (2**63)
