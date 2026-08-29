"""Memory engine — orchestrates the write and read paths.

Write path: extract coarse facts from a session (single LLM call, pluggable),
embed them, and apply them to the C++ store. Read path: deterministic hybrid
retrieval (no LLM) plus optional answer generation with a reader model.
"""

from __future__ import annotations

from datetime import datetime

from ..core import MemoryStore, SearchHit, to_ms
from ..eval.protocol import ReaderClient, Session
from .embedder import Embedder
from .extractor import Extractor, NullExtractor

DEFAULT_TOKEN_BUDGET = 700
DEFAULT_TOP_K = 15


def assemble_context(hits: list[SearchHit], token_budget: int) -> str:
    """Format retrieved facts for injection, honoring the token budget."""
    lines: list[str] = []
    used = 0
    for hit in hits:
        estimated = len(hit.text) // 4 + 1
        if lines and used + estimated > token_budget:
            break
        kind = {0: "world", 1: "opinion", 2: "preference", 3: "episode"}.get(
            hit.kind, "fact"
        )
        lines.append(f"- [{kind}] {hit.text}")
        used += estimated
    return "\n".join(lines)


class MemoryEngine:
    """Ties an extractor, embedder, and the C++ store into one system."""

    def __init__(
        self,
        container_tag: str = "",
        extractor: Extractor | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._store = MemoryStore(container_tag)
        self._extractor: Extractor = extractor or NullExtractor()
        self._embedder = embedder
        self._facts_ingested = 0

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def facts_ingested(self) -> int:
        return self._facts_ingested

    # --- write path --------------------------------------------------------

    def ingest(self, session: Session) -> int:
        """Extract, embed, and store a session's facts. Returns fact count."""
        facts = self._extractor.extract(session)
        if not facts:
            return 0
        texts = [f.text for f in facts]
        vectors = self._embedder.embed(texts) if self._embedder else [None] * len(facts)
        default_ts = to_ms(session.timestamp)
        for fact, vec in zip(facts, vectors, strict=True):
            fact_id = self._store.add_fact(
                fact.text,
                kind=fact.kind,
                is_static=fact.is_static,
                confidence=fact.confidence,
                ts=fact.ts or default_ts,
                ref=session.session_id,
                entities=fact.entities,
            )
            if vec is not None:
                self._store.add_embedding(fact_id, vec)
        self._facts_ingested += len(facts)
        return len(facts)

    # --- read path ---------------------------------------------------------

    def recall(
        self,
        question: str,
        question_date: datetime | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        include_expired: bool = False,
        expand_depth: int = 0,
    ) -> list[SearchHit]:
        at_time = to_ms(question_date) if question_date is not None else 0
        query_vec = None
        if self._embedder is not None:
            query_vec = self._embedder.embed([question])[0]
        return self._store.search(
            question,
            query_vec=query_vec,
            at_time=at_time,
            top_k=top_k,
            token_budget=token_budget,
            include_expired=include_expired,
            expand_depth=expand_depth,
        )

    def answer(
        self,
        question: str,
        question_date: datetime | None,
        reader: ReaderClient,
    ) -> str:
        hits = self.recall(question, question_date)
        context = assemble_context(hits, DEFAULT_TOKEN_BUDGET)
        prompt = (
            "You are a helpful assistant with access to long-term memories "
            "about the user, retrieved from a memory store. Answer the "
            "question using only the memories below. If the memories do not "
            "contain enough information to answer, say so explicitly.\n\n"
            f"<memories>\n{context}\n</memories>\n\n"
            f"Question (date: {question_date}): {question}"
        )
        return reader.complete([{"role": "user", "content": prompt}], temperature=0.0)

    def profile(self, question_date: datetime | None = None) -> object:
        at_time = to_ms(question_date) if question_date is not None else 0
        return self._store.profile(at_time=at_time)