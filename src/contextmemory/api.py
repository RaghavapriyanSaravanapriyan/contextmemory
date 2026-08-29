"""Public API of the ContextMemory memory layer.

The API mirrors the surface of the leading memory layers (Supermemory's
add/search/profile, Mem0's add/search) so head-to-head comparisons map
cleanly. It is a thin orchestration layer over the C++ core + extraction
engine.
"""

from __future__ import annotations

from datetime import datetime

from .core import MemoryStore, Profile, SearchHit
from .engine.embedder import Embedder
from .engine.extractor import Extractor, LLMExtractor, NullExtractor
from .engine.memory import DEFAULT_TOKEN_BUDGET, DEFAULT_TOP_K, MemoryEngine
from .eval.protocol import ReaderClient, Session


class MemoryClient:
    """The public memory layer API for one container tag."""

    def __init__(
        self,
        container_tag: str = "",
        *,
        extractor: Extractor | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._engine = MemoryEngine(
            container_tag=container_tag,
            extractor=extractor or NullExtractor(),
            embedder=embedder,
        )

    @property
    def container_tag(self) -> str:
        return self._engine.store.container_tag

    # --- write path --------------------------------------------------------

    def add(self, content: str, *, ts: int | datetime = 0, ref: str = "") -> int:
        """Store a single fact directly (no extraction)."""
        ts_ms = _ts_ms(ts)
        return self._engine.store.add_fact(
            content, kind=0, is_static=False, confidence=1.0, ts=ts_ms, ref=ref
        )

    def session(self, session: Session) -> int:
        """Ingest a whole conversation through the extraction pipeline."""
        return self._engine.ingest(session)

    def forget(self, fact_id: int, ts: int | datetime = 0) -> None:
        self._engine.store.forget(fact_id, _ts_ms(ts))

    # --- read path ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        question_date: datetime | None = None,
        top_k: int = DEFAULT_TOP_K,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> list[SearchHit]:
        return self._engine.recall(
            query,
            question_date,
            top_k=top_k,
            token_budget=token_budget,
        )

    def profile(self, question_date: datetime | None = None) -> Profile:
        return self._engine.profile(question_date)

    # --- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        self._engine.store.save(path)

    def load(self, path: str) -> None:
        self._engine.store.load(path)


def _ts_ms(ts: int | datetime) -> int:
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    return ts


__all__ = [
    "LLMExtractor",
    "MemoryClient",
    "MemoryStore",
    "NullExtractor",
    "Profile",
    "SearchHit",
    "Session",
    "from_reader_client",
]


def from_reader_client(
    client: ReaderClient,
    container_tag: str = "",
    *,
    extractor: Extractor | None = None,
    embedder: Embedder | None = None,
) -> MemoryClient:
    """Build a client whose write path uses an LLM-backed extractor."""
    return MemoryClient(
        container_tag,
        extractor=extractor or LLMExtractor(client),
        embedder=embedder,
    )