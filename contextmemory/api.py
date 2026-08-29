"""Public API of the ContextMemory memory layer.

The API mirrors the surface of the leading memory layers (Supermemory's
add/search/profile, Mem0's add/search) so head-to-head comparisons map
cleanly. It is a thin orchestration layer over the C++ ETMC core + extraction
engine.
"""

from __future__ import annotations

from datetime import datetime

from .core import (
    WORLD,
    CellInput,
    EvidencePack,
    MemoryStore,
    Profile,
    SearchHit,
    StateProjection,
)
from .engine.embedder import Embedder
from .engine.extractor import Extractor, LLMExtractor, NullExtractor
from .engine.memory import (
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_TOP_K,
    IngestReport,
    MemoryEngine,
    RecallReport,
)
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

    def set_extractor(self, extractor: Extractor | None) -> None:
        """Swap the write-path extractor without losing stored memory."""
        self._engine.set_extractor(extractor)

    @property
    def engine(self) -> MemoryEngine:
        return self._engine

    # --- write path --------------------------------------------------------

    def add(
        self,
        content: str,
        *,
        subject: str = "",
        predicate: str = "",
        object: str = "",
        kind: int = WORLD,
        ts: int | datetime = 0,
        ref: str = "",
    ) -> int:
        """Store a single cell directly (no extraction)."""
        ts_ms = _ts_ms(ts)
        return self._engine.store.reconcile(
            CellInput(
                text=content,
                subject=subject,
                predicate=predicate,
                object=object,
                kind=kind,
                observed_at=ts_ms,
                valid_from=ts_ms,
                source_ref=ref,
            )
        )

    def session(self, session: Session) -> IngestReport:
        """Ingest a whole conversation through the extraction pipeline."""
        return self._engine.ingest(session)

    # --- read path ---------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        question_date: datetime | None = None,
        top_k: int = DEFAULT_TOP_K,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> list[SearchHit]:
        report = self._engine.recall(
            query,
            question_date,
            token_budget=token_budget,
            top_k=top_k,
        )
        return report.hits

    def recall(
        self,
        query: str,
        *,
        question_date: datetime | None = None,
        top_k: int = DEFAULT_TOP_K,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> RecallReport:
        return self._engine.recall(
            query,
            question_date,
            token_budget=token_budget,
            top_k=top_k,
        )

    def ask(
        self,
        query: str,
        reader: ReaderClient,
        *,
        question_date: datetime | None = None,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> tuple[str, RecallReport]:
        """Recall evidence, pack it, and generate an answer with a reader."""
        return self._engine.answer(
            query, question_date, reader, token_budget=token_budget
        )

    def profile(self, question_date: datetime | None = None) -> Profile:
        return self._engine.profile(question_date)

    def projection(self, subject: str, predicate: str) -> StateProjection | None:
        return self._engine.projection(subject, predicate)

    # --- persistence -------------------------------------------------------

    def save(self, path: str) -> None:
        self._engine.save(path)

    def load(self, path: str) -> None:
        self._engine.load(path)


def _ts_ms(ts: int | datetime) -> int:
    if isinstance(ts, datetime):
        return int(ts.timestamp() * 1000)
    return ts


__all__ = [
    "CellInput",
    "EvidencePack",
    "IngestReport",
    "LLMExtractor",
    "MemoryClient",
    "MemoryStore",
    "NullExtractor",
    "Profile",
    "RecallReport",
    "SearchHit",
    "Session",
    "StateProjection",
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