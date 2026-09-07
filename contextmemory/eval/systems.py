"""Reference memory systems for the evaluation harness.

These are baselines, not the production system. They define the floor:
* ``FullHistorySystem`` -- upper bound on information (everything in
  context), the strongest a context-window agent can do.
* ``RecencyWindowSystem`` -- the common naive approach: keep the most
  recent N sessions and nothing else.
* ``CoreMemorySystem`` -- the production memory layer: extraction write
  path (pluggable) + deterministic C++ store read path.
"""

from __future__ import annotations

from datetime import datetime

from .protocol import MemorySystem, ReaderClient, Session


class FullHistorySystem(MemorySystem):
    """Retains every turn and dumps the full history into the prompt."""

    def __init__(self, reader: ReaderClient) -> None:
        self._reader = reader
        self._sessions: list[Session] = []

    def ingest(self, session: Session) -> None:
        self._sessions.append(session)

    def answer(self, question: str, question_date: datetime) -> str:
        parts = []
        for session in self._sessions:
            parts.append(f"[session {session.session_id} @ {session.timestamp}]")
            for turn in session.turns:
                parts.append(f"{turn.role}: {turn.content}")
        history = "\n".join(parts)
        prompt = (
            "You are a helpful assistant with access to the full transcript of "
            "past conversations with a user. Answer the user's question using "
            "only the transcript. If the transcript does not contain enough "
            "information to answer, say so explicitly.\n\n"
            f"<transcript>\n{history}\n</transcript>\n\n"
            f"Question (date: {question_date}): {question}"
        )
        return self._reader.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )


class RecencyWindowSystem(MemorySystem):
    """Keeps only the most recent ``window`` sessions.

    Models a chat assistant that only sees the recent conversation history,
    i.e. a system with no real long-term memory.
    """

    def __init__(self, reader: ReaderClient, window: int) -> None:
        self._reader = reader
        self._window = window
        self._sessions: list[Session] = []

    def ingest(self, session: Session) -> None:
        self._sessions.append(session)
        if len(self._sessions) > self._window:
            self._sessions = self._sessions[-self._window :]

    def answer(self, question: str, question_date: datetime) -> str:
        parts = []
        for session in self._sessions:
            parts.append(f"[session {session.session_id} @ {session.timestamp}]")
            for turn in session.turns:
                parts.append(f"{turn.role}: {turn.content}")
        history = "\n".join(parts)
        prompt = (
            "You are a helpful assistant. Answer the user's question using "
            "the recent conversation transcript below. If the transcript does "
            "not contain enough information to answer, say so explicitly.\n\n"
            f"<transcript>\n{history}\n</transcript>\n\n"
            f"Question (date: {question_date}): {question}"
        )
        return self._reader.complete(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
        )


class RecencyCharsSystem(MemorySystem):
    """Keeps only the most recent ``max_chars`` of transcript.

    The long-scale baseline: a 100K-token conversation cannot fit a small
    local reader, so the honest local analogue of "recent context" is a
    trailing character window. Same reader model as every other system.
    """

    def __init__(self, reader: ReaderClient, max_chars: int = 8000) -> None:
        self._reader = reader
        self._max_chars = max_chars
        self._chunks: list[str] = []

    def ingest(self, session: Session) -> None:
        text = "\n".join(f"{t.role}: {t.content}" for t in session.turns)
        self._chunks.append(text)

    def answer(self, question: str, question_date: datetime) -> str:
        buf: list[str] = []
        total = 0
        for chunk in reversed(self._chunks):
            buf.append(chunk)
            total += len(chunk)
            if total >= self._max_chars:
                break
        history = "\n".join(reversed(buf))[-self._max_chars :]
        prompt = (
            "Answer using only the transcript below. If it lacks the "
            f"information, say so.\n\n<transcript>\n{history}\n</transcript>\n\n"
            f"Question: {question}"
        )
        return self._reader.complete(
            [{"role": "user", "content": prompt}], temperature=0.0
        )


class CoreMemorySystem(MemorySystem):
    """The production memory layer as an eval harness system.

    Write path: extract coarse facts per session (``extractor``; default
    ``NullExtractor`` for deterministic runs), embed, store in the C++ core.
    Read path: deterministic hybrid retrieval + reader answer generation.
    """

    def __init__(
        self,
        reader: ReaderClient,
        *,
        extractor=None,
        embedder=None,
        container_tag: str = "eval",
    ) -> None:
        from ..engine.extractor import NullExtractor
        from ..engine.memory import MemoryEngine

        self._reader = reader
        self._engine = MemoryEngine(
            container_tag=container_tag,
            extractor=extractor or NullExtractor(),
            embedder=embedder,
        )

    def ingest(self, session: Session) -> None:
        self._engine.ingest(session)

    def answer(self, question: str, question_date: datetime) -> str:
        answer, _report = self._engine.answer(question, question_date,
                                              self._reader)
        return answer