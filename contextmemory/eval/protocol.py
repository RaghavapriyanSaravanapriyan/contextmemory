"""Core protocols for the memory evaluation harness.

The harness tests complete memory systems end-to-end: a system must ingest
interactions over time and answer questions after all sessions have been
seen, using nothing but what it stored.

The two protocols here are the seams every adapter implements:

* ``MemorySystem`` -- the system under test (write path + read path).
* ``ReaderClient`` -- the model used for answer generation and extraction.
  Model-agnostic by design: any OpenAI-compatible endpoint works (frontier
  APIs, vLLM, Ollama, LM Studio).
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

import httpx

Message = dict[str, str]


@dataclass(frozen=True)
class Turn:
    """A single user/assistant exchange inside a session."""

    role: str
    content: str
    has_answer: bool | None = None


@dataclass(frozen=True)
class Session:
    """A timestamped conversation session."""

    session_id: str
    timestamp: datetime
    turns: list[Turn] = field(default_factory=list)


@runtime_checkable
class ReaderClient(Protocol):
    """A model client used for answer generation and extraction.

    Implementations must be model-agnostic: the harness should not care
    whether the underlying model is a frontier API model or a local
    open-weight model.
    """

    @abstractmethod
    def complete(self, messages: list[Message], temperature: float = 0.0) -> str:
        """Return the model's completion for a list of chat messages."""
        ...


@runtime_checkable
class MemorySystem(Protocol):
    """The system under test.

    ``ingest`` is the write path (called once per session, in
    chronological order). ``answer`` is the read path: it must answer the
    question using only what was stored, and must be deterministic enough
    to return a meaningful answer at ``question_date``.
    """

    @abstractmethod
    def ingest(self, session: Session) -> None:
        """Memorize a timestamped session."""
        ...

    @abstractmethod
    def answer(self, question: str, question_date: datetime) -> str:
        """Answer a question using stored memory."""
        ...


class OpenAICompatClient:
    """A minimal OpenAI-compatible chat client.

    Works against any ``base_url`` exposing ``/v1/chat/completions``:
    OpenAI, vLLM, Ollama, LM Studio, etc. Kept dependency-light (httpx only)
    and stateless enough to be reused across adapters.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        max_tokens: int | None = None,
    ) -> None:
        self._model = model
        # Accept both roots ("http://localhost:11434") and already-suffixed
        # bases ("http://localhost:11434/v1"); always talk to /v1/chat/completions.
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        self._client = httpx.Client(
            base_url=root,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        self._max_tokens = max_tokens

    def complete(
        self,
        messages: list[Message],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        limit = max_tokens if max_tokens is not None else self._max_tokens
        if limit is not None:
            payload["max_tokens"] = limit
        resp = self._client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        self._client.close()