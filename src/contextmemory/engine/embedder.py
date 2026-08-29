"""Embedder interface for the memory engine.

Embeddings feed the vector retrieval channel. The interface is model-agnostic
so the rig can use a local CPU-friendly model or an API. The deterministic
hash embedder exists for tests and latency baselines: same text always maps
to the same vector, so retrieval behavior is reproducible without any model.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    """Maps text to dense vectors."""

    @property
    def dim(self) -> int:
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class DeterministicHashEmbedder:
    """Pseudo-random but reproducible embeddings derived from text hashes.

    Not semantically meaningful — only deterministic. Use for integration
    tests and the deterministic latency bench, not for real retrieval quality.
    """

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self._dim = dim
        self._seed = seed

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(f"{self._seed}:{text}".encode()).digest()
        vec = []
        for i in range(self._dim):
            b = digest[i % len(digest)]
            v = ((b / 255.0) - 0.5) * 2.0  # ~[-1, 1]
            vec.append(v)
        # Normalize.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]