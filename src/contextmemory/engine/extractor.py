"""Extraction layer — the write path's only LLM use.

The design constraint from the architecture decision: the write path performs
a single-pass LLM extraction per session (Mem0/Hindsight style: coarse
facts, no per-entity/per-edge calls), then hands structured ops to the C++
core. The extractor is model-agnostic: any OpenAI-compatible endpoint works
(frontier APIs, vLLM, Ollama, LM Studio).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..core import EPISODE, WORLD, to_ms
from ..eval.protocol import ReaderClient, Session

_FACT_JSON_HINT = (
    "Return ONLY a JSON array of fact objects. Each object has fields:\n"
    '- "text": string, one concise durable fact\n'
    '- "kind": "world" | "opinion" | "preference" | "episode"\n'
    '- "is_static": boolean (stable long-term facts only)\n'
    '- "confidence": number 0..1 (opinions only; else 1.0)\n'
    '- "entities": array of proper-noun entity names (people, orgs, projects)\n'
    "Rules: prefer few coarse facts over many fine-grained ones; skip small "
    "talk and procedural noise; keep verb tense and names as stated."
)

_KINDS = {"world": 0, "opinion": 1, "preference": 2, "episode": 3}


@dataclass(frozen=True)
class ExtractedFact:
    """A durable fact produced by the extraction layer."""

    text: str
    kind: int = WORLD
    is_static: bool = False
    confidence: float = 1.0
    ts: int = 0  # event timestamp (ms); 0 = session timestamp
    entities: list[str] = field(default_factory=list)


class Extractor(Protocol):
    """Turn a session into a small set of durable facts."""

    def extract(self, session: Session) -> list[ExtractedFact]:
        ...


def parse_facts(payload: str, default_ts: int) -> list[ExtractedFact]:
    """Parse the model's JSON array into facts, tolerantly."""
    text = payload.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(data, list):
        return []
    facts: list[ExtractedFact] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        kind = _KINDS.get(str(item.get("kind", "")).lower(), WORLD)
        entities = item.get("entities") or []
        if not isinstance(entities, list):
            entities = []
        entities = [str(e) for e in entities if str(e)]
        confidence = item.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        facts.append(
            ExtractedFact(
                text=str(item["text"]),
                kind=kind,
                is_static=bool(item.get("is_static", False)),
                confidence=max(0.0, min(1.0, confidence)),
                ts=default_ts,
                entities=entities,
            )
        )
    return facts


class LLMExtractor:
    """Single-pass LLM fact extraction (model-agnostic).

    One completion per session, deterministic (temperature 0). The returned
    facts are stored and embedded by the memory engine.
    """

    def __init__(self, client: ReaderClient) -> None:
        self._client = client

    def extract(self, session: Session) -> list[ExtractedFact]:
        default_ts = to_ms(session.timestamp)
        transcript = "\n".join(
            f"{turn.role}: {turn.content}" for turn in session.turns
        )
        prompt = (
            "Extract durable facts about the user from this conversation for "
            "a long-term memory system.\n\n"
            f"<conversation>\n{transcript}\n</conversation>\n\n"
            f"{_FACT_JSON_HINT}"
        )
        payload = self._client.complete(
            [{"role": "user", "content": prompt}], temperature=0.0
        )
        return parse_facts(payload, default_ts)


class NullExtractor:
    """Deterministic no-LLM extractor for tests and latency baselines.

    Stores each turn verbatim as an episodic fact. This is the "raw capture,
    no distillation" floor of the write path; it keeps the harness runnable
    without a model while still exercising the full C++ store.
    """

    def extract(self, session: Session) -> list[ExtractedFact]:
        ts = to_ms(session.timestamp)
        facts = []
        for turn in session.turns:
            if turn.content.strip():
                facts.append(
                    ExtractedFact(
                        text=turn.content,
                        kind=EPISODE,
                        is_static=False,
                        confidence=1.0,
                        ts=ts,
                    )
                )
        return facts