"""Structured extraction — the write path's only LLM use.

The architecture constraint: the write path performs a single-pass extraction
per session (Qwen3 non-thinking mode), then hands compact CellInput records to
the C++ core's deterministic reconcile. Extraction must prefer an empty list
over invented facts and must distinguish user statements, assistant claims,
tool observations, plans, and speculation.

The extractor is model-agnostic: any OpenAI-compatible endpoint works (frontier
APIs, Ollama, llama.cpp, vLLM, LM Studio).
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Protocol

from ..core import WORLD, CellInput, to_ms
from ..eval.protocol import ReaderClient, Session

_KINDS = {"world": WORLD, "preference": 1, "opinion": 2, "experience": 3,
          "procedure": 4}

_SCHEMA = {
    "type": "object",
    "properties": {
        "cells": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "kind": {"type": "string",
                             "enum": ["world", "preference", "opinion",
                                      "experience", "procedure"]},
                    "confidence": {"type": "number", "minimum": 0,
                                   "maximum": 1},
                    "salience": {"type": "number", "minimum": 0, "maximum": 1},
                    "event_date": {"type": "string", "description":
                                   "ISO date of when the statement became "
                                   "true; empty if unknown"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "evidence_span": {"type": "string"},
                },
                "required": ["text"],
            },
        }
    },
    "required": ["cells"],
}

_PROMPT = """Extract durable facts about the user from this conversation for a \
long-term memory system.

<conversation>
{transcript}
</conversation>

Return ONLY a JSON object with a "cells" array. Each cell is one \
self-contained factual statement with:
- "text": one concise, self-contained fact (resolve pronouns, keep names)
- "subject": canonical entity the fact is about (e.g. "user", "acme")
- "predicate": normalized attribute or relation (e.g. "location", "employer", \
"preference", "plan")
- "object": the normalized value (e.g. "Seattle", "Globex")
- "kind": "world" | "preference" | "opinion" | "experience" | "procedure"
- "confidence": 0..1 (opinions only; 1.0 for stated facts)
- "salience": 0..1 how important this is for future answers
- "event_date": ISO date when the fact became true (resolve relative dates \
like "last month" against today {today}); empty if unknown
- "tags": 1-4 short routing tags, lowercase with underscores
- "entities": proper-noun entity names (people, orgs, cities, projects)
- "evidence_span": the exact quoted source phrase supporting the fact

Rules:
- Prefer 3-8 coarse facts over many fine-grained ones.
- Capture EVERY explicit number, date, duration, price, and named entity
  the user states - scores, times, counts and deadlines are the facts most
  often lost, and they are usually the ones asked about later.
- Skip small talk, chit-chat, and procedural noise.
- If the conversation has no durable facts, return {{"cells": []}}.
- NEVER invent a fact. If unsure, omit it.
- Distinguish what the user said from what the assistant suggested.
- Respond with ONLY the JSON object. No thinking, no explanation, no markdown.

Example response shape (do not copy these facts):
{{"cells": [
  {{"text": "The user lives in Seattle.", "subject": "user",
    "predicate": "location", "object": "Seattle", "kind": "world",
    "confidence": 1.0, "salience": 0.8, "event_date": "",
    "tags": ["location"], "entities": ["Seattle"],
    "evidence_span": "I moved to Seattle."}}
]}}
"""

_EXTRACTION_MAX_TOKENS = 1536

# Long sessions must be extracted in windows. A single pass over a long
# transcript loses facts: the model attends to the head and tail and drops
# what sits in the middle (measured: a 12-turn LongMemEval session whose
# verbatim "personal best 27:12" never became a cell, while three
# lower-value facts from the same session did). Windowing is one extra
# bounded call per window, keeps each prompt inside the extraction context,
# and is deterministic - the same session always yields the same windows.
_WINDOW_CHARS = 8000


class Extractor(Protocol):
    """Turn a session into a small set of structured cells."""

    def extract(self, session: Session) -> list[CellInput]:
        ...


def _parse_date(value: str, reference: datetime) -> int:
    """Best-effort ISO / relative date parsing to epoch ms. 0 on failure."""
    if not value:
        return 0
    v = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return to_ms(datetime.strptime(v, fmt))
        except ValueError:
            continue
    lv = v.lower()
    try:
        if "yesterday" in lv:
            return to_ms(reference.replace(hour=0, minute=0, second=0,
                                           microsecond=0)) - 86_400_000
        if "today" in lv or "now" in lv:
            return to_ms(reference)
        m = re.search(r"(\d{1,2})\s+days?\s+ago", lv)
        if m:
            return to_ms(reference) - int(m.group(1)) * 86_400_000
        m = re.search(r"(\d{1,2})\s+weeks?\s+ago", lv)
        if m:
            return to_ms(reference) - int(m.group(1)) * 7 * 86_400_000
        m = re.search(r"(\d{1,2})\s+months?\s+ago", lv)
        if m:
            return to_ms(reference) - int(m.group(1)) * 30 * 86_400_000
    except (ValueError, OverflowError):
        return 0
    return 0


def parse_cells(payload: str, default_ts: int, reference: datetime) -> list[CellInput]:
    """Tolerantly parse the model's JSON into CellInput records."""
    text = payload.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('"cells"')
        if start == -1:
            return []
        brace = text.find("[", start)
        end = text.rfind("]")
        if brace == -1 or end == -1 or end <= brace:
            return []
        try:
            data = json.loads(text[brace:end + 1])
            data = {"cells": data}
        except json.JSONDecodeError:
            return []
    cells = data.get("cells", []) if isinstance(data, dict) else data
    if not isinstance(cells, list):
        return []
    out: list[CellInput] = []
    for item in cells:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        entities = item.get("entities") or []
        tags = item.get("tags") or []
        if not isinstance(entities, list):
            entities = []
        if not isinstance(tags, list):
            tags = []
        entities = [str(e) for e in entities if str(e)]
        tags = [str(t).lower().replace(" ", "_") for t in tags if str(t)]
        kind = _KINDS.get(str(item.get("kind", "")).lower(), WORLD)
        confidence = _clamp_float(item.get("confidence", 1.0), 1.0)
        salience = _clamp_float(item.get("salience", 0.5), 0.5)
        event_ts = _parse_date(str(item.get("event_date", "")), reference)
        out.append(
            CellInput(
                text=str(item["text"]).strip(),
                subject=str(item.get("subject", "") or "").strip(),
                predicate=str(item.get("predicate", "") or "").strip(),
                object=str(item.get("object", "") or "").strip(),
                kind=kind,
                observed_at=default_ts,
                valid_from=event_ts or default_ts,
                confidence=confidence,
                salience=salience,
                source_ref=str(item.get("evidence_span", "") or ""),
                tags=tags,
                entities=entities,
            )
        )
    return out


def _clamp_float(value, default: float) -> float:
    try:
        f = float(value)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return default


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split().__repr__()


class LLMExtractor:
    """Single-pass-per-window structured LLM extraction (model-agnostic).

    One completion per window, temperature 0, strict JSON guidance. The
    returned cells are reconciled and stored by the memory engine.

    A session longer than ``window_chars`` is split on turn boundaries into
    several windows, each extracted independently and merged. The split is
    deterministic and lossless (every turn appears in exactly one window),
    which is what fixes recall on long sessions.
    """

    def __init__(
        self, client: ReaderClient, *, window_chars: int = _WINDOW_CHARS
    ) -> None:
        self._client = client
        self._window_chars = max(1000, window_chars)

    def extract(self, session: Session) -> list[CellInput]:
        default_ts = to_ms(session.timestamp)
        cells: list[CellInput] = []
        for transcript in self._windows(session):
            prompt = _PROMPT.format(
                transcript=transcript,
                today=session.timestamp.date().isoformat(),
            )
            payload = self._client.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=_EXTRACTION_MAX_TOKENS,
                json_mode=True,
            )
            cells.extend(parse_cells(payload, default_ts, session.timestamp))
        return _dedupe_cells(cells)

    def _windows(self, session: Session) -> list[str]:
        """Split the session transcript on turn boundaries.

        Each turn is never split across windows, so no fact is lost at a
        boundary. Windows are packed greedily up to ``window_chars``.
        """
        rendered = [f"{turn.role}: {turn.content}" for turn in session.turns]
        windows: list[str] = []
        current: list[str] = []
        size = 0
        for line in rendered:
            # A single huge turn (a pasted document, a log) still has to be
            # extracted, so it becomes its own window rather than blowing
            # the budget and never being seen.
            if current and size + len(line) > self._window_chars:
                windows.append("\n".join(current))
                current, size = [], 0
            current.append(line)
            size += len(line) + 1
        if current:
            windows.append("\n".join(current))
        return windows or [""]


def _dedupe_cells(cells: list[CellInput]) -> list[CellInput]:
    """Drop repeated cells produced by overlapping windows.

    Order is preserved: the first mention wins, so a fact stated early in
    the session keeps its original (earlier) timestamp.
    """
    seen: set[str] = set()
    out: list[CellInput] = []
    for cell in cells:
        key = _normalize(cell.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(cell)
    return out


class NullExtractor:
    """Deterministic no-LLM extractor for tests, latency baselines, and the
    offline demo.

    Stores each turn verbatim as an experience cell with the session timestamp.
    This is the "raw capture, no distillation" floor: it exercises the full C++
    reconcile/retrieval path deterministically.
    """

    def extract(self, session: Session) -> list[CellInput]:
        ts = to_ms(session.timestamp)
        cells: list[CellInput] = []
        for turn in session.turns:
            content = turn.content.strip()
            if not content:
                continue
            cells.append(
                CellInput(
                    text=content,
                    kind=EXPERIENCE,
                    observed_at=ts,
                    valid_from=ts,
                    salience=0.5,
                )
            )
        return cells


EXPERIENCE = 3  # local alias to keep the NullExtractor dependency-light