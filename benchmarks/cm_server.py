#!/usr/bin/env python3
"""ContextMemory HTTP server for the MemoryBench harness.

Implements the v2 API surface the MemoryBench ``contextmemory`` provider
expects, backed by the ContextMemory engine:

    GET  /v1/health           -> {status: "ok", ...}
    GET  /v1/spaces           -> {spaces: [{id, container_tag}]}
    POST /v1/spaces           -> {id}
    POST /v1/sessions         -> {job_id, session_id}   (async ingest job)
    GET  /v1/jobs/{job_id}    -> {id, status}           (pending/complete/failed)
    POST /v1/search           -> {hits: [...]}

Each question in the harness maps to one ``container_tag`` (an isolated
ContextMemory space). Spaces are created lazily on first use.

Write path: deterministic by default (``NullExtractor`` — every turn stored
verbatim as a cell through the C++ reconcile core, no LLM). Pass
``--extractor llm --model qwen3:4b`` to enable LLM extraction via Ollama
(slow on CPU; ~45s/session).

Read path: deterministic C++ hybrid retrieval + evidence packing. No LLM on
the read path — the harness's answering model handles generation.

stdlib only; no dependencies.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from contextmemory.api import MemoryClient
from contextmemory.engine.embedder import DeterministicHashEmbedder
from contextmemory.engine.extractor import NullExtractor
from contextmemory.eval.protocol import Session, Turn

HOST = "127.0.0.1"
PORT = 8799
MAX_WORKERS = 8

_CLIENTS: dict[str, MemoryClient] = {}
_SPACES: dict[str, str] = {}  # container_tag -> id (id == container_tag)
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_WORKERS: dict[str, str] = {}
_LOGGER_PREFIX = "[cm-server]"


def _log(msg: str) -> None:
    print(f"{_LOGGER_PREFIX} {msg}", flush=True)


def build_client(container_tag: str, *, extractor, journal_dir: str) -> MemoryClient:
    """Build a client for a container tag.

    Must be called while ``_JOBS_LOCK`` is held (the caller manages the lock;
    ``threading.Lock`` is not reentrant).
    """
    if container_tag not in _CLIENTS:
        journal = f"{journal_dir}/{container_tag}.cm.bin" if journal_dir else None
        _CLIENTS[container_tag] = MemoryClient(
            container_tag,
            extractor=extractor,
            embedder=DeterministicHashEmbedder(),
            journal_path=journal,
        )
    return _CLIENTS[container_tag]


def _parse_timestamp(value) -> datetime:
    """Best-effort ISO timestamp; falls back to the ingest time."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now()


def run_job(job_id: str) -> None:
    """Process one ingest job in a worker thread."""
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        space_id = job["space_id"]
        container_tag = job["container_tag"]
        body = job["body"]
        client = job["client"]

    try:
        ts = _parse_timestamp(body.get("timestamp"))
        turns = [
            Turn(role=str(t.get("role", "user")), content=str(t.get("content", "")))
            for t in (body.get("turns") or [])
            if t.get("content")
        ]
        session = Session(
            session_id=str(body.get("external_id") or f"job-{job_id}"),
            timestamp=ts,
            turns=turns,
        )
        report = client.session(session)
        with _JOBS_LOCK:
            job["status"] = "complete"
            job["cells"] = report.new_cells
            job["detail"] = (
                f"{report.new_cells} new / {report.dup_cells} dup "
                f"(reconcile {report.reconcile_ms:.1f}ms)"
            )
    except Exception as exc:  # noqa: BLE001 - job must surface failure
        with _JOBS_LOCK:
            job["status"] = "failed"
            job["detail"] = str(exc)
        _log(f"job {job_id} failed for space {space_id}: {exc}")


def _worker_loop() -> None:
    while True:
        with _JOBS_LOCK:
            pending = [jid for jid, j in _JOBS.items() if j["status"] == "queued"]
            if not pending:
                time.sleep(0.05)
                continue
            job_id = pending[0]
            _JOBS[job_id]["status"] = "running"
        run_job(job_id)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence request spam
        pass

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/v1/health":
            return self._json(200, {
                "status": "ok",
                "service": "contextmemory",
                "version": "2.0.0",
                "spaces": len(_SPACES),
                "cells": sum(c.engine.store.cell_count for c in _CLIENTS.values()),
            })

        if path == "/v1/spaces":
            spaces = [
                {"id": _SPACES[tag], "container_tag": tag}
                for tag in sorted(_SPACES)
            ]
            return self._json(200, {"spaces": spaces})

        if path.startswith("/v1/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
            if not job:
                return self._json(404, {"error": f"job {job_id} not found"})
            return self._json(200, {
                "id": job_id,
                "status": job["status"],
                "detail": job.get("detail", ""),
            })

        return self._json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_body()

        if path == "/v1/spaces":
            tag = str(body.get("container_tag") or body.get("name") or "").strip()
            if not tag:
                return self._json(400, {"error": "container_tag required"})
            with _JOBS_LOCK:
                _SPACES.setdefault(tag, tag)
                build_client(tag, extractor=_EXTRACTOR, journal_dir=_JOURNAL_DIR)
            _log(f"space ready: {tag}")
            return self._json(200, {"id": _SPACES[tag]})

        if path == "/v1/sessions":
            space_id = str(body.get("space_id") or "").strip()
            if space_id not in _SPACES:
                return self._json(404, {"error": f"unknown space {space_id}"})
            job_id = uuid.uuid4().hex
            with _JOBS_LOCK:
                client = _CLIENTS.get(space_id) or build_client(
                    space_id, extractor=_EXTRACTOR, journal_dir=_JOURNAL_DIR
                )
                _JOBS[job_id] = {
                    "id": job_id,
                    "space_id": space_id,
                    "container_tag": space_id,
                    "body": body,
                    "client": client,
                    "status": "queued",
                }
            _log(f"session queued: {body.get('external_id')} -> job {job_id[:8]}")
            return self._json(200, {
                "job_id": job_id,
                "session_id": str(body.get("external_id") or ""),
            })

        if path == "/v1/search":
            space_id = str(body.get("space_id") or "").strip()
            query = str(body.get("query") or "").strip()
            if space_id not in _SPACES:
                return self._json(404, {"error": f"unknown space {space_id}"})
            client = _CLIENTS[space_id]
            top_k = int(body.get("top_k") or 10)
            budget = int(body.get("token_budget") or 2000)
            start = time.time()
            report = client.recall(query, top_k=top_k, token_budget=budget)
            hits = [
                {
                    "id": str(h.cell_id),
                    "cell_id": h.cell_id,
                    "text": h.text,
                    "score": round(float(h.score), 4),
                    "subject": h.subject or "",
                    "predicate": h.predicate or "",
                    "object": h.object or "",
                    "kind": int(h.kind),
                    "status": int(h.status),
                    "salience": float(h.salience),
                    "confidence": float(h.confidence),
                    "valid_from": h.valid_from,
                    "valid_until": h.valid_until,
                    "observed_at": h.valid_from,
                }
                for h in report.hits
            ]
            latency_ms = round((time.time() - start) * 1000, 2)
            return self._json(200, {
                "hits": hits,
                "latency_ms": latency_ms,
                "time_mode": report.time_mode_name,
                "sufficient": report.sufficient,
                "tokens": report.tokens,
            })

        return self._json(404, {"error": "Not Found"})


def main(argv: list[str] | None = None) -> int:
    global _EXTRACTOR, _JOURNAL_DIR
    parser = argparse.ArgumentParser(description="ContextMemory MemoryBench server")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--extractor", choices=["null", "llm"], default="null")
    parser.add_argument("--model", default="qwen3:4b",
                        help="Ollama model for LLM extraction (--extractor llm)")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--journal-dir", default="/tmp/opencode/cm-bench-journals")
    args = parser.parse_args(argv)

    _JOURNAL_DIR = args.journal_dir
    if args.extractor == "llm":
        from contextmemory.engine.extractor import LLMExtractor
        from contextmemory.engine.ollama import OllamaChatClient

        client = OllamaChatClient(args.ollama_url, args.model, timeout=300.0)
        _EXTRACTOR = LLMExtractor(client)
        _log(f"extractor: LLM ({args.model})")
    else:
        _EXTRACTOR = NullExtractor()
        _log("extractor: null (deterministic, no LLM)")

    threading.Thread(target=_worker_loop, daemon=True).start()
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    _log(f"ContextMemory MemoryBench server on http://{HOST}:{args.port}")
    server.serve_forever()
    return 0


_EXTRACTOR = NullExtractor()
_JOURNAL_DIR = ""


if __name__ == "__main__":
    raise SystemExit(main())