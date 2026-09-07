"""Lightweight high-performance HTTP & observability server for ContextMemory.

Provides real-time endpoints for the 3D Web Observatory UI and any HTTP
client (agents, apps, curl):

    GET  /v1/health    - System telemetry and engine status
    GET  /v1/graph     - Real-time 3D memory nodes, edges, & clusters from ETMC
    GET  /v1/metrics   - Live cell counts and MEASURED latency (null when none)
    GET  /v1/events    - Stream of real-time memory creation & retrieval events
    GET  /v1/profile   - Static + dynamic profile (?space_id=brain)
    POST /v1/ask       - Memory recall and evidence packing
    POST /v1/recall    - Ranked hits + sufficiency + tokens (agent read path)
    POST /v1/memories  - Store new memory content into persistent journal
    POST /v1/forget    - Forget a memory by id
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

from contextmemory.api import MemoryClient
from contextmemory.core import kind_name, status_name
from contextmemory.eval.protocol import Session, Turn

_CLIENTS: dict[str, MemoryClient] = {}
_EVENT_LOG: list[dict[str, Any]] = []
_EVENT_SEQ = 0
_LOCK = threading.Lock()


def _version() -> str:
    try:
        from contextmemory import __version__

        return __version__
    except Exception:
        return "0.1.0"


def get_client(space_id: str = "brain") -> MemoryClient:
    tag = space_id or "brain"
    with _LOCK:
        if tag not in _CLIENTS:
            from contextmemory.engine.embedder import DeterministicHashEmbedder

            _CLIENTS[tag] = MemoryClient(
                tag, embedder=DeterministicHashEmbedder()
            )
        return _CLIENTS[tag]


def push_event(event_type: str, data: dict[str, Any]) -> None:
    global _EVENT_SEQ
    with _LOCK:
        _EVENT_SEQ += 1
        evt = {
            "id": f"evt_{int(time.time()*1000)}_{_EVENT_SEQ}",
            "type": event_type,
            "timestamp": int(time.time() * 1000),
            "data": data,
        }
        _EVENT_LOG.append(evt)
        if len(_EVENT_LOG) > 1000:
            del _EVENT_LOG[: len(_EVENT_LOG) - 1000]


def _measured_latency() -> dict[str, Any]:
    """p50/p95 over recent measured query_executed events.

    No synthetic constants: when no queries have run yet the fields are null
    (honest "no data") rather than fabricated numbers.
    """
    with _LOCK:
        samples = [
            float(e["data"].get("latency_ms", 0.0))
            for e in _EVENT_LOG
            if e.get("type") == "query_executed"
            and isinstance(e.get("data"), dict)
            and "latency_ms" in e["data"]
        ][-200:]
    if not samples:
        return {"p50_ms": None, "p95_ms": None, "avg_ms": None, "n": 0}
    ordered = sorted(samples)
    n = len(ordered)
    p50 = ordered[min(int(0.50 * n), n - 1)]
    p95 = ordered[min(int(0.95 * n), n - 1)]
    return {
        "p50_ms": round(p50, 3),
        "p95_ms": round(p95, 3),
        "avg_ms": round(sum(samples) / n, 3),
        "n": n,
    }


def build_graph_response(space_id: str = "brain") -> dict[str, Any]:
    client = get_client(space_id)
    profile = client.profile()
    all_hits = profile.static_facts + profile.dynamic_facts

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    clusters: list[dict[str, Any]] = []

    seen_nodes: set[str] = set()
    entity_map: dict[str, list[str]] = {}

    for hit in all_hits:
        nid = str(hit.cell_id)
        if nid in seen_nodes:
            continue
        seen_nodes.add(nid)

        k_name = kind_name(hit.kind)
        s_name = status_name(hit.status)

        node = {
            "id": nid,
            "space_id": space_id,
            "cell_id": hit.cell_id,
            "text": hit.text,
            "subject": hit.subject or "Memory",
            "predicate": hit.predicate or "relates",
            "object": hit.object or "Fact",
            "kind": k_name,
            "status": s_name,
            "confidence": hit.confidence,
            "salience": hit.salience,
            "observed_at": hit.valid_from,
            "valid_from": hit.valid_from,
            "valid_until": hit.valid_until,
            "root_id": str(hit.root_id),
            "parent_id": str(hit.parent_id),
            "tags": list(hit.tags),
            "entities": [hit.subject, hit.object] if hit.subject and hit.object else [],
            "created_at": hit.valid_from,
            "updated_at": hit.valid_from,
        }
        nodes.append(node)

        # Track entities for clustering & edges
        subj = (hit.subject or "").strip()
        if subj:
            entity_map.setdefault(subj, []).append(nid)

    # Generate edges between nodes sharing subjects/entities
    edge_idx = 1
    for ent, member_nids in entity_map.items():
        if len(member_nids) > 1:
            for i in range(len(member_nids) - 1):
                edges.append({
                    "id": f"e_{edge_idx}",
                    "from": member_nids[i],
                    "to": member_nids[i + 1],
                    "type": "entity",
                    "derived": True,
                })
                edge_idx += 1
            clusters.append({
                "id": f"c_{ent.lower()}",
                "type": "entity",
                "label": ent,
                "member_ids": member_nids,
                "count": len(member_nids),
            })

    # Default fallback root node if zero memories exist yet
    if not nodes:
        nodes.append({
            "id": "root_0",
            "space_id": space_id,
            "cell_id": 0,
            "text": (
                "ContextMemory Neural Core Online — Awaiting long-term memory inputs."
            ),
            "subject": "System",
            "predicate": "status",
            "object": "Online",
            "kind": "world",
            "status": "active",
            "confidence": 1.0,
            "salience": 1.0,
            "observed_at": int(time.time() * 1000),
            "valid_from": int(time.time() * 1000),
            "valid_until": 2**62,
            "root_id": "root_0",
            "parent_id": "",
            "tags": ["system", "core"],
            "entities": ["System"],
            "created_at": int(time.time() * 1000),
            "updated_at": int(time.time() * 1000),
        })

    return {
        "space_id": space_id,
        "generation": int(time.time()),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "clusters": clusters,
    }


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class RequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default HTTP console spam

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _json(self, status_code: int, payload: dict[str, Any]) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/v1/health":
            client = get_client("brain")
            return self._json(200, {
                "status": "ok",
                "service": "contextmemory",
                "version": _version(),
                "mode": "local",
                "uptime_ms": int(time.time() * 1000),
                "cell_count": client.engine.store.cell_count,
            })

        if parsed.path == "/v1/graph":
            space_id = qs.get("space_id", ["brain"])[0]
            graph = build_graph_response(space_id)
            return self._json(200, graph)

        if parsed.path == "/v1/metrics":
            space_id = qs.get("space_id", ["brain"])[0]
            client = get_client(space_id)
            store = client.engine.store
            return self._json(200, {
                "health": {
                    "status": "ok",
                    "service": "contextmemory",
                    "version": _version(),
                },
                "stats": {
                    "spaces": 1,
                    "memories": store.cell_count,
                    "active": store.cell_count,
                    "episodes": store.episode_count,
                    "generation": int(time.time()),
                },
                "latency": _measured_latency(),
                "counts": {
                    "active": store.cell_count,
                    "total": store.cell_count,
                    "episodes": store.episode_count,
                    "spaces": 1,
                },
            })

        if parsed.path == "/v1/events":
            with _LOCK:
                return self._json(200, {"events": list(_EVENT_LOG)})

        if parsed.path == "/v1/profile":
            space_id = qs.get("space_id", ["brain"])[0]
            client = get_client(space_id)
            prof = client.profile()
            return self._json(200, {
                "static": [
                    {"id": str(h.cell_id), "text": h.text,
                     "subject": h.subject, "predicate": h.predicate,
                     "confidence": h.confidence}
                    for h in prof.static_facts[:20]
                ],
                "dynamic": [
                    {"id": str(h.cell_id), "text": h.text,
                     "subject": h.subject, "predicate": h.predicate}
                    for h in prof.dynamic_facts[:20]
                ],
            })

        return self._json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            body = {}

        if parsed.path == "/v1/ask":
            query = str(body.get("query", "")).strip()
            space_id = str(body.get("space_id", "brain"))

            start_t = time.time()
            client = get_client(space_id)
            recall_rep = client.recall(query, top_k=6)
            latency_ms = round((time.time() - start_t) * 1000, 2)

            hits = []
            for h in recall_rep.hits:
                hits.append({
                    "id": str(h.cell_id),
                    "text": h.text,
                    "score": h.score,
                    "subject": h.subject,
                    "predicate": h.predicate,
                    "object": h.object,
                    "salience": h.salience,
                    "kind": kind_name(h.kind),
                })

            answer = (
                f"ContextMemory retrieved {len(hits)} connected memories across the "
                f"graph index with {latency_ms}ms latency."
            )

            tokens = recall_rep.pack.tokens if recall_rep.pack else 0
            evt_data = {
                "query": query,
                "hits": hits,
                "latency_ms": latency_ms,
                "tokens": tokens,
            }
            push_event("query_executed", evt_data)

            return self._json(200, {
                "trace_id": f"tr_{int(time.time()*1000)}",
                "query": query,
                "answer": answer,
                "hits": hits,
                "candidates": hits,
                "trace": {"latency_ms": latency_ms},
                "tokens": tokens,
            })

        if parsed.path == "/v1/memories":
            content = str(body.get("content", "")).strip()
            if not content:
                return self._json(400, {"error": "content is required"})
            container = str(body.get("container", "brain"))
            client = get_client(container)

            from datetime import datetime

            session = Session(
                session_id=f"web_{int(time.time())}",
                timestamp=datetime.now(),
                turns=[Turn(role="user", content=content)],
            )
            rep = client.session(session)

            push_event("memory_created", {
                "content": content,
                "cells_added": rep.new_cells,
                "total_cells": client.engine.store.cell_count,
            })

            return self._json(200, {
                "status": "ok",
                "cells_added": rep.new_cells,
                "total_cells": client.engine.store.cell_count,
            })

        if parsed.path == "/v1/recall":
            query = str(body.get("query", "")).strip()
            if not query:
                return self._json(400, {"error": "query is required"})
            space_id = str(body.get("space_id",
                                    body.get("container", "brain")))
            try:
                top_k = max(1, min(32, int(body.get("top_k", 6))))
            except (TypeError, ValueError):
                top_k = 6
            start_t = time.time()
            client = get_client(space_id)
            recall_rep = client.recall(query, top_k=top_k)
            latency_ms = round((time.time() - start_t) * 1000, 2)
            hits = [{
                "id": str(h.cell_id), "text": h.text, "score": h.score,
                "subject": h.subject, "predicate": h.predicate,
                "object": h.object, "kind": kind_name(h.kind),
                "status": status_name(h.status),
            } for h in recall_rep.hits]
            push_event("query_executed", {
                "query": query, "hits": hits, "latency_ms": latency_ms,
            })
            return self._json(200, {
                "query": query, "hits": hits,
                "sufficient": recall_rep.sufficient,
                "tokens": recall_rep.pack.tokens if recall_rep.pack else 0,
                "latency_ms": latency_ms,
            })

        if parsed.path == "/v1/forget":
            try:
                mid = int(body.get("id", 0))
            except (TypeError, ValueError):
                return self._json(400, {"error": "id must be an integer"})
            if mid <= 0:
                return self._json(400, {"error": "id must be positive"})
            space_id = str(body.get("space_id",
                                    body.get("container", "brain")))
            client = get_client(space_id)
            forgotten = client.engine.store.forget(mid)
            if forgotten:
                push_event("memory_forgotten", {"cell_id": mid})
                return self._json(200, {"status": "ok", "forgot": mid})
            return self._json(404, {"error": f"memory {mid} not found"})

        return self._json(404, {"error": "Not Found"})


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_server(port: int = 8765) -> HTTPServer | None:
    if is_port_open(port):
        return None

    server = ThreadedHTTPServer(("127.0.0.1", port), RequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# Re-bind handler class
RequestHandlerRequestHandler = RequestHandler


if __name__ == "__main__":
    print("Starting ContextMemory Observatory Server on http://127.0.0.1:8765")
    server = ThreadedHTTPServer(("127.0.0.1", 8765), RequestHandler)
    server.serve_forever()
