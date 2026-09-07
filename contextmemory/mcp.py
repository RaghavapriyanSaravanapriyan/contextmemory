"""MCP (Model Context Protocol) server for ContextMemory.

Exposes the memory engine to any MCP client (Claude Code, Cursor, OpenCode,
VS Code, Cline, ...) over stdio:

    memory    save meaningful information (extraction path)
    recall    retrieve relevant memories for a query
    context   retrieve session context
    profile   static (durable) + dynamic (recent) user profile, ~50ms
    timeline  version history for a subject/predicate (updates, staleness)
    forget    remove a memory by id

This is a self-contained stdio JSON-RPC server with no external MCP
dependency, so the bridge works offline and installs nothing extra. Tool
arguments map directly onto the ContextMemory API.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from typing import Any

from .api import MemoryClient
from .engine.embedder import DeterministicHashEmbedder
from .eval.protocol import Session, Turn

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "contextmemory", "version": "0.1.0"}

_TOOLS = [
    {
        "name": "memory",
        "description": "Save meaningful information to long-term memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string",
                            "description": "The information to remember"},
                "container": {"type": "string",
                              "description": "Scope/namespace (default 'brain')"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "recall",
        "description": "Retrieve the most relevant memories for a query.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "container": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "context",
        "description": "Retrieve relevant context for the current session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string",
                          "description": "What context is needed"},
                "container": {"type": "string"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "forget",
        "description": "Remove a memory by its id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "Memory id to forget"},
                "container": {"type": "string"},
            },
            "required": ["id"],
        },
    },
    {
        "name": "profile",
        "description": (
            "Get the user's memory profile: durable static facts plus "
            "recent dynamic activity. One call, milliseconds, always fresh."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "container": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "timeline",
        "description": (
            "Show how one fact evolved: current value plus superseded "
            "history for a subject/predicate (e.g. user/location)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "subject": {"type": "string",
                            "description": "Entity, e.g. 'user'"},
                "predicate": {"type": "string",
                              "description": "Attribute, e.g. 'location'"},
                "container": {"type": "string"},
            },
            "required": ["subject", "predicate"],
        },
    },
]


class MCPServer:
    """Stdio MCP server backed by a ContextMemory client."""

    def __init__(self, container: str = "brain") -> None:
        self._clients: dict[str, MemoryClient] = {}
        self._container = container

    def _client(self, container: str) -> MemoryClient:
        key = container or self._container
        if key not in self._clients:
            self._clients[key] = MemoryClient(
                key, embedder=DeterministicHashEmbedder()
            )
        return self._clients[key]

    # --- tool implementations ----------------------------------------------

    def _tool_memory(self, args: dict[str, Any]) -> str:
        content = str(args.get("content", "")).strip()
        if not content:
            return "no content provided"
        client = self._client(str(args.get("container", "")))
        session = Session(
            session_id="mcp",
            timestamp=datetime.now(),
            turns=[Turn(role="user", content=content)],
        )
        rep = client.session(session)

        try:
            from contextmemory.server.app import push_event
            push_event("memory_created", {
                "content": content,
                "cells_added": rep.new_cells,
                "total_cells": client.engine.store.cell_count,
            })
        except Exception:
            pass

        return (f"stored {rep.cells} cell(s) ({rep.new_cells} new, "
                f"{rep.dup_cells} duplicate)")

    def _tool_recall(self, args: dict[str, Any]) -> str:
        import time

        query = str(args.get("query", ""))
        client = self._client(str(args.get("container", "")))
        t0 = time.perf_counter()
        report = client.recall(query, top_k=6)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        try:
            from contextmemory.server.app import push_event
            hits_data = [
                {"id": str(h.cell_id), "text": h.text, "score": h.score}
                for h in report.hits
            ]
            push_event("query_executed", {
                "query": query,
                "hits": hits_data,
                "latency_ms": latency_ms,
            })
        except Exception:
            pass

        if not report.hits:
            return "no memories found"
        lines = []
        for h in report.hits[:6]:
            lines.append(f"- [{h.cell_id}] {h.text}")
        return "\n".join(lines)

    def _tool_context(self, args: dict[str, Any]) -> str:
        topic = str(args.get("topic", ""))
        client = self._client(str(args.get("container", "")))
        report = client.recall(topic, top_k=4)
        if not report.hits:
            return "no context found"
        return "\n".join(f"- [{h.cell_id}] {h.text}" for h in report.hits[:4])

    def _tool_forget(self, args: dict[str, Any]) -> str:
        try:
            mid = int(args.get("id", 0))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "forget requires an integer memory id"
        if mid <= 0:
            return "forget requires a positive memory id"
        client = self._client(str(args.get("container", "")))
        store = client.engine.store
        try:
            forgotten = store.forget(mid)
        except AttributeError:
            return f"forget(id={mid}) unsupported by this core build"
        if forgotten:
            try:
                from contextmemory.server.app import push_event
                push_event("memory_forgotten", {"cell_id": mid})
            except Exception:
                pass
            return f"forgot memory {mid}"
        return f"forget(id={mid}) not found or already forgotten"

    def _tool_profile(self, args: dict[str, Any]) -> str:
        client = self._client(str(args.get("container", "")))
        prof = client.profile()
        lines = ["static (durable):"]
        for h in prof.static_facts[:10]:
            lines.append(f"- [{h.cell_id}] {h.text}")
        lines.append("dynamic (recent):")
        for h in prof.dynamic_facts[:10]:
            lines.append(f"- [{h.cell_id}] {h.text}")
        if len(lines) == 2:
            return "profile is empty"
        return "\n".join(lines)

    def _tool_timeline(self, args: dict[str, Any]) -> str:
        subject = str(args.get("subject", "")).strip()
        predicate = str(args.get("predicate", "")).strip()
        if not subject or not predicate:
            return "timeline requires subject and predicate"
        client = self._client(str(args.get("container", "")))
        proj = client.projection(subject, predicate)
        if proj is None:
            return f"no history for {subject}/{predicate}"
        # One bounded recall, then keep the version chain newest-first.
        report = client.recall(f"{subject} {predicate}", top_k=32)
        chain = [h for h in report.hits
                 if h.subject == subject and h.predicate == predicate]
        chain.sort(key=lambda h: (h.valid_from, h.cell_id), reverse=True)
        if not chain:
            return f"no history for {subject}/{predicate}"
        lines = [f"{subject}/{predicate} ({proj.version_count} versions):"]
        for h in chain[:16]:
            tag = "current" if h.cell_id == proj.active_cell else "past"
            lines.append(f"- [{h.cell_id} | {tag}] {h.text}")
        return "\n".join(lines)

    def _dispatch(self, name: str, args: dict[str, Any]) -> str:
        handler = {
            "memory": self._tool_memory,
            "recall": self._tool_recall,
            "context": self._tool_context,
            "profile": self._tool_profile,
            "timeline": self._tool_timeline,
            "forget": self._tool_forget,
        }.get(name)
        if handler is None:
            raise ValueError(f"unknown tool: {name}")
        return handler(args)

    # --- JSON-RPC loop ------------------------------------------------------

    def serve_stdio(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self._handle(msg)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()

    def _handle(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        method = msg.get("method")
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg.get("id"),
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            }
        if method == "notifications/initialized":
            return None
        if method == "ping":
            return {"jsonrpc": "2.0", "id": msg.get("id"), "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": msg.get("id"),
                    "result": {"tools": _TOOLS}}
        if method == "tools/call":
            params = msg.get("params", {}) or {}
            name = params.get("name", "")
            args = params.get("arguments", {}) or {}
            try:
                text = self._dispatch(name, args)
                content = [{"type": "text", "text": text}]
                result: dict[str, Any] = {"content": content}
            except Exception as exc:  # noqa: BLE001 - report to the client
                result = {"content": [{"type": "text",
                                       "text": f"error: {exc}"}],
                          "isError": True}
            return {"jsonrpc": "2.0", "id": msg.get("id"), "result": result}
        return {
            "jsonrpc": "2.0",
            "id": msg.get("id"),
            "error": {"code": -32601, "message": f"method not found: {method}"},
        }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="ContextMemory MCP server (stdio)")
    parser.add_argument("--container", default="brain")
    args = parser.parse_args(argv)
    MCPServer(container=args.container).serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())