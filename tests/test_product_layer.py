"""Tests for the Stage 2 product layer: config, observability, MCP."""

from __future__ import annotations

import json
import subprocess
import sys

from contextmemory.config import AppConfig
from contextmemory.engine.memory import RecallReport
from contextmemory.mcp import _TOOLS, MCPServer
from contextmemory.observability import RetrievalEvent, RetrievalTracker

# --- config ----------------------------------------------------------------


def test_config_defaults_are_unonboarded(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = AppConfig.load()
    assert cfg.onboarded is False
    assert cfg.building == "ai-agent"


def test_config_roundtrip_persists(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfg = AppConfig.load()
    cfg.complete_onboarding(
        building="coding-agent", provider="ollama", model="qwen3:4b",
        base_url="http://localhost:11434",
    )
    loaded = AppConfig.load()
    assert loaded.onboarded is True
    assert loaded.building == "coding-agent"
    assert loaded.model == "qwen3:4b"
    assert loaded.provider == "ollama"


def test_config_missing_file_returns_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert AppConfig.load().onboarded is False


def test_building_label_resolution() -> None:
    assert AppConfig(building="ai-agent").resolve_building_label() == "AI Agent"
    assert AppConfig(building="custom").resolve_building_label() == "Custom Setup"


# --- observability ----------------------------------------------------------


def _report(*, compile_ms=0.1, embed_ms=0.0, search_ms=0.5, pack_ms=0.2,
            sufficient=True, used_fallback=False) -> RecallReport:
    P = type("P", (), {
        "tokens": 10,
        "used_fallback": used_fallback,
        "sufficient": sufficient,
    })

    r = RecallReport()
    r.compile_ms = compile_ms
    r.embed_ms = embed_ms
    r.search_ms = search_ms
    r.pack_ms = pack_ms
    r.pack = P()
    return r


def test_tracker_records_and_stats() -> None:
    t = RetrievalTracker()
    t.record(RetrievalEvent("q1", report=_report(compile_ms=0.05),
                            hits=2))
    t.record(RetrievalEvent("q2", report=_report(compile_ms=0.05, embed_ms=1.0),
                            hits=5))
    assert t.count == 2
    snap = t.snapshot()
    assert snap["count"] == 2
    assert snap["avg_ms"] > 0
    assert snap["fast_path_rate"] == 0.5  # one skipped embedding
    assert snap["hit_rate"] == 1.0


def test_tracker_empty_is_safe() -> None:
    snap = RetrievalTracker().snapshot()
    assert snap["count"] == 0
    assert snap["avg_ms"] == 0.0


def test_tracker_caps_history() -> None:
    t = RetrievalTracker()
    for i in range(1100):
        t.record(RetrievalEvent(f"q{i}", report=_report(), hits=1))
    assert t.count <= 1000


def test_event_explains_fast_path() -> None:
    e = RetrievalEvent("q", report=_report(embed_ms=0.0, sufficient=True), hits=3)
    text = "\n".join(e.explain())
    assert "Embedding generation skipped" in text
    assert "no deep rerank" in text


def test_event_without_report_is_honest() -> None:
    e = RetrievalEvent("q", exception="boom")
    assert "No measurement available" in "\n".join(e.explain())


# --- MCP --------------------------------------------------------------------


def test_mcp_tools_are_declared() -> None:
    names = {t["name"] for t in _TOOLS}
    assert names == {"memory", "recall", "context", "forget"}


def test_mcp_initialize_handshake() -> None:
    srv = MCPServer()
    resp = srv._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert resp["result"]["serverInfo"]["name"] == "contextmemory"
    assert resp["result"]["protocolVersion"]


def test_mcp_memory_recall_roundtrip() -> None:
    srv = MCPServer()
    srv._handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    r1 = srv._handle({
        "jsonrpc": "2.0", "id": 2, "method": "tools/call",
        "params": {"name": "memory",
                   "arguments": {"content": "I live in Seattle."}},
    })
    assert "cell(s)" in r1["result"]["content"][0]["text"]
    r2 = srv._handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "recall",
                   "arguments": {"query": "where does the user live"}},
    })
    assert "Seattle" in r2["result"]["content"][0]["text"]


def test_mcp_unknown_tool_reports_error() -> None:
    srv = MCPServer()
    resp = srv._handle({
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "nope", "arguments": {}},
    })
    assert resp["result"]["isError"] is True


def test_mcp_unknown_method() -> None:
    srv = MCPServer()
    resp = srv._handle({"jsonrpc": "2.0", "id": 1, "method": "bogus"})
    assert resp["error"]["code"] == -32601


def test_mcp_cli_entrypoint() -> None:
    payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    proc = subprocess.run(
        [sys.executable, "-m", "contextmemory.mcp"],
        input=payload + "\n", capture_output=True, text=True, timeout=30,
        cwd=__file__.rsplit("/", 2)[0] + "/../",
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["result"]["serverInfo"]["name"] == "contextmemory"