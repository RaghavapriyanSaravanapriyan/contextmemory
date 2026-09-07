"""Tests for the HQ product surface: MCP tools, CLI twins, HTTP API, setup.

All offline and model-free: the deterministic engine path needs no GPU.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

from contextmemory.mcp import _TOOLS, MCPServer


def test_mcp_profile_tool() -> None:
    srv = MCPServer(container="hq-profile-test")
    out = srv._dispatch("memory", {"content": "I live in Seattle."})
    assert "cell(s)" in out
    profile = srv._dispatch("profile", {})
    assert "Seattle" in profile
    assert "static" in profile and "recent" in profile or "dynamic" in profile


def test_mcp_timeline_tool() -> None:
    srv = MCPServer(container="hq-timeline-test")
    srv._dispatch("memory", {"content": "I live in Seattle."})
    missing = srv._dispatch(
        "timeline", {"subject": "user", "predicate": "location"})
    assert "no history" in missing or "versions" in missing
    bad = srv._dispatch("timeline", {"subject": "", "predicate": ""})
    assert "requires" in bad


def test_mcp_tool_schemas_have_no_required_container() -> None:
    # container is always optional: agents must not be forced to scope.
    by_name = {t["name"]: t for t in _TOOLS}
    for name in ("profile", "timeline", "recall", "context"):
        assert "container" not in by_name[name]["inputSchema"].get(
            "required", []), name


def test_cli_recall_and_profile(capsys) -> None:
    from contextmemory.cli import _cmd_profile, _cmd_recall

    ns = argparse.Namespace(
        container="hq-cli-test", query="where do I live",
        top_k=5, budget=512, json=True,
    )
    assert _cmd_recall(ns) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "where do I live"
    assert "hits" in payload and "sufficient" in payload

    ns = argparse.Namespace(container="hq-cli-test", top_k=5)
    assert _cmd_profile(ns) == 0
    out = capsys.readouterr().out
    assert "static" in out and "dynamic" in out


def test_cli_setup_show(capsys) -> None:
    from contextmemory.cli import _cmd_setup

    ns = argparse.Namespace(show=True)
    assert _cmd_setup(ns) == 0
    assert "provider" in capsys.readouterr().out


def test_setup_probe_unreachable_returns_empty() -> None:
    from contextmemory.setup import _probe_ollama

    assert _probe_ollama("http://127.0.0.1:9", timeout=0.5) == []


def test_http_api_profile_recall_forget() -> None:
    from contextmemory.server import app as srvapp

    server = srvapp.start_server(18765)
    assert server is not None
    try:
        base = "http://127.0.0.1:18765"

        def post(path: str, body: dict) -> tuple[int, dict]:
            req = urllib.request.Request(
                base + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"}, method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status, json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read().decode())

        def get(path: str) -> tuple[int, dict]:
            with urllib.request.urlopen(base + path, timeout=10) as r:
                return r.status, json.loads(r.read().decode())

        status, _ = post("/v1/memories",
                         {"content": "", "container": "hq-http-test"})
        assert status == 400  # empty content rejected, not stored
        status, body = post("/v1/memories",
                            {"content": "I live in Seattle.",
                             "container": "hq-http-test"})
        assert status == 200 and body["status"] == "ok"

        status, _ = post("/v1/recall",
                         {"query": "", "container": "hq-http-test"})
        assert status == 400
        status, body = post("/v1/recall",
                            {"query": "where do I live",
                             "container": "hq-http-test"})
        assert status == 200 and body["hits"]

        status, prof = get("/v1/profile?space_id=hq-http-test")
        assert status == 200 and "static" in prof and "dynamic" in prof

        status, _ = post("/v1/forget",
                         {"id": "abc", "container": "hq-http-test"})
        assert status == 400
        mid = int(body["hits"][0]["id"])
        status, forgotten = post("/v1/forget",
                                 {"id": mid, "container": "hq-http-test"})
        assert status == 200 and forgotten["forgot"] == mid

        status, metrics = get("/v1/metrics?space_id=hq-http-test")
        assert status == 200
        # Measured latency object exists; null only before any query.
        assert "latency" in metrics
    finally:
        server.shutdown()
