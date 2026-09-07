#!/usr/bin/env python3
"""OpenAI-compatible -> Ollama proxy that forces non-streaming chat.

Qwen3-family models emit a long ``reasoning`` trace before the answer. Ollama's
OpenAI-compatible streaming endpoint pushes that reasoning as
``delta.reasoning_content`` and many OpenAI clients (the Vercel AI SDK, which
``generateText`` uses) hang waiting for ``delta.content`` while the model
thinks, or they time out. Non-streaming responses keep ``content`` clean
(reasoning lands in a separate field).

This proxy accepts OpenAI ``/v1/chat/completions`` (streaming or not),
performs the underlying call against Ollama's OWN OpenAI-compatible endpoint
with ``stream: false`` (the fast, validated path — reasoning is kept in a
separate field and ``content`` stays clean), and returns the content.
Streaming clients receive a faithful SSE replay.

stdlib only; no dependencies.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

OLLAMA_URL = "http://localhost:11434"
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 11435
REASONING_EXTRA = 2048  # headroom so content isn't truncated by a long trace


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence request spam
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._json(200, {
                "object": "list",
                "data": [{"id": "qwen3:4b", "object": "model", "owned_by": "ollama"}],
            })
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        print(f"[proxy] POST {self.path}", flush=True)
        if not self.path.startswith("/v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        first_msg = (body.get("messages", [{}]) or [{}])[0]
        log_line = (
            f"[proxy] req model={body.get('model')} "
            f"stream={body.get('stream')} "
            f"msgs={len(body.get('messages', []))} "
            f"prompt={len(first_msg.get('content', ''))} chars"
        )
        print(log_line, flush=True)
        model = body.get("model", "qwen3:4b")
        messages = body.get("messages", [])
        want_stream = bool(body.get("stream", False))

        payload = dict(body)
        payload["stream"] = False  # force non-streaming upstream

        # For streaming clients, open the SSE stream immediately (role chunk
        # + keep-alive) so the client never waits on a first byte while the
        # upstream generation runs. Content is replayed once upstream returns.
        if want_stream:
            self._open_stream(model)

        t0 = time.time()
        print(f"[proxy] forwarding to {OLLAMA_URL}/v1/chat/completions "
              f"stream=false", flush=True)
        req = urllib.request.Request(
            OLLAMA_URL + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            print(f"[proxy] upstream HTTP {exc.code}", flush=True)
            self._json(exc.code, {"error": {"message": exc.read().decode(
                errors="replace")[:500]}})
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[proxy] upstream error: {exc}", flush=True)
            self._json(502, {"error": {"message": str(exc)}})
            return
        print(f"[proxy] upstream done in {round(time.time() - t0, 1)}s",
              flush=True)

        first_choice = (data.get("choices") or [{}])[0]
        first_msg = first_choice.get("message", {}) or {}
        content = first_msg.get("content") or ""
        reasoning = first_msg.get("reasoning") or ""
        latency = round((time.time() - t0) * 1000)

        if want_stream:
            return self._stream_response(model, content)

        self._json(200, {
            "id": f"chatcmpl-proxy-{int(t0*1000)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(json.dumps(messages)) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(json.dumps(messages)) + len(content)) // 4,
            },
            "x_latency_ms": latency,
            "x_reasoning_chars": len(reasoning),
        })

    def _open_stream(self, model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self._chunk(model, {"role": "assistant"})
        self.wfile.flush()

    def _stream_response(self, model, content):
        def chunk(delta, finish=None):
            payload = {
                "id": "chatcmpl-proxy", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        for i in range(0, len(content), 8):
            chunk({"content": content[i:i + 8]})
        chunk({}, finish="stop")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _chunk(self, model, delta, finish=None):
        payload = {
            "id": "chatcmpl-proxy", "object": "chat.completion.chunk",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())

    def _json(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Threaded(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else LISTEN_PORT
    server = Threaded((LISTEN_HOST, port), Handler)
    print(f"ollama-proxy on http://{LISTEN_HOST}:{port} -> {OLLAMA_URL} "
          "(forces non-streaming chat)", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())