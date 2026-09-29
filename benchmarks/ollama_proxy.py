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
import os
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
DUMP_DIR = os.environ.get("SM_PROXY_DUMP_DIR", "")  # opt-in request/response dump


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
        # Qwen3-family models burn their whole budget on a hidden reasoning
        # trace through the OpenAI-compatible layer, and the compat layer
        # IGNORES `think: false` when `tools` are present (verified). The
        # native /api/chat endpoint honors think:false together with tools,
        # so qwen3-family requests are translated to the native protocol.
        if str(payload.get("model", "")).startswith("qwen3"):
            return self._native_chat(payload, want_stream, t_msg=first_msg)

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
        tool_calls = first_msg.get("tool_calls") or []
        latency = round((time.time() - t0) * 1000)

        self._dump(body, data, content, reasoning, tool_calls)

        if want_stream:
            return self._stream_response(model, content)

        # Pass tool calls through: extraction pipelines that use function
        # calling break silently if tool_calls is dropped from the response.
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = first_choice.get("finish_reason") or "stop"
        self._json(200, {
            "id": f"chatcmpl-proxy-{int(t0*1000)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": len(json.dumps(messages)) // 4,
                "completion_tokens": len(content) // 4,
                "total_tokens": (len(json.dumps(messages)) + len(content)) // 4,
            },
            "x_latency_ms": latency,
            "x_reasoning_chars": len(reasoning),
        })

    def _native_chat(self, payload, want_stream, t_msg):
        """Translate an OpenAI chat request to Ollama's native /api/chat.

        Why: the OpenAI-compatible layer ignores `think: false` when `tools`
        are present, so qwen3-family models spend minutes on a hidden
        reasoning trace before (or instead of) the tool call. The native
        endpoint honors think:false together with tools.
        """
        model = payload.get("model", "qwen3:4b")
        native = {
            "model": model,
            "messages": payload.get("messages", []),
            "stream": False,
            "think": False,
            "keep_alive": "10m",
        }
        tools = payload.get("tools")
        if tools:
            native["tools"] = tools
            tc = payload.get("tool_choice")
            if isinstance(tc, str):
                native["tool_choice"] = tc
        options = {}
        if payload.get("temperature") is not None:
            options["temperature"] = payload["temperature"]
        if payload.get("max_tokens"):
            options["num_predict"] = payload["max_tokens"]
        options["num_ctx"] = payload.get("num_ctx", 8192)
        if options:
            native["options"] = options
        if (payload.get("response_format") or {}).get("type") == "json_object":
            native["format"] = "json"

        if want_stream:
            self._open_stream(model)
        t0 = time.time()
        print(f"[proxy] forwarding to {OLLAMA_URL}/api/chat "
              f"(native, think=false, tools={len(tools or [])})", flush=True)
        req = urllib.request.Request(
            OLLAMA_URL + "/api/chat",
            data=json.dumps(native).encode(),
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

        message = data.get("message", {}) or {}
        content = message.get("content") or ""
        reasoning = message.get("thinking") or ""
        tool_calls = message.get("tool_calls") or []
        # Native tool_call arguments are a JSON object; the OpenAI wire
        # format wants a JSON string. Serialize dicts in place (copy).
        if tool_calls:
            fixed = []
            for tc in tool_calls:
                tc = dict(tc)
                fn = dict(tc.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, dict):
                    fn["arguments"] = json.dumps(args)
                tc["function"] = fn
                tc.setdefault("type", "function")
                tc.setdefault("id", f"call_{int(t0*1000)}_{len(fixed)}")
                fixed.append(tc)
            tool_calls = fixed
        done_reason = data.get("done_reason") or "stop"
        finish = "tool_calls" if tool_calls else (
            "length" if done_reason == "length" else "stop")

        self._dump({"model": model, "tools": tools,
                    "messages": native["messages"]},
                   {"content": content, "reasoning": reasoning[:2000],
                    "tool_calls": tool_calls, "finish_reason": finish},
                   content, reasoning, tool_calls)

        if want_stream:
            return self._stream_response(model, content)

        message_out = {"role": "assistant", "content": content}
        if tool_calls:
            message_out["tool_calls"] = tool_calls
        self._json(200, {
            "id": f"chatcmpl-proxy-{int(t0*1000)}",
            "object": "chat.completion",
            "created": int(t0),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message_out,
                "finish_reason": finish,
            }],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (data.get("prompt_eval_count", 0)
                                 + data.get("eval_count", 0)),
            },
            "x_latency_ms": round((time.time() - t0) * 1000),
            "x_reasoning_chars": len(reasoning),
        })

    def _dump(self, body, data, content, reasoning, tool_calls):
        """Opt-in full dump of one exchange for pipeline diagnosis."""
        if not DUMP_DIR:
            return
        try:
            os.makedirs(DUMP_DIR, exist_ok=True)
            ts = time.strftime("%H%M%S") + f"-{int(time.time()*1000)%1000:03d}"
            with open(f"{DUMP_DIR}/{ts}.json", "w", encoding="utf-8") as fh:
                json.dump({
                    "request": {
                        "model": body.get("model"),
                        "tools": body.get("tools"),
                        "tool_choice": body.get("tool_choice"),
                        "response_format": body.get("response_format"),
                        "messages": body.get("messages"),
                    },
                    "response": {
                        "content": content,
                        "reasoning": reasoning[:2000],
                        "tool_calls": tool_calls,
                        "finish_reason": (data.get("choices") or [{}])[0].get(
                            "finish_reason"),
                    },
                }, fh, indent=1)
        except OSError:
            pass

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