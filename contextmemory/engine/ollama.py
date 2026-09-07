"""Ollama connection manager for the TUI and live engine.

Supports two modes:

* **external** — connect to an already-running ``ollama serve`` (started in
  another terminal or as a system service). This is the "LLM runs via Ollama
  in another terminal" flow.
* **managed** — launch ``ollama serve`` as a supervised child process from
  this process, so the TUI runs Ollama under the hood.

The manager probes the server, lists available models, can start/stop a
managed server, and builds OpenAI-compatible readers for any model it finds.
Model-agnostic: any model Ollama serves (qwen3, llama3, gemma, ...) can be
selected and used for extraction and answer generation.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Iterator
from typing import Any

import httpx

_THINK_BLOCK = re.compile(r"<think>.*?</think>|<think>.*$|</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove Qwen3-style ``<think>`` blocks from visible output.

    Ollama's ``think: false`` disables the separate ``thinking`` field, but
    thinking models (qwen3, ...) may still emit reasoning inline in
    ``content`` — wrapped in tags or as an orphaned ``</think>``. Strip the
    tagged reasoning before showing or scoring a response. Untagged
    freeform reasoning cannot be detected reliably; it should not be hidden
    silently.
    """
    return _THINK_BLOCK.sub("", text).strip()


def _strip_think_incremental(chunk: str, in_think: bool) -> tuple[str, bool]:
    """Suppress ``<think>`` regions in a streaming chunk.

    Returns (visible_text, in_think). A trailing partial tag (``<``, ``<t``,
    ...) is withheld by the caller via the residual buffer pattern: this
    function only strips complete regions and reports state; the caller
    holds ``buf`` across chunks. Simpler and allocation-light: process the
    small chunk string only.
    """
    text = chunk
    visible_parts: list[str] = []
    while True:
        if in_think:
            end = text.find("</think>")
            if end == -1:
                return "".join(visible_parts), True
            text = text[end + len("</think>"):]
            in_think = False
            continue
        start = text.find("<think>")
        if start == -1:
            # Possible split "<think" tail: withhold "<..." suffix. The
            # caller prepends it to the next chunk via its residual buffer.
            lt = text.rfind("<")
            if lt != -1 and len(text) - lt <= 7:
                visible_parts.append(text[:lt])
                return "".join(visible_parts), False
            # Orphaned close tag from qwen3 templates.
            visible_parts.append(text.replace("</think>", ""))
            return "".join(visible_parts), False
        # Emit text before the block, enter thinking.
        visible_parts.append(text[:start].replace("</think>", ""))
        text = text[start + len("<think>"):]
        in_think = True
        if not text:
            return "".join(visible_parts), True

DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_API_KEY = "ollama"
_SERVE_WAIT_S = 12.0
_DEFAULT_MAX_TOKENS = 2048
# Speed defaults: keep the model resident so repeat calls skip the load
# penalty, and bound context so a 1B model on CPU stays in cache.
# Extraction needs room for multi-turn transcripts (LongMemEval sessions
# carry ~12 turns), so it gets a wider window than chat.
_KEEP_ALIVE = "10m"
_DEFAULT_NUM_CTX = 2048
_EXTRACTION_NUM_CTX = 4096


class OllamaError(RuntimeError):
    """Raised when an Ollama operation cannot complete."""


class OllamaChatClient:
    """Native ``/api/chat`` client for Ollama with thinking disabled.

    Uses Ollama's native endpoint instead of the OpenAI-compatible one because
    the compatibility layer ignores the ``think`` flag on Qwen3-family models,
    which then spend their entire budget on a hidden reasoning trace and return
    no answer. ``think: false`` only works on ``/api/chat``.

    Model-agnostic: extraction and answer generation use the same client; any
    model Ollama serves works. ``num_predict`` bounds generation so a verbose
    local model cannot hang the write path.

    Speed: one pooled ``httpx.Client`` (keep-alive), ``keep_alive: 10m`` so
    the model stays resident across calls, a bounded ``num_ctx`` so small
    CPU models stay fast, and token-by-token ``stream_*`` methods so the
    first token reaches the user in ~100ms instead of after the full
    generation.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        num_ctx: int = _DEFAULT_NUM_CTX,
        keep_alive: str = _KEEP_ALIVE,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._num_ctx = num_ctx
        self._keep_alive = keep_alive
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers=headers,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
        )

    def _options(
        self,
        temperature: float,
        max_tokens: int | None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        opts: dict[str, Any] = {"temperature": temperature}
        cap = max_tokens if max_tokens is not None else self._max_tokens
        if cap:
            opts["num_predict"] = cap
        ctx = num_ctx if num_ctx is not None else self._num_ctx
        if ctx:
            opts["num_ctx"] = ctx
        return opts

    def warm(self) -> bool:
        """Preload the model so the first real call skips the load penalty.

        Uses the client's configured ``num_ctx``: Ollama reloads the model
        when the context size changes, so warming with a different size
        would not prevent the first-call reload.
        """
        try:
            resp = self._client.post(
                "/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": self._keep_alive,
                    "options": {"num_predict": 1, "num_ctx": self._num_ctx,
                                "temperature": 0.0},
                },
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": self._options(
                temperature, max_tokens,
                _EXTRACTION_NUM_CTX if json_mode else None,
            ),
        }
        if json_mode:
            # Ollama's JSON grammar forces valid JSON output — essential for
            # small local models that otherwise ramble instead of emitting the
            # requested {"cells": [...]} object.
            payload["format"] = "json"
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        message = resp.json().get("message", {}) or {}
        return strip_thinking(message.get("content", "") or "")

    def stream_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield content deltas as they arrive (``stream: true`` NDJSON).

        TTFT drops from full-generation latency to time-to-first-chunk
        (~100ms on a warm 1B model); total time is unchanged but perceived
        latency collapses. Thinking blocks are suppressed incrementally so
        Qwen3 reasoning never flashes on screen.
        """
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": self._options(temperature, max_tokens),
        }
        in_think = False
        buf = ""
        with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue
                msg = chunk.get("message", {}) or {}
                # Ollama streams thinking separately; think:false keeps it
                # empty, but never render it even if present.
                delta = msg.get("content", "") or ""
                if not delta:
                    continue
                buf += delta
                # Hold back a possible split tag tail ("<", "<th", ...)
                # across NDJSON chunks; process only the safe prefix.
                hold = ""
                if not in_think:
                    lt = buf.rfind("<")
                    if lt != -1 and ">" not in buf[lt:] and len(buf) - lt <= 8:
                        hold, buf = buf[lt:], buf[:lt]
                # Incremental <think> suppression without buffering the
                # whole response.
                out, in_think = _strip_think_incremental(buf, in_think)
                buf = hold
                if out:
                    yield out
        # Flush any remainder (a dangling "<think" tail is reasoning noise).
        if buf and not in_think and "<think" not in buf:
            yield buf.replace("</think>", "")

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Run one native Ollama chat turn with function tools enabled.

        Tool routing uses a tight token cap (fast decision, ~200-400ms on
        a 1B model); the follow-up text answer should use
        :meth:`stream_complete` so the user sees tokens immediately.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "stream": False,
            "think": False,
            "keep_alive": self._keep_alive,
            "options": self._options(temperature, max_tokens),
        }
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        message = resp.json().get("message", {}) or {}
        content = strip_thinking(message.get("content") or "")
        message["content"] = content
        return message

    def route_tools_fast(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Tool-routing pass capped at 128 tokens for minimum latency."""
        return self.chat_with_tools(
            messages, tools, temperature=temperature, max_tokens=128
        )

    def close(self) -> None:
        self._client.close()


class OllamaManager:
    """Probe, supervise, and talk to a local Ollama server."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        api_key: str = _DEFAULT_API_KEY,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._models: list[str] = []
        self._last_error: str | None = None

    # --- probe --------------------------------------------------------------

    def ping(self, timeout: float = 0.8) -> bool:
        """True if the server responds on /api/tags."""
        try:
            httpx.get(f"{self.base_url}/api/tags", timeout=timeout)
            return True
        except (httpx.HTTPError, OSError):
            return False

    @property
    def running(self) -> bool:
        return self.ping()

    @property
    def is_managed(self) -> bool:
        """True when we launched the server ourselves and it is alive."""
        return self._proc is not None and self._proc.poll() is None

    @property
    def last_error(self) -> str | None:
        return self._last_error

    # --- models -------------------------------------------------------------

    def list_models(self, refresh: bool = True) -> list[str]:
        """Return model names served by Ollama (``/api/tags``)."""
        if not refresh and self._models:
            return self._models
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            self._models = sorted(
                m["name"] for m in resp.json().get("models", [])
            )
            self._last_error = None
        except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError) as exc:
            self._last_error = f"cannot list models: {exc}"
            self._models = []
        return self._models

    def model_details(self, refresh: bool = True) -> list[dict]:
        """Model info straight from Ollama: name, size GB, parameter size.

        Only fields Ollama actually reports are included. Empty list when the
        server is unreachable.
        """
        if not refresh and self._models:
            pass
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
        except (httpx.HTTPError, OSError, KeyError, TypeError, ValueError):
            return []
        out = []
        for m in models:
            size = int(m.get("size") or 0)
            details = m.get("details") or {}
            out.append({
                "name": m.get("name", ""),
                "size_gb": round(size / (1024 ** 3), 1) if size else 0.0,
                "parameter_size": details.get("parameter_size", ""),
                "quantization": details.get("quantization_level", ""),
            })
        return sorted(out, key=lambda d: d["name"])

    # --- managed server -----------------------------------------------------

    def start_managed(self, wait: float = _SERVE_WAIT_S) -> bool:
        """Launch ``ollama serve`` as a child process and wait until reachable.

        Safe to call when an external server is already running (it returns
        True immediately). Returns False and records an error on failure.
        """
        if self.running:
            self._last_error = None
            return True
        binary = shutil.which("ollama")
        if not binary:
            self._last_error = (
                "ollama binary not found on PATH. Install Ollama first: "
                "https://ollama.com/download"
            )
            return False
        try:
            self._proc = subprocess.Popen(
                [binary, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self._last_error = f"failed to launch ollama serve: {exc}"
            return False
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if self.ping(timeout=0.4):
                self._last_error = None
                return True
            time.sleep(0.2)
        self._last_error = (
            "ollama serve started but did not become reachable on "
            f"{self.base_url}"
        )
        return False

    def stop_managed(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    # --- reader -------------------------------------------------------------

    def reader(
        self,
        model: str,
        *,
        max_tokens: int | None = None,
        num_ctx: int = _DEFAULT_NUM_CTX,
    ) -> OllamaChatClient:
        """A reader for ``model`` on this server (native, thinking disabled)."""
        return OllamaChatClient(
            self.base_url,
            model,
            api_key=self._api_key,
            timeout=self._timeout,
            max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
            num_ctx=num_ctx,
            keep_alive=_KEEP_ALIVE,
        )

    def set_config(self, base_url: str, api_key: str) -> None:
        """Reconfigure the endpoint after the user edits it in the TUI."""
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._models = []

    def close(self) -> None:
        self.stop_managed()
