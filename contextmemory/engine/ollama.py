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

import shutil
import subprocess
import time

import httpx

DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_API_KEY = "ollama"
_SERVE_WAIT_S = 12.0
_DEFAULT_MAX_TOKENS = 2048


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
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, headers=headers
        )

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
            "options": {"temperature": temperature},
        }
        if json_mode:
            # Ollama's JSON grammar forces valid JSON output — essential for
            # small local models that otherwise ramble instead of emitting the
            # requested {"cells": [...]} object.
            payload["format"] = "json"
        cap = max_tokens if max_tokens is not None else self._max_tokens
        if cap:
            payload["options"]["num_predict"] = cap
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        message = resp.json().get("message", {}) or {}
        return (message.get("content", "") or "").strip()

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

    def reader(self, model: str, *, max_tokens: int | None = None) -> OllamaChatClient:
        """A reader for ``model`` on this server (native, thinking disabled)."""
        return OllamaChatClient(
            self.base_url,
            model,
            api_key=self._api_key,
            timeout=self._timeout,
            max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
        )

    def set_config(self, base_url: str, api_key: str) -> None:
        """Reconfigure the endpoint after the user edits it in the TUI."""
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._models = []

    def close(self) -> None:
        self.stop_managed()