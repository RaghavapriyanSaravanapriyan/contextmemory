"""Tests for the native Ollama client helpers (thinking suppression)."""

from __future__ import annotations

from contextmemory.engine.ollama import strip_thinking


def test_strip_thinking_removes_full_block() -> None:
    text = "<think>reasoning about the user</think>\n\nNice to meet you!"
    assert strip_thinking(text) == "Nice to meet you!"


def test_strip_thinking_removes_unclosed_block() -> None:
    # Truncated generations end mid-reasoning; drop the dangling block.
    assert strip_thinking("answer <think>reasoning") == "answer"


def test_strip_thinking_removes_orphaned_close_tag() -> None:
    # Observed from qwen3:4b with think=false: responses can end with a
    # bare </think> after the opening tag was consumed by the template.
    assert strip_thinking("</think>") == ""


def test_strip_thinking_keeps_plain_text() -> None:
    assert strip_thinking("plain answer") == "plain answer"
