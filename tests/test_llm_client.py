"""Tests for the Ollama chat client.

strip_think is pure (unit). The chat/probe tests hit the real local Ollama with
qwen3.5:4b and are skipped if Ollama is unreachable (public CI passes).
"""

import pytest

import time

from rehearse.llm_client import (
    DEFAULT_MODEL,
    ChatResult,
    _consume_stream,
    chat,
    strip_think,
    think_probe,
    trim_truncated,
    warmup,
)


def _line(obj_json: str) -> bytes:
    return obj_json.encode()


# --- _consume_stream (pure, no server) ------------------------------------

def test_consume_stream_basic():
    lines = [
        b'{"message":{"content":"Hello"},"done":false}',
        b'{"message":{"content":" world"},"done":true,"done_reason":"stop"}',
    ]
    p = _consume_stream(lines, time.monotonic())
    assert p.content == "Hello world"
    assert p.done_reason == "stop"
    assert p.ttft_s is not None


def test_consume_stream_tolerates_empty_and_garbage():
    lines = [
        b"",
        b"not json at all",
        b'{"message":{"content":"ok"},"done":true,"done_reason":"stop"}',
    ]
    p = _consume_stream(lines, time.monotonic())
    assert p.content == "ok"  # garbage/empty skipped, not fatal


def test_consume_stream_captures_thinking_field():
    lines = [
        b'{"message":{"thinking":"let me reason","content":""},"done":false}',
        b'{"message":{"content":"answer"},"done":true,"done_reason":"stop"}',
    ]
    p = _consume_stream(lines, time.monotonic())
    assert p.thinking == "let me reason"
    assert p.content == "answer"


def test_consume_stream_done_reason_length():
    lines = [b'{"message":{"content":"cut off here and"},"done":true,"done_reason":"length"}']
    p = _consume_stream(lines, time.monotonic())
    assert p.done_reason == "length"


# --- trim_truncated -------------------------------------------------------

def test_trim_truncated_drops_fragment():
    assert trim_truncated("Hello there. How are yo") == "Hello there."
    assert trim_truncated('He said "Hi." Then the do') == 'He said "Hi."'


def test_trim_truncated_keeps_when_no_boundary():
    assert trim_truncated("Sure thing") == "Sure thing"


# --- strip_think (pure) ---------------------------------------------------

def test_strip_think_removes_block():
    text, had = strip_think("<think>reasoning here</think>Hello there!")
    assert text == "Hello there!"
    assert had is True


def test_strip_think_multiline_and_case():
    text, had = strip_think("<THINK>\nline1\nline2\n</THINK>  Hi.")
    assert text == "Hi."
    assert had is True


def test_strip_think_no_block():
    text, had = strip_think("Just a plain reply.")
    assert text == "Just a plain reply."
    assert had is False


# --- real Ollama integration (skipped if unreachable) ---------------------

def _ollama_up():
    # Lenient reachability check that ALSO warms the model, so the strict
    # think_probe TTFT check below runs warm (not cold-start). Do NOT use
    # think_probe here — its TTFT SLA would spuriously fail on a cold model.
    try:
        warmup()
        return True
    except Exception:
        return False


_UP = _ollama_up()


@pytest.mark.skipif(not _UP, reason="Ollama / qwen3.5:4b not reachable")
def test_chat_short_reply_no_think():
    r = chat([{"role": "user", "content": "Say hi in one short sentence."}])
    assert isinstance(r, ChatResult)
    assert r.text and not r.had_think_block
    assert "<think>" not in r.text.lower()
    assert r.ttft_s is not None and r.ttft_s >= 0
    # num_predict cap keeps it short
    assert len(r.text) < 600


@pytest.mark.skipif(not _UP, reason="Ollama / qwen3.5:4b not reachable")
def test_think_probe_passes_on_default_model():
    r = think_probe()
    assert r.had_think_block is False
    assert r.text
    print(f"\n[probe] model={DEFAULT_MODEL} ttft={r.ttft_s:.2f}s total={r.total_s:.2f}s")
