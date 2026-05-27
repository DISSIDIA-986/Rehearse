"""Ollama chat client for the conversation loop (D1 + non-thinking, structural).

Talks to the local Ollama /api/chat. Key requirements baked in:
- `think: false` so a thinking model (qwen3.5) does NOT burn time on reasoning
  tokens before answering — that would wreck time-to-first-token. We detect a
  thinking violation TWO ways (Codex BLOCKER 3: non-thinking is structural):
  (a) Ollama returns reasoning in a separate `message.thinking` field, and
  (b) some models inline `<think>...</think>` in content. The startup probe
  fails loudly on EITHER, plus on slow TTFT.
- `keep_alive: -1` keeps the model warm (no 30-60s cold start per turn).
- `num_predict` caps reply length (D1: short replies). If the cap truncates
  mid-sentence (`done_reason == "length"`), we drop the trailing fragment so
  TTS never speaks a chopped sentence.
- streaming so we can record llm_first_token for the latency budget, while the
  loop still waits for the full reply before TTS (D1 full-serialize).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_NUM_PREDICT = 120  # ~3 short sentences; hard cap on reply length
DEFAULT_PROBE_MAX_TTFT_S = 3.0  # warm probe should answer fast

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENT_END_TRIM_RE = re.compile(r"[.!?。！？]+[\"')\]]*")


@dataclass
class ChatResult:
    text: str  # assistant reply, <think> stripped + truncation trimmed
    ttft_s: float | None  # time to first streamed content token
    total_s: float  # total generation wall time
    had_think_block: bool  # model reasoned despite think=false (tag OR thinking field)


@dataclass
class _Parsed:
    content: str
    thinking: str
    ttft_s: float | None
    done_reason: str | None


def strip_think(text: str) -> tuple[str, bool]:
    """Remove inline <think>...</think>; return (clean_text, had_block)."""
    had = bool(_THINK_RE.search(text))
    return _THINK_RE.sub("", text).strip(), had


def trim_truncated(text: str) -> str:
    """Drop a trailing incomplete sentence (call only on a length-capped reply)."""
    matches = list(_SENT_END_TRIM_RE.finditer(text))
    if matches:
        return text[: matches[-1].end()].strip()
    return text.strip()  # no sentence boundary at all -> keep what we have


def _consume_stream(lines: Iterable[bytes | str], start: float) -> _Parsed:
    """Parse Ollama's NDJSON stream tolerantly (skip empty/garbled lines).

    Pure over an iterable of lines so it is unit-testable without a live server.
    `start` is a monotonic timestamp used to compute time-to-first-token.
    """
    parts: list[str] = []
    thinking: list[str] = []
    ttft: float | None = None
    done_reason: str | None = None
    for raw in lines:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue  # tolerate a partial/garbled line rather than killing the turn
        msg = obj.get("message") or {}
        content = msg.get("content") or ""
        think = msg.get("thinking") or ""
        if content:
            if ttft is None:
                ttft = time.monotonic() - start
            parts.append(content)
        if think:
            thinking.append(think)
        if obj.get("done"):
            done_reason = obj.get("done_reason")
            break
    return _Parsed("".join(parts), "".join(thinking), ttft, done_reason)


def chat(
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    num_predict: int = DEFAULT_NUM_PREDICT,
    temperature: float = 0.7,
    think: bool = False,
    keep_alive: int | str = -1,
) -> ChatResult:
    """Send a chat turn to Ollama (streaming) and return the full reply + timing."""
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "keep_alive": keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            parsed = _consume_stream(resp, start)
    except urllib.error.HTTPError as e:  # 404 model-not-found, 500, ...
        body = e.read().decode("utf-8", "replace")[:300] if e.fp else ""
        raise RuntimeError(f"Ollama chat HTTP {e.code}: {body or e.reason}") from e
    except (urllib.error.URLError, TimeoutError) as e:  # refused, timeout, DNS
        raise RuntimeError(f"Ollama chat request failed: {e}") from e

    total = time.monotonic() - start
    text, had_tag = strip_think(parsed.content)
    if parsed.done_reason == "length":  # cap truncated mid-sentence -> trim fragment
        text = trim_truncated(text)
    had_think = had_tag or bool(parsed.thinking.strip())
    return ChatResult(text=text, ttft_s=parsed.ttft_s, total_s=total, had_think_block=had_think)


def warmup(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> None:
    """Preload the model into memory (keep_alive=-1) before the TTFT probe.

    The first call after Ollama unloads a model pays a 30-60s cold start, so a
    fair `think_probe` TTFT check must run AFTER this. Startup order is:
    warmup() -> think_probe().
    """
    chat([{"role": "user", "content": "hi"}], model=model, host=host, num_predict=1)


def think_probe(
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    max_ttft_s: float = DEFAULT_PROBE_MAX_TTFT_S,
) -> ChatResult:
    """Startup health check: confirm non-thinking + fast TTFT (run after warmup()).

    Raises RuntimeError if the model reasons despite think=false (a <think> tag
    OR a non-empty `thinking` field), returns nothing, or warm TTFT exceeds
    max_ttft_s. Call once at startup (after warmup()) so the latency contract
    fails loudly instead of degrading silently.
    """
    r = chat(
        [{"role": "user", "content": "Say a short friendly hello in one sentence."}],
        model=model,
        host=host,
        num_predict=40,
    )
    if r.had_think_block:
        raise RuntimeError(
            f"model {model!r} reasoned despite think=false (think block / thinking "
            "field present) — non-thinking mode not honored; TTFT will be unacceptable"
        )
    if not r.text.strip():
        raise RuntimeError(f"model {model!r} returned an empty reply on the probe")
    if r.ttft_s is not None and r.ttft_s > max_ttft_s:
        raise RuntimeError(
            f"warm TTFT {r.ttft_s:.2f}s exceeds {max_ttft_s}s for {model!r} — "
            "check the model is preloaded (keep_alive) and not thinking"
        )
    return r
