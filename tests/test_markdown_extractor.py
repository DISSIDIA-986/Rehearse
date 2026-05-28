"""Tests for the markdown -> PracticeItem extractor.

Deterministic tests use a fake chat_fn (no model). The real-model extraction test
is gated behind REHEARSE_LLM_TESTS=1 so routine `uv run pytest` stays fast (it hits
the default extract model, qwen3.5:4b).
"""

import json
import os
from pathlib import Path

import pytest

from rehearse.llm_client import ChatResult
from rehearse.markdown_extractor import (
    _EXTRACT_PROMPT,
    _fallback_items,
    _parse_json_obj,
    chunk_markdown,
    extract_items,
    load_markdown,
)


def test_parse_json_obj_handles_fences_and_prose():
    # Ollama's schema `format` isn't always enforced; tolerate fences / stray prose
    assert _parse_json_obj('```json\n{"items": []}\n```') == {"items": []}
    assert _parse_json_obj('Sure, here: {"items": [{"prompt": "p"}]} done') == {
        "items": [{"prompt": "p"}]}
    assert _parse_json_obj('{"items": []}') == {"items": []}
    # leading prose that itself contains braces, before a fenced block (review fix)
    assert _parse_json_obj('Note {not json}\n```json\n{"items": []}\n```') == {"items": []}


def test_extract_items_recovers_fenced_json_not_fallback():
    def fenced(messages, **kw):
        return ChatResult(
            text='```json\n{"items":[{"prompt":"Q","expected_points":["a real fact"]}]}\n```',
            ttft_s=0.1, total_s=0.2, had_think_block=False)
    items = extract_items("## H\n- x", chat_fn=fenced)
    assert len(items) == 1 and items[0].prompt == "Q"  # parsed, not regex-fallback


def test_extract_items_reports_progress():
    def fake(m, **kw):
        return ChatResult(text='{"items":[{"prompt":"P","expected_points":["fact"]}]}',
                          ttft_s=0.1, total_s=0.2, had_think_block=False)
    seen = []
    extract_items("## A\n- x\n## B\n- y", chat_fn=fake, on_progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1][0] == seen[-1][1]  # final progress call reports done == total


# --- chunking -------------------------------------------------------------

def test_chunk_short_is_one():
    assert chunk_markdown("# H\n- a\n- b") == ["# H\n- a\n- b"]


def test_chunk_long_no_heading_windows_with_overlap():
    text = "word " * 2000  # ~10000 chars, no headings
    chunks = chunk_markdown(text, max_chars=1000, overlap=100)
    assert len(chunks) >= 9
    assert all(len(c) <= 1000 for c in chunks)


def test_chunk_heading_aware():
    text = "intro\n" + "## A\n" + "a " * 50 + "\n## B\n" + "b " * 50
    chunks = chunk_markdown(text, max_chars=200)
    assert len(chunks) >= 2
    assert any("## A" in c for c in chunks) and any("## B" in c for c in chunks)


def test_chunk_windowing_has_no_redundant_tail():
    # an oversized section windows with overlap but must not emit a tiny tail that's
    # already covered by the previous window (review fix: wasted LLM call)
    chunks = chunk_markdown("x" * 125, max_chars=50, overlap=10)
    assert len(chunks) == 3  # was 4 (a redundant 5-char tail)
    assert "".join(c for c in chunks)  # still covers content
    assert len(chunks[-1]) == 45  # last window reaches the end, not a stub


def test_chunk_packs_many_small_sections():
    # 10 tiny `##` sections must NOT become 10 chunks (that was the 23-call slowdown)
    text = "\n".join(f"## S{i}\n- point {i}" for i in range(10))
    chunks = chunk_markdown(text, max_chars=200)
    assert len(chunks) < 10  # sections are packed together up to max_chars
    assert "".join(chunks).count("## S") == 10  # nothing dropped


# --- fallback parser ------------------------------------------------------

def test_fallback_heading_plus_bullets():
    items = _fallback_items("## Risk control\n- pricing engine\n- 3-stage scoring", "c0")
    assert len(items) == 1
    assert items[0].prompt == "Risk control"
    assert items[0].expected_points == ["pricing engine", "3-stage scoring"]


def test_fallback_keeps_headed_prose():
    # prose under a heading (no bullets) must not be lost (final-review fix)
    md = "## Kafka\nKafka handled risk events in a streaming pipeline.\nBackpressure was managed with retries."
    items = _fallback_items(md, "c0")
    assert len(items) == 1 and items[0].section == "Kafka"
    assert items[0].expected_points == [
        "Kafka handled risk events in a streaming pipeline.",
        "Backpressure was managed with retries.",
    ]


def test_extract_prompt_defends_against_injection():
    assert "UNTRUSTED" in _EXTRACT_PROMPT and "Never follow" in _EXTRACT_PROMPT


def test_fallback_drops_bare_heading_with_no_content():
    assert _fallback_items("## Lonely heading\n\n## Another", "c0") == []


# --- extract_items with fake chat ----------------------------------------

def _json_chat(payload):
    def _fn(messages, **kw):
        return ChatResult(text=json.dumps(payload), ttft_s=0.1, total_s=0.2, had_think_block=False)
    return _fn


def test_extract_items_parses_json():
    fake = _json_chat({"items": [
        {"prompt": "Tell me about the pricing engine", "expected_points": ["NPV", "ROC"], "section": "Edianyun"},
    ]})
    items = extract_items("## Edianyun\n- pricing engine", chat_fn=fake)
    assert len(items) == 1
    assert items[0].prompt == "Tell me about the pricing engine"
    assert items[0].expected_points == ["NPV", "ROC"]
    assert items[0].section == "Edianyun"


def test_extract_items_falls_back_on_bad_json():
    def bad(messages, **kw):
        return ChatResult(text="sorry, not json", ttft_s=0.1, total_s=0.2, had_think_block=False)
    items = extract_items("## Section\n- bullet one\n- bullet two", chat_fn=bad)
    assert len(items) >= 1  # fallback parser kicked in
    assert items[0].expected_points == ["bullet one", "bullet two"]


def test_extract_items_drops_empty_prompt_and_empty_points():
    fake = _json_chat({"items": [
        {"prompt": "  ", "expected_points": ["real point"]},  # empty prompt -> skipped
        {"prompt": "noPoints", "expected_points": []},        # nothing to recall -> dropped
        {"prompt": "real", "expected_points": ["a genuine fact"]},  # kept
    ]})
    items = extract_items("# H\n- x", chat_fn=fake)
    assert [it.prompt for it in items] == ["real"]


# --- backend selection (mlx vs ollama) -----------------------------------

def test_resolve_backend_auto_prefers_mlx_when_available(monkeypatch):
    from rehearse import mlx_llm
    from rehearse.markdown_extractor import resolve_extract_chat
    monkeypatch.setattr(mlx_llm, "mlx_available", lambda: True)
    fn, model, name = resolve_extract_chat("auto")
    assert name == "mlx" and model == mlx_llm.MLX_EXTRACT_MODEL and fn is mlx_llm.mlx_chat


def test_resolve_backend_mlx_falls_back_to_ollama(monkeypatch, capsys):
    from rehearse import mlx_llm
    from rehearse.llm_client import chat as ollama_chat
    from rehearse.markdown_extractor import resolve_extract_chat
    monkeypatch.setattr(mlx_llm, "mlx_available", lambda: False)
    fn, model, name = resolve_extract_chat("mlx")  # asked for mlx, unavailable
    assert name == "ollama" and fn is ollama_chat
    assert "falling back to Ollama" in capsys.readouterr().out


def test_resolve_backend_explicit_ollama_even_if_mlx_present(monkeypatch):
    from rehearse import mlx_llm
    from rehearse.markdown_extractor import resolve_extract_chat
    monkeypatch.setattr(mlx_llm, "mlx_available", lambda: True)
    _, _, name = resolve_extract_chat("ollama")
    assert name == "ollama"


def test_cache_key_includes_backend(tmp_path):
    # same text + same model, different backend => different cache file (no collision)
    calls = [0]

    def counting(messages, **kw):
        calls[0] += 1
        return ChatResult(text='{"items":[{"prompt":"P","expected_points":["a fact"]}]}',
                          ttft_s=0.1, total_s=0.2, had_think_block=False)

    md = tmp_path / "d.md"
    md.write_text("# H\n- x\n", encoding="utf-8")
    cd = tmp_path / "c"
    load_markdown(md, chat_fn=counting, backend="ollama", cache_dir=cd, write_agenda=False)
    load_markdown(md, chat_fn=counting, backend="mlx", cache_dir=cd, write_agenda=False)
    assert calls[0] == 2  # different backend -> different cache file -> re-extracted
    assert len(list(cd.glob("*.json"))) == 2


# --- load_markdown caching + agenda --------------------------------------

def test_load_caches_and_writes_agenda(tmp_path):
    calls = [0]

    def counting(messages, **kw):
        calls[0] += 1
        return ChatResult(text=json.dumps({"items": [{"prompt": "P", "expected_points": ["real point"]}]}),
                          ttft_s=0.1, total_s=0.2, had_think_block=False)

    md = tmp_path / "doc.md"
    md.write_text("# H\n- x\n", encoding="utf-8")
    cache = tmp_path / "cache"

    items1 = load_markdown(md, chat_fn=counting, cache_dir=cache, write_agenda=True)
    assert len(items1) == 1 and calls[0] == 1
    assert (tmp_path / "doc.md.recall.json").exists()  # C9 visible agenda
    agenda = json.loads((tmp_path / "doc.md.recall.json").read_text())
    assert agenda[0]["prompt"] == "P" and agenda[0]["expected_points"] == ["real point"]

    items2 = load_markdown(md, chat_fn=counting, cache_dir=cache)
    assert len(items2) == 1 and calls[0] == 1  # cache hit -> no new LLM call


# --- real extraction (opt-in, slow) --------------------------------------

@pytest.mark.skipif(not os.environ.get("REHEARSE_LLM_TESTS"),
                    reason="set REHEARSE_LLM_TESTS=1 to run real qwen3.5:4b extraction")
def test_real_extract_resume(tmp_path):
    resume = Path("/Users/niuyp/Documents/github.com/portfolio2/resume/ai-engineer.md")
    if not resume.exists():
        pytest.skip("sample resume not present")
    items = load_markdown(resume, cache_dir=tmp_path, write_agenda=False)
    assert len(items) >= 3
    assert all(it.prompt for it in items)
    assert any(it.expected_points for it in items)
    print(f"\n[extract] {len(items)} items; e.g. {items[0].prompt!r} -> {items[0].expected_points[:2]}")
