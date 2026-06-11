"""Fast-mode extraction (no LLM): instant regex parser for well-structured docs."""

import pytest

from rehearse.llm_client import ChatResult
from rehearse.markdown_extractor import (
    extract_items_fast,
    load_markdown,
    resolve_extract_chat,
)


def test_extract_items_fast_no_llm():
    md = ("## Risk control\n- 3-stage scoring\n- pricing engine\n\n"
          "## Onboarding\n- Verified ID\n- KYC checks")
    items = extract_items_fast(md)
    assert [it.section for it in items] == ["Risk control", "Onboarding"]
    assert items[0].expected_points == ["3-stage scoring", "pricing engine"]
    assert items[1].expected_points == ["Verified ID", "KYC checks"]


def test_extract_items_fast_drops_content_free():
    # all-stopword "points" still get filtered (same substance gate as the LLM path)
    items = extract_items_fast("## H\n- the and of to\n- real pricing fact")
    assert len(items) == 1 and items[0].expected_points == ["real pricing fact"]


def test_resolve_extract_chat_rejects_fast():
    # fast has no chat_fn — load_markdown handles it directly without resolving
    with pytest.raises(ValueError):
        resolve_extract_chat("fast")


def test_load_markdown_fast_skips_chat_and_uses_separate_cache(tmp_path):
    md = tmp_path / "d.md"
    md.write_text("## Section\n- a real fact\n- another fact\n", encoding="utf-8")
    cd = tmp_path / "c"

    # fast: no chat_fn provided — would crash if load_markdown tried to call one
    items_fast = load_markdown(md, backend="fast", cache_dir=cd, write_agenda=False)
    assert items_fast and items_fast[0].section == "Section"

    # ollama path: different backend in cache key -> separate cache file (no collision)
    def fake(messages, **kw):
        return ChatResult(text='{"items":[{"prompt":"P","expected_points":["a genuine recalled fact"]}]}',
                          ttft_s=0.1, total_s=0.2, had_think_block=False)
    load_markdown(md, chat_fn=fake, backend="ollama", cache_dir=cd, write_agenda=False)
    assert len(list(cd.glob("*.json"))) == 2  # one fast cache + one ollama cache


def test_load_markdown_fast_caches_on_second_call(tmp_path):
    md = tmp_path / "d.md"
    md.write_text("## Heading\n- first real point\n- second real point\n", encoding="utf-8")
    cd = tmp_path / "c"
    items1 = load_markdown(md, backend="fast", cache_dir=cd, write_agenda=False)
    items2 = load_markdown(md, backend="fast", cache_dir=cd, write_agenda=False)
    assert [it.prompt for it in items1] == [it.prompt for it in items2]
    assert len(list(cd.glob("*.json"))) == 1  # same key, single cache file
