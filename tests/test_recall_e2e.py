"""End-to-end composition: markdown -> extractor -> CoverageTracker -> RecallSession.
All deterministic (fake LLM + fake bag-of-words embed); no real models, no mic."""

import re
from types import SimpleNamespace

from localvocal.coverage import CoverageTracker
from localvocal.markdown_extractor import load_markdown
from localvocal.recall_session import RecallSession

_MD = """\
## Pricing
- Built NPV pricing engine
- Used ACPI models

## Risk
- Kafka streaming risk control
"""

_VOCAB = ["npv", "pricing", "engine", "acpi", "models", "kafka", "streaming", "risk", "control", "built", "used"]


def _fake_embed(texts):
    out = []
    for t in texts:
        toks = set(re.findall(r"[a-z]+", t.lower()))
        out.append([1.0 if w in toks else 0.0 for w in _VOCAB])
    return out


def _fake_chat(messages, *, model=None, num_predict=None, temperature=None, fmt=None):
    # mimic the LLM honoring the JSON schema: one item per heading
    payload = (
        '{"items":['
        '{"prompt":"Tell me about pricing","expected_points":["Built NPV pricing engine","Used ACPI models"],"section":"Pricing"},'
        '{"prompt":"How did risk control work?","expected_points":["Kafka streaming risk control"],"section":"Risk"}'
        ']}'
    )
    return SimpleNamespace(text=payload, ttft_s=0.0)


def test_extract_to_session_full_recall(tmp_path):
    md = tmp_path / "resume.md"
    md.write_text(_MD, encoding="utf-8")

    items = load_markdown(md, model="fake", chat_fn=_fake_chat,
                          cache_dir=tmp_path / "cache")
    assert len(items) == 2
    assert items[0].section == "Pricing" and len(items[0].expected_points) == 2

    # the visible agenda was written next to the doc (C9) and never contains a cache key
    agenda = md.with_name("resume.md.recall.json")
    assert agenda.exists()

    session = RecallSession(items, tracker=CoverageTracker(items, embed=_fake_embed),
                            source_title="resume.md")
    assert session.progress() == (1, 2)
    assert "pricing" in session.opening_line().lower()

    # answer item 0 fully (both bullets' words + their NPV/ACPI anchors)
    session.coach_prompt()
    out = session.record("I built the NPV pricing engine and used ACPI models")
    assert out.coverage.complete and out.advanced
    assert session.current.section == "Risk"

    # answer item 1 fully -> session done
    session.coach_prompt()
    out = session.record("It was Kafka streaming risk control")
    assert out.coverage.complete and session.done

    summ = session.summary()
    assert summ.hit_bullets == 3 and summ.total_bullets == 3


def test_partial_answer_does_not_advance(tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(_MD, encoding="utf-8")
    items = load_markdown(md, model="fake", chat_fn=_fake_chat, cache_dir=tmp_path / "c")
    session = RecallSession(items, tracker=CoverageTracker(items, embed=_fake_embed))
    session.opening_line()
    session.coach_prompt()
    # only one of two bullets -> not complete, stays on item 0
    out = session.record("I built the NPV pricing engine")
    assert not out.coverage.complete and not out.advanced
    assert session.current.section == "Pricing"
