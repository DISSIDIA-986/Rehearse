"""Tests for honest coverage scoring (cosine + anchor gate). Fake embed = no model."""

from localvocal.coverage import CoverageTracker, extract_anchors, has_substance
from localvocal.practice_item import PracticeItem


def test_extract_anchors_finds_facts_not_filler():
    a = extract_anchors("Built NPV and ROC engine over 17 years with LangChain, 99.9% uptime")
    assert "npv" in a and "roc" in a and "17" in a and "langchain" in a
    assert "built" not in a and "engine" not in a and "years" not in a


def _kw_embed(keyword):
    # [1,0] if the keyword is present (semantically "on topic"), else [0,1]
    def fn(texts):
        return [[1.0, 0.0] if keyword in t.lower() else [0.0, 1.0] for t in texts]
    return fn


def _item():
    return PracticeItem(id="q1", prompt="Tell me about the pricing engine",
                        expected_points=["Built NPV pricing engine"], section="Edianyun")


def test_hit_requires_semantic_AND_anchor():
    it = _item()
    tr = CoverageTracker([it], embed=_kw_embed("pricing"), threshold=0.55)
    cov = tr.score(it, "I built the NPV pricing engine end to end")
    assert cov.bullets[0].status == "hit"
    assert cov.complete


def test_partial_when_anchor_missing():
    it = _item()
    tr = CoverageTracker([it], embed=_kw_embed("pricing"), threshold=0.55)
    cov = tr.score(it, "I built the pricing engine")  # semantically on-topic, 'NPV' missing
    assert cov.bullets[0].status == "partial"
    assert not cov.complete


def test_miss_when_unrelated():
    it = _item()
    tr = CoverageTracker([it], embed=_kw_embed("pricing"), threshold=0.55)
    cov = tr.score(it, "I did some finance work")
    assert cov.bullets[0].status == "miss"


def test_cumulative_answer_builds_to_hit():
    it = _item()
    tr = CoverageTracker([it], embed=_kw_embed("pricing"), threshold=0.55)
    assert tr.score(it, "I built the pricing engine").bullets[0].status == "partial"
    cov = tr.score(it, "it computed NPV")  # cumulative now has pricing + NPV
    assert cov.bullets[0].status == "hit"


def test_no_anchor_bullet_scores_on_semantics_only():
    it = PracticeItem(id="q", prompt="p", expected_points=["focus on real communication"])
    tr = CoverageTracker([it], embed=_kw_embed("communication"), threshold=0.55)
    cov = tr.score(it, "I focus on real communication")  # no hard facts -> cosine alone
    assert cov.bullets[0].status == "hit"


def test_numeric_anchor_is_whole_token_not_substring():
    # "17" must NOT be credited by "2017" (final-review HIGH fix)
    it = PracticeItem(id="q", prompt="p", expected_points=["Shipped 17 models"])
    tr = CoverageTracker([it], embed=_kw_embed("models"), threshold=0.55)
    assert tr.score(it, "I shipped models in 2017").bullets[0].status == "partial"
    tr2 = CoverageTracker([it], embed=_kw_embed("models"), threshold=0.55)
    assert tr2.score(it, "I shipped 17 models").bullets[0].status == "hit"


def test_anchorless_prose_needs_lexical_overlap():
    # vague on-topic talk must NOT fully cover a prose bullet on cosine alone
    it = PracticeItem(id="q", prompt="p", expected_points=["focus on real communication"])
    tr = CoverageTracker([it], embed=_kw_embed("communication"), threshold=0.55)
    assert tr.score(it, "communication matters").bullets[0].status == "partial"
    tr2 = CoverageTracker([it], embed=_kw_embed("communication"), threshold=0.55)
    assert tr2.score(it, "I focus on real communication").bullets[0].status == "hit"


def test_coverage_is_sticky_and_cumulative_is_capped():
    from localvocal.coverage import MAX_CUM_CHARS
    it = PracticeItem(id="q", prompt="p", expected_points=["NPV"])
    tr = CoverageTracker([it], embed=lambda ts: [[1.0, 0.0] for _ in ts], threshold=0.0)
    assert tr.score(it, "NPV").bullets[0].status == "hit"
    cov = tr.score(it, "x" * (MAX_CUM_CHARS + 1000))  # truncates the anchor out
    assert len(cov.cumulative_answer) <= MAX_CUM_CHARS
    assert cov.bullets[0].status == "hit"  # sticky: a recalled point stays recalled


def test_content_free_point_is_filtered_not_cosine_credited():
    # an all-stopword "point" has nothing to recall -> never scored on cosine alone
    assert not has_substance("the and of to")
    assert has_substance("Built NPV engine") and has_substance("real communication")
    it = PracticeItem(id="q", prompt="p", expected_points=["the and of to", "real pricing work"])
    tr = CoverageTracker([it], embed=_kw_embed("pricing"), threshold=0.55)
    cov = tr.score(it, "anything vaguely on topic about pricing")
    assert len(cov.bullets) == 1  # only the substantive point is scored
    assert cov.bullets[0].bullet == "real pricing work"


def test_summary_counts():
    it = _item()
    tr = CoverageTracker([it], embed=_kw_embed("pricing"), threshold=0.55)
    tr.score(it, "I built the NPV pricing engine")
    s = tr.summary()
    assert s.total_bullets == 1 and s.hit_bullets == 1
    assert s.items_attempted == 1 and s.items_total == 1
    assert "1/1" in str(s)
