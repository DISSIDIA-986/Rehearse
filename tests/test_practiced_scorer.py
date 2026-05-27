import pytest

from localvocal.practiced_scorer import (
    PracticeHit,
    cosine,
    ollama_embed,
    score_practiced,
)


def test_cosine_basics():
    assert cosine([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
    assert cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
    assert cosine([1, 0], [1, 0, 0]) == 0.0  # length mismatch
    assert cosine([0, 0], [0, 0]) == 0.0  # zero vector
    assert cosine([], [1]) == 0.0


_TABLE = {
    "I have no interest in it": [1.0, 0.0, 0.0],
    "no interest at all": [0.96, 0.05, 0.0],   # paraphrase -> high sim
    "the weather is nice": [0.0, 1.0, 0.0],    # unrelated
}


def _fake_embed(texts):
    return [_TABLE.get(t, [0.0, 0.0, 1.0]) for t in texts]


def test_score_hit_above_threshold():
    hits = score_practiced(
        "I have no interest in it",
        ["no interest at all", "the weather is nice"],
        _fake_embed,
        threshold=0.8,
    )
    assert len(hits) == 1
    assert hits[0].target == "no interest at all"
    assert hits[0].similarity > 0.9


def test_score_orders_best_first():
    hits = score_practiced(
        "I have no interest in it",
        ["I have no interest in it", "no interest at all"],
        _fake_embed,
        threshold=0.5,
    )
    assert [h.target for h in hits][0] == "I have no interest in it"
    assert hits[0].similarity >= hits[1].similarity


def test_score_empty_inputs():
    assert score_practiced("", ["x"], _fake_embed) == []
    assert score_practiced("x", [], _fake_embed) == []


def test_score_raises_on_bad_embedder_shape():
    # embedder returns too few vectors -> explicit error, not a silent zip drop
    bad = lambda texts: [[1.0, 0.0]]  # noqa: E731  (always 1 vector)
    with pytest.raises(ValueError):
        score_practiced("hi", ["a", "b"], bad)


def test_cosine_handles_nan_inf():
    assert cosine([float("nan"), 1.0], [1.0, 1.0]) == 0.0
    assert cosine([float("inf"), 0.0], [1.0, 0.0]) == 0.0


def test_default_threshold_calibrated():
    from localvocal.practiced_scorer import DEFAULT_THRESHOLD
    # calibrated to the 0.454 (unrelated) .. 0.548 (paraphrase) gap
    assert 0.45 < DEFAULT_THRESHOLD < 0.55


# --- real nomic-embed integration (skipped if Ollama unreachable) ---------

def _ollama_up():
    try:
        ollama_embed(["ping"])
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama / nomic-embed-text not reachable")
def test_real_nomic_embed_roundtrip():
    same = score_practiced(
        "I have no interest in it",
        ["I have no interest in it", "the stock market crashed today"],
        ollama_embed,
        threshold=0.6,
    )
    assert same and same[0].target == "I have no interest in it"
