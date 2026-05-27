from rehearse.anki_loader import Sentence
from rehearse.session_seeder import PracticeStat, select_targets


def _s(text):
    return Sentence(text=text, translation=None, native=None, deck="d", card_index=0)


def test_empty():
    assert select_targets([], {}, n=3) == []


def test_clamps_n_to_available():
    s = [_s("a"), _s("b")]
    assert len(select_targets(s, {}, n=5)) == 2


def test_least_practiced_first():
    a, b, c = _s("a"), _s("b"), _s("c")
    stats = {
        "a": PracticeStat(count=3, last_ts=100),
        "b": PracticeStat(count=0, last_ts=0),   # never practiced -> first
        "c": PracticeStat(count=1, last_ts=50),
    }
    picked = select_targets([a, b, c], stats, n=2)
    assert [p.text for p in picked] == ["b", "c"]


def test_same_count_orders_by_oldest():
    a, b = _s("a"), _s("b")
    stats = {
        "a": PracticeStat(count=1, last_ts=200),
        "b": PracticeStat(count=1, last_ts=100),  # older -> first
    }
    assert [p.text for p in select_targets([a, b], stats, n=1)] == ["b"]


def test_n_zero_or_negative_returns_empty():
    s = [_s("a"), _s("b")]
    assert select_targets(s, {}, n=0) == []
    assert select_targets(s, {}, n=-3) == []


def test_unknown_sentences_treated_as_unpracticed():
    a = _s("a")
    # no stats at all -> still returns
    assert select_targets([a], None, n=3) == [a]
