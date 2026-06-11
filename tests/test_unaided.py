"""Tests for shadow-mode unaided-production detection (rehearse/unaided.py).

Pure logic, injected fake embedder (controls cosine exactly) — no models, no mic."""

from __future__ import annotations

import math

import pytest

from rehearse.anki_loader import Sentence
from rehearse.session_seeder import PracticeStat
from rehearse.unaided import (
    MAX_CANDIDATES,
    UNAIDED_THRESHOLD,
    detect_unaided,
    select_candidates,
)


def _sent(text, idx=0):
    return Sentence(text=text, translation=None, native=None, deck="d", card_index=idx)


# --- select_candidates ----------------------------------------------------

def test_candidates_only_previously_practiced():
    a, b, c = _sent("Alpha.", 0), _sent("Bravo.", 1), _sent("Charlie.", 2)
    stats = {a.key: PracticeStat(count=2), b.key: PracticeStat(count=0)}  # c absent
    out = select_candidates(stats, [a, b, c], active_keys=set())
    assert [s.key for s in out] == [a.key]  # only count>0 (b is 0, c unknown)


def test_candidates_exclude_active_targets():
    a, b = _sent("Alpha.", 0), _sent("Bravo.", 1)
    stats = {a.key: PracticeStat(count=1), b.key: PracticeStat(count=1)}
    out = select_candidates(stats, [a, b], active_keys={a.key})
    assert [s.key for s in out] == [b.key]  # a is this turn's target -> not "unaided"


def test_candidates_due_order_least_practiced_first():
    a, b, c = _sent("A.", 0), _sent("B.", 1), _sent("C.", 2)
    stats = {a.key: PracticeStat(count=3, last_ts=5.0),
             b.key: PracticeStat(count=1, last_ts=9.0),
             c.key: PracticeStat(count=1, last_ts=2.0)}
    out = select_candidates(stats, [a, b, c], active_keys=set())
    # count asc, then last_ts asc -> c(1,2.0), b(1,9.0), a(3,5.0)
    assert [s.key for s in out] == [c.key, b.key, a.key]


def test_candidates_capped():
    sents = [_sent(f"S{i}.", i) for i in range(100)]
    stats = {s.key: PracticeStat(count=1, last_ts=float(i)) for i, s in enumerate(sents)}
    out = select_candidates(stats, sents, active_keys=set(), cap=10)
    assert len(out) == 10


def test_candidates_empty_when_nothing_practiced():
    a = _sent("Alpha.", 0)
    assert select_candidates({}, [a], active_keys=set()) == []  # fresh user -> no cost


# --- detect_unaided -------------------------------------------------------

def _unit(*xs):
    v = list(xs)
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class _Embed:
    """Fake embedder: returns the vector mapped to each text; counts calls."""

    def __init__(self, table):
        self.table = table
        self.embedded: list[str] = []

    def __call__(self, texts):
        self.embedded += list(texts)
        return [self.table[t] for t in texts]


def test_detect_empty_user_text_is_none():
    c = _sent("Alpha.")
    assert detect_unaided("", [c], _Embed({})) is None
    assert detect_unaided("   ", [c], _Embed({})) is None


def test_detect_no_candidates_is_none():
    assert detect_unaided("hello", [], _Embed({})) is None


def test_detect_below_threshold_is_none():
    c = _sent("Alpha.")
    # user orthogonal-ish to candidate -> cosine 0 < threshold
    emb = _Embed({"user said": _unit(1, 0), c.text: _unit(0, 1)})
    assert detect_unaided("user said", [c], emb) is None


def test_detect_at_threshold_is_hit():
    c = _sent("Alpha.")
    # craft cosine exactly == UNAIDED_THRESHOLD: user=(1,0); cand=(t, sqrt(1-t^2))
    t = UNAIDED_THRESHOLD
    emb = _Embed({"u": _unit(1, 0), c.text: [t, math.sqrt(1 - t * t)]})
    hit = detect_unaided("u", [c], emb)
    assert hit is not None and hit.key == c.key
    assert hit.similarity == pytest.approx(t, abs=1e-6)


def test_detect_returns_single_best_hit():
    c1, c2 = _sent("One.", 1), _sent("Two.", 2)
    # both above threshold; c2 closer to user -> only c2 returned (one per turn)
    emb = _Embed({"u": _unit(1, 0),
                  c1.text: _unit(0.9, 0.2),
                  c2.text: _unit(0.99, 0.05)})
    hit = detect_unaided("u", [c1, c2], emb)
    assert hit.key == c2.key


def test_detect_caches_candidate_embeddings_across_turns():
    c = _sent("Alpha.")
    emb = _Embed({"u1": _unit(1, 0), "u2": _unit(1, 0), c.text: _unit(1, 0)})
    cache: dict = {}
    detect_unaided("u1", [c], emb, cache=cache)
    detect_unaided("u2", [c], emb, cache=cache)
    # candidate embedded ONCE (cached); user embedded each turn
    assert emb.embedded.count(c.text) == 1
    assert emb.embedded.count("u1") == 1 and emb.embedded.count("u2") == 1
