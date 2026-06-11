"""Unit tests for the pure bookkeeping helpers hoisted out of the (mic-only,
otherwise uncovered) English loops in main_loop.py."""

from rehearse.anki_loader import Sentence
from rehearse.main_loop import apply_practiced
from rehearse.practiced_scorer import PracticeHit
from rehearse.session_seeder import PracticeStat


def _sent(text):
    return Sentence(text=text, translation=None, native=None, deck="d", card_index=0)


def test_apply_practiced_no_hits_counts_attempts_only():
    stats: dict = {}
    keys: set = set()
    da, dh = apply_practiced([], 3, [_sent("hello")], stats, keys, now=100.0)
    assert (da, dh) == (3, 0)
    assert stats == {} and keys == set()  # no hits -> nothing recorded


def test_apply_practiced_records_hit_under_sentence_key():
    s = _sent("It just didn't appeal to me.")
    stats: dict = {}
    keys: set = set()
    da, dh = apply_practiced([PracticeHit(s.text, 0.9)], 2, [s], stats, keys, now=42.0)
    assert (da, dh) == (2, 1)
    assert keys == {s.key}
    assert stats[s.key].count == 1 and stats[s.key].last_ts == 42.0


def test_apply_practiced_accumulates_across_turns():
    s = _sent("How are you?")
    stats = {s.key: PracticeStat(count=1, last_ts=1.0)}
    keys = {s.key}
    apply_practiced([PracticeHit(s.text, 0.8)], 1, [s], stats, keys, now=99.0)
    assert stats[s.key].count == 2 and stats[s.key].last_ts == 99.0


def test_apply_practiced_unmatched_target_uses_lowercased_key():
    # a hit whose target isn't in `sentences` falls back to target.lower()
    da, dh = apply_practiced([PracticeHit("FREE Floating Target", 0.7)], 1, [],
                             stats := {}, keys := set(), now=5.0)
    assert (da, dh) == (1, 1)
    assert "free floating target" in keys
    assert stats["free floating target"].count == 1
