"""Unit tests for the pure bookkeeping helpers hoisted out of the (mic-only,
otherwise uncovered) English loops in main_loop.py."""

from rehearse.anki_loader import Sentence
from rehearse.main_loop import _open_store, apply_practiced
from rehearse.practice_store import PracticeStore
from rehearse.practiced_scorer import PracticeHit
from rehearse.session_seeder import PracticeStat, select_targets


def _sent(text):
    return Sentence(text=text, translation=None, native=None, deck="d", card_index=0)


def test_apply_practiced_no_hits_counts_attempts_only():
    stats: dict = {}
    keys: set = set()
    da, dh, turn_keys = apply_practiced([], 3, [_sent("hello")], stats, keys, now=100.0)
    assert (da, dh) == (3, 0)
    assert turn_keys == []
    assert stats == {} and keys == set()  # no hits -> nothing recorded


def test_apply_practiced_records_hit_under_sentence_key():
    s = _sent("It just didn't appeal to me.")
    stats: dict = {}
    keys: set = set()
    da, dh, turn_keys = apply_practiced([PracticeHit(s.text, 0.9)], 2, [s], stats, keys, now=42.0)
    assert (da, dh) == (2, 1)
    assert turn_keys == [s.key]
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
    da, dh, turn_keys = apply_practiced([PracticeHit("FREE Floating Target", 0.7)], 1, [],
                                        stats := {}, keys := set(), now=5.0)
    assert (da, dh) == (1, 1)
    assert turn_keys == ["free floating target"]
    assert "free floating target" in keys
    assert stats["free floating target"].count == 1


# --- _open_store + the per-turn persistence sequence run_loop actually uses ---

def test_open_store_no_persist_is_in_memory():
    store, stats = _open_store(None, no_persist=True)
    assert store is None and stats == {}


def test_open_store_loads_existing(tmp_path):
    db = tmp_path / "p.db"
    with PracticeStore(db) as s:
        s.record_practiced(["seed"], now=1.0)
    store, stats = _open_store(db, no_persist=False)
    try:
        assert stats["seed"].count == 1
    finally:
        store.close()


def test_open_store_closes_connection_if_load_fails(tmp_path, monkeypatch):
    # if load_stats() raises after the store opened, _open_store must close it
    # (no leaked connection) and degrade to in-memory, never crash startup.
    closed = {"n": 0}
    real_close = PracticeStore.close

    def spy_close(self):
        closed["n"] += 1
        return real_close(self)

    monkeypatch.setattr(PracticeStore, "load_stats",
                        lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(PracticeStore, "close", spy_close)
    store, stats = _open_store(tmp_path / "p.db", no_persist=False)
    assert store is None and stats == {}
    assert closed["n"] >= 1  # the opened store was closed, not leaked


def test_turn_persistence_sequence_survives_reload(tmp_path):
    # mirror run_loop's inner contract: apply_practiced -> turn_keys -> record_practiced
    db = tmp_path / "p.db"
    s = _sent("Let's grab a coffee.")
    other = _sent("See you tomorrow.")
    sentences = [s, other]

    store, stats = _open_store(db, no_persist=False)
    keys: set = set()
    now = 123.0
    _, _, turn_keys = apply_practiced([PracticeHit(s.text, 0.95)], 2, sentences,
                                      stats, keys, now)
    store.record_practiced(turn_keys, now)
    store.close()

    # "next session": persisted hit must steer select_targets to the unpracticed one
    store2, stats2 = _open_store(db, no_persist=False)
    try:
        assert stats2[s.key].count == 1 and stats2[s.key].last_ts == 123.0
        assert select_targets(sentences, stats2, n=1)[0].key == other.key
    finally:
        store2.close()
