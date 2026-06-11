"""Tests for cross-session practice persistence (rehearse/practice_store.py).

Real sqlite3 against real temp files (no mocks) — including the e2e "two
sessions" scenario, corruption recovery, concurrent connections, migration, and
the nasty inputs the live loop can feed it (empty/blank/over-long keys)."""

from __future__ import annotations

import sqlite3

import pytest

from rehearse.practice_store import (
    SCHEMA_VERSION,
    PracticeStore,
    default_db_path,
)
from rehearse.session_seeder import PracticeStat, select_targets
from rehearse.anki_loader import Sentence


def _db(tmp_path):
    return tmp_path / "practice.db"


# --- first run / empty ----------------------------------------------------

def test_first_run_empty_stats(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        assert s.load_stats() == {}
        assert s.recovered_from is None


def test_memory_db_works():
    with PracticeStore(":memory:") as s:
        s.record_practiced(["k"], now=1.0)
        assert s.load_stats()["k"].count == 1


def test_creates_parent_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "practice.db"
    with PracticeStore(nested):
        assert nested.exists()


def test_default_db_path_under_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert default_db_path() == tmp_path / "rehearse" / "practice.db"


# --- record / load round-trip --------------------------------------------

def test_record_then_load(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        n = s.record_practiced(["hello there"], now=42.0)
        assert n == 1
        st = s.load_stats()["hello there"]
        assert st.count == 1 and st.last_ts == 42.0


def test_record_increments_and_overwrites_ts(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        s.record_practiced(["k"], now=10.0)
        s.record_practiced(["k"], now=20.0)
        st = s.load_stats()["k"]
        assert st.count == 2 and st.last_ts == 20.0  # count accumulates, ts = latest


def test_record_counts_each_occurrence_in_one_call(tmp_path):
    # mirrors apply_practiced: a key hit twice in one turn -> count += 2
    with PracticeStore(_db(tmp_path)) as s:
        s.record_practiced(["k", "k"], now=5.0)
        assert s.load_stats()["k"].count == 2


def test_record_empty_iterable_is_noop(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        assert s.record_practiced([], now=1.0) == 0
        assert s.load_stats() == {}


# --- nasty keys -----------------------------------------------------------

def test_blank_and_empty_keys_skipped(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        written = s.record_practiced(["", "   ", "\t\n", "real"], now=1.0)
        assert written == 1
        assert set(s.load_stats()) == {"real"}


def test_long_key_round_trips_verbatim(tmp_path):
    # long input must be handled (not crash) AND kept identical to the in-memory
    # stats — truncating here would silently diverge from apply_practiced's key.
    long_key = "x" * 5000
    with PracticeStore(_db(tmp_path)) as s:
        s.record_practiced([long_key], now=1.0)
        keys = list(s.load_stats())
        assert keys == [long_key]


def test_key_stored_verbatim_for_parity(tmp_path):
    # keys are stored EXACTLY as given so they match apply_practiced's in-memory
    # key char-for-char; only a fully-blank key is dropped (never transformed).
    with PracticeStore(_db(tmp_path)) as s:
        s.record_practiced(["  spaced key  ", "normal"], now=1.0)
        loaded = set(s.load_stats())
        assert "  spaced key  " in loaded and "normal" in loaded


def test_unicode_key_round_trips(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        s.record_practiced(["café 日本語 ✓"], now=1.0)
        assert "café 日本語 ✓" in s.load_stats()


# --- e2e: two sessions, real files ---------------------------------------

def test_persists_across_sessions(tmp_path):
    path = _db(tmp_path)
    # session 1
    with PracticeStore(path) as s1:
        s1.record_practiced(["it just didn't appeal to me"], now=100.0)
    # session 2: a brand-new process/connection sees session 1's history
    with PracticeStore(path) as s2:
        stats = s2.load_stats()
        assert stats["it just didn't appeal to me"].count == 1
        s2.record_practiced(["it just didn't appeal to me"], now=200.0)
    with PracticeStore(path) as s3:
        st = s3.load_stats()["it just didn't appeal to me"]
        assert st.count == 2 and st.last_ts == 200.0


def test_loaded_stats_drive_select_targets(tmp_path):
    # the whole point: persisted counts make select_targets surface least-practiced
    path = _db(tmp_path)
    a = Sentence("Alpha one.", None, None, "d", 0)
    b = Sentence("Bravo two.", None, None, "d", 1)
    with PracticeStore(path) as s:
        s.record_practiced([a.key], now=1.0)  # 'a' practiced, 'b' never
    with PracticeStore(path) as s:
        stats = s.load_stats()
    picked = select_targets([a, b], stats, n=1)
    assert picked[0].key == b.key  # least-practiced (count 0) comes first


# --- corruption recovery --------------------------------------------------

def test_corrupt_db_is_quarantined_and_recreated(tmp_path):
    path = _db(tmp_path)
    path.write_bytes(b"this is not a sqlite database at all, just junk bytes")
    s = PracticeStore(path)  # must NOT raise
    try:
        assert s.recovered_from is not None and s.recovered_from.exists()
        assert s.load_stats() == {}  # fresh, usable DB
        s.record_practiced(["k"], now=1.0)
        assert s.load_stats()["k"].count == 1
    finally:
        s.close()


def test_wrong_schema_db_is_quarantined(tmp_path):
    # a valid SQLite file whose sentence_stat lacks our columns (foreign/old DB)
    # passes integrity_check but would break load_stats -> must be quarantined
    path = _db(tmp_path)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE sentence_stat (id INTEGER PRIMARY KEY, note TEXT)")
    con.execute("INSERT INTO sentence_stat (note) VALUES ('not our schema')")
    con.commit(); con.close()
    with PracticeStore(path) as s:
        assert s.recovered_from is not None and s.recovered_from.exists()
        assert s.load_stats() == {}  # fresh, correct schema
        s.record_practiced(["k"], now=1.0)
        assert s.load_stats()["k"].count == 1


def test_quarantine_removes_stale_wal_sidecars(tmp_path):
    path = _db(tmp_path)
    path.write_bytes(b"junk, not sqlite")
    # simulate leftover WAL sidecars from the broken DB
    (tmp_path / "practice.db-wal").write_bytes(b"stale wal content marker")
    (tmp_path / "practice.db-shm").write_bytes(b"stale shm content marker")
    with PracticeStore(path) as s:
        assert s.recovered_from is not None
        s.record_practiced(["k"], now=1.0)
        assert "k" in s.load_stats()  # not corrupted by the stale sidecars
        # the NEW db opens its own WAL; what must be gone is the STALE content
        # (the old sidecars were unlinked before the fresh connection).
        wal = tmp_path / "practice.db-wal"
        if wal.exists():
            assert wal.read_bytes()[:9] != b"stale wal"


def test_locked_writer_fails_fast_not_hangs(tmp_path):
    # a competing EXCLUSIVE writer must make our write raise quickly (busy_timeout
    # ~250ms) so the loop's try/except degrades to a warning, NOT a 5s stall
    import time as _t
    path = _db(tmp_path)
    holder = PracticeStore(path)
    blocker = sqlite3.connect(str(path))
    try:
        holder.record_practiced(["seed"], now=1.0)  # ensure DB + WAL exist
        blocker.execute("BEGIN EXCLUSIVE")  # hold a write lock
        t0 = _t.monotonic()
        with pytest.raises(sqlite3.OperationalError):
            holder.record_practiced(["blocked"], now=2.0)
        assert _t.monotonic() - t0 < 2.0  # bounded by busy_timeout, nowhere near 5s
    finally:
        blocker.rollback(); blocker.close(); holder.close()


def test_busy_timeout_is_short(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        assert s._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 250


def test_truncated_db_recovered(tmp_path):
    # a real DB header followed by garbage (interrupted write) is also unhealthy
    path = _db(tmp_path)
    PracticeStore(path).close()  # make a valid DB
    data = path.read_bytes()
    path.write_bytes(data[:len(data) // 2] + b"\x00\xff" * 50)  # corrupt the tail
    s = PracticeStore(path)
    try:
        # either it was caught as corrupt (quarantined) or integrity_check passed;
        # in BOTH cases the store must be usable, never crash
        s.record_practiced(["k"], now=1.0)
        assert "k" in s.load_stats()
    finally:
        s.close()


# --- concurrency (WAL: a second process) ---------------------------------

def test_two_connections_see_each_others_commits(tmp_path):
    path = _db(tmp_path)
    a = PracticeStore(path)
    b = PracticeStore(path)
    try:
        a.record_practiced(["from_a"], now=1.0)
        assert "from_a" in b.load_stats()  # b (separate connection) sees a's commit
        b.record_practiced(["from_b"], now=2.0)
        assert set(a.load_stats()) == {"from_a", "from_b"}
    finally:
        a.close(); b.close()


# --- schema / migration ---------------------------------------------------

def test_user_version_is_set(tmp_path):
    PracticeStore(_db(tmp_path)).close()
    con = sqlite3.connect(str(_db(tmp_path)))
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        con.close()


def test_versionless_db_with_data_is_upgraded_not_wiped(tmp_path):
    # simulate a pre-versioning DB: correct table, user_version still 0, has a row
    path = _db(tmp_path)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE sentence_stat (key TEXT PRIMARY KEY, "
                "count INTEGER NOT NULL DEFAULT 0, last_ts REAL NOT NULL DEFAULT 0)")
    con.execute("INSERT INTO sentence_stat VALUES ('legacy', 3, 7.0)")
    con.commit(); con.close()
    with PracticeStore(path) as s:
        st = s.load_stats()["legacy"]
        assert st.count == 3 and st.last_ts == 7.0  # data preserved
    con = sqlite3.connect(str(path))
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        con.close()


# --- v2: unaided shadow signal (T-P2-2b) ----------------------------------

def test_migration_v1_to_v2_adds_columns_preserves_data(tmp_path):
    # a real v1 DB (base table, user_version=1, has data) must upgrade in place
    path = _db(tmp_path)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE sentence_stat (key TEXT PRIMARY KEY, "
                "count INTEGER NOT NULL DEFAULT 0, last_ts REAL NOT NULL DEFAULT 0)")
    con.execute("INSERT INTO sentence_stat VALUES ('old', 4, 9.0)")
    con.execute("PRAGMA user_version = 1")
    con.commit(); con.close()
    with PracticeStore(path) as s:
        st = s.load_stats()["old"]
        assert (st.count, st.last_ts, st.unaided_count) == (4, 9.0, 0)  # preserved + defaulted
        s.record_unaided("old", 0.7, now=10.0)
        assert s.load_stats()["old"].unaided_count == 1  # v2 columns usable
    con = sqlite3.connect(str(path))
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 2
        cols = {r[1] for r in con.execute("PRAGMA table_info(sentence_stat)")}
        assert {"unaided_count", "unaided_last_ts"} <= cols
    finally:
        con.close()


def test_record_unaided_increments_and_logs_without_touching_schedule(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        s.record_practiced(["k"], now=1.0)  # count=1, last_ts=1.0
        assert s.record_unaided("k", 0.83, now=5.0) is True
        st = s.load_stats()["k"]
        assert st.unaided_count == 1
        # CRITICAL: scheduling fields are untouched -> can't poison select_targets
        assert st.count == 1 and st.last_ts == 1.0
        assert s.unaided_events() == [(5.0, "k", 0.83)]


def test_record_unaided_blank_key_is_noop(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        assert s.record_unaided("   ", 0.9, now=1.0) is False
        assert s.unaided_events() == []


def test_record_unaided_unknown_key_creates_row(tmp_path):
    # defensive: even if the key has no prior row, record (count stays 0)
    with PracticeStore(_db(tmp_path)) as s:
        s.record_unaided("fresh", 0.7, now=2.0)
        st = s.load_stats()["fresh"]
        assert st.unaided_count == 1 and st.count == 0


def test_load_stats_defaults_unaided_to_zero(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        s.record_practiced(["k"], now=1.0)
        assert s.load_stats()["k"].unaided_count == 0


def test_record_unaided_accumulates(tmp_path):
    with PracticeStore(_db(tmp_path)) as s:
        s.record_unaided("k", 0.7, now=1.0)
        s.record_unaided("k", 0.8, now=2.0)
        assert s.load_stats()["k"].unaided_count == 2
        assert len(s.unaided_events()) == 2


def test_migration_repairs_v2_stamped_db_missing_objects(tmp_path):
    # a DB stamped user_version=2 but LACKING the v2 columns/event table (partial
    # or aborted migration) must self-repair on open, not crash load_stats
    path = _db(tmp_path)
    con = sqlite3.connect(str(path))
    con.execute("CREATE TABLE sentence_stat (key TEXT PRIMARY KEY, "
                "count INTEGER NOT NULL DEFAULT 0, last_ts REAL NOT NULL DEFAULT 0)")
    con.execute("INSERT INTO sentence_stat VALUES ('k', 2, 5.0)")
    con.execute("PRAGMA user_version = 2")  # stamped v2 but v2 objects absent
    con.commit(); con.close()
    with PracticeStore(path) as s:
        st = s.load_stats()["k"]  # must not raise on missing unaided_count
        assert st.count == 2 and st.unaided_count == 0
        assert s.unaided_events() == []  # event table created
        s.record_unaided("k", 0.7, now=1.0)
        assert s.load_stats()["k"].unaided_count == 1


def test_reopening_v2_db_is_idempotent(tmp_path):
    # an already-v2 DB must reopen cleanly: no duplicate ALTER, event table intact,
    # data + version preserved (migration runs but is a no-op past v2)
    path = _db(tmp_path)
    with PracticeStore(path) as s:
        s.record_practiced(["k"], now=1.0)
        s.record_unaided("k", 0.9, now=2.0)
    with PracticeStore(path) as s:  # reopen — must not raise or lose data
        st = s.load_stats()["k"]
        assert st.count == 1 and st.unaided_count == 1
        assert len(s.unaided_events()) == 1
        s.record_unaided("k", 0.8, now=3.0)
        assert s.load_stats()["k"].unaided_count == 2
    con = sqlite3.connect(str(path))
    try:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        con.close()


# --- lifecycle ------------------------------------------------------------

def test_close_is_idempotent(tmp_path):
    s = PracticeStore(_db(tmp_path))
    s.close()
    s.close()  # second close must not raise
