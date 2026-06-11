"""Cross-session practice persistence (Phase 2, T-P2-2a) — SQLite spaced-rep state.

The live English loop scores 'practiced' hits per turn (D3), but v1 kept the
stats in memory, so "conversational spaced repetition" reset every session.
This store persists each sentence's practice count + last-practiced time to a
small SQLite DB (stdlib `sqlite3`, zero new deps) so `select_targets()` surfaces
genuinely least-practiced sentences ACROSS sessions — the project's core value
prop, see docs/DESIGN.md "What Makes This Cool".

Scope (2a): persist the existing prompted-hit stats only. The harder "unaided
production" mastery signal (T-P2-2b) is deferred behind a stricter gate so it
can't poison the schedule before it's proven.

Pure storage: `session_seeder` / `select_targets` / `apply_practiced` are
unchanged. It wires in at main_loop's edge — `load_stats()` at startup instead
of `{}`, `record_practiced()` after each scored turn. Fully headless-testable
(inject a tmp path or ':memory:'); never crashes a live turn (a corrupt DB is
quarantined and recreated, write errors degrade to a warning).
"""

from __future__ import annotations

import os
import sqlite3
import time as _time
from collections.abc import Iterable
from pathlib import Path

from rehearse.session_seeder import PracticeStat

SCHEMA_VERSION = 2  # v2: + unaided_count/unaided_last_ts + practice_event (T-P2-2b shadow)
_COLUMNS = {"key", "count", "last_ts"}  # the BASE columns _healthy requires (v2 adds more)


def default_db_path() -> Path:
    """Durable user-state location (XDG data dir, NOT the cache dir — practice
    history must survive a cache wipe)."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "rehearse" / "practice.db"


class PracticeStore:
    """SQLite-backed per-sentence practice stats. One connection, used only from
    the main thread (the live loop never touches it from the audio callback)."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = default_db_path()
        self._memory = str(db_path) == ":memory:"
        self.path = db_path if self._memory else Path(db_path)
        self.recovered_from: Path | None = None  # set if a corrupt DB was quarantined
        if not self._memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and not _healthy(self.path):
                # Quarantine, don't delete: keep the bad file for forensics, never
                # crash the session over it (fail-closed -> fresh DB). Also drop the
                # stale -wal/-shm sidecars, which would otherwise corrupt the fresh DB.
                self.recovered_from = self.path.with_name(
                    f"{self.path.name}.corrupt-{int(_time.time())}")
                self.path.rename(self.recovered_from)
                for ext in ("-wal", "-shm"):
                    side = self.path.with_name(self.path.name + ext)
                    if side.exists():
                        side.unlink()
        self._open_and_migrate(":memory:" if self._memory else str(self.path))

    def _open_and_migrate(self, target: str) -> None:
        # Two `rehearse` instances opening the SAME fresh DB at once race on WAL
        # journal-mode init + schema DDL. The journal-mode switch can raise
        # "database is locked" (busy_timeout does NOT cover it), and the migration's
        # check-then-ALTER can raise "duplicate column name". Both are TRANSIENT:
        # once the winner finishes (sub-ms DDL), WAL is already set and the columns
        # exist, so a retry is a near-no-op. Bounded retry so BOTH sessions keep
        # persistence instead of one degrading to in-memory. :memory: never contends.
        last: Exception | None = None
        for attempt in range(8):
            try:
                self._conn = _connect(target)
                self._migrate()
                return
            except sqlite3.OperationalError as e:
                msg = str(e).lower()
                if "locked" not in msg and "duplicate column" not in msg:
                    raise  # a real schema/IO error — let the caller quarantine/degrade
                last = e
                try:
                    self._conn.close()
                except Exception:
                    pass
                _time.sleep(0.02 * (attempt + 1))  # ~0.02..0.16s backoff, ~0.7s total
        raise last  # exhausted retries — caller (_open_store) degrades to in-memory

    # --- schema ----------------------------------------------------------
    def _migrate(self) -> None:
        # Ensure the FULL current schema exists, idempotently and regardless of
        # user_version — so a DB stamped v2 but missing v2 objects (partial/aborted
        # migration, hand-edited file) is repaired here, not left to crash load_stats.
        ver = self._conn.execute("PRAGMA user_version").fetchone()[0]
        self._conn.execute(  # v1 base table
            "CREATE TABLE IF NOT EXISTS sentence_stat ("
            "  key TEXT PRIMARY KEY,"
            "  count INTEGER NOT NULL DEFAULT 0,"
            "  last_ts REAL NOT NULL DEFAULT 0)"
        )
        # v2: T-P2-2b shadow columns + event log (additive, data-preserving). Guarded
        # by column existence, NOT by user_version, so it self-repairs.
        def _add_col(ddl: str) -> None:
            # The (check cols) -> (ALTER) pair is not atomic across processes: two
            # `rehearse` instances opening a fresh/v1 DB at once can both see the
            # column missing, and the loser's ALTER raises "duplicate column name".
            # Swallow ONLY that (the column now exists, which is the goal) so a
            # concurrent open self-heals instead of degrading to in-memory.
            try:
                self._conn.execute(ddl)
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise

        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(sentence_stat)")}
        if "unaided_count" not in cols:
            _add_col("ALTER TABLE sentence_stat ADD COLUMN "
                     "unaided_count INTEGER NOT NULL DEFAULT 0")
        if "unaided_last_ts" not in cols:
            _add_col("ALTER TABLE sentence_stat ADD COLUMN "
                     "unaided_last_ts REAL NOT NULL DEFAULT 0")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS practice_event ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts REAL NOT NULL,"
            "  key TEXT NOT NULL,"
            "  kind TEXT NOT NULL,"        # 'unaided' (room for 'prompted' later)
            "  similarity REAL)"
        )
        if ver < SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    # --- reads -----------------------------------------------------------
    def load_stats(self) -> dict[str, PracticeStat]:
        """Every sentence's persisted (count, last_ts, unaided_count), keyed by
        sentence.key — drop-in for the in-memory `stats` the loop started empty.
        unaided_count is shadow data; select_targets ignores it (for now)."""
        rows = self._conn.execute(
            "SELECT key, count, last_ts, unaided_count FROM sentence_stat").fetchall()
        return {k: PracticeStat(count=c, last_ts=t, unaided_count=u)
                for k, c, t, u in rows}

    # --- writes ----------------------------------------------------------
    def record_practiced(self, keys: Iterable[str], now: float) -> int:
        """+1 count and last_ts=now per key occurrence, in ONE transaction.
        Mirrors `apply_practiced`'s in-memory mutation so memory and disk agree —
        keys are stored VERBATIM (no strip/truncate; any transform here would
        diverge from the exact key apply_practiced put in `stats`). A fully-blank
        key (which the scorer never produces) is skipped. Returns rows written."""
        clean = [k for k in keys if k and k.strip()]
        if not clean:
            return 0
        with self._conn:  # atomic: all hits this turn commit together or not at all
            self._conn.executemany(
                "INSERT INTO sentence_stat(key, count, last_ts) VALUES(?, 1, ?) "
                "ON CONFLICT(key) DO UPDATE SET count = count + 1, last_ts = excluded.last_ts",
                [(k, now) for k in clean],
            )
        return len(clean)

    def record_unaided(self, key: str, similarity: float, now: float) -> bool:
        """SHADOW signal (T-P2-2b): the user spontaneously produced `key` without
        it being a steered target this turn. Bumps unaided_count + logs an event
        for later calibration. Does NOT touch `count`/`last_ts`, so select_targets
        scheduling is untouched (a false positive can't poison the schedule).
        Returns False (no-op) for a blank key. One transaction."""
        key = (key or "").strip()
        if not key:
            return False
        with self._conn:
            self._conn.execute(
                "INSERT INTO sentence_stat(key, count, last_ts, unaided_count, unaided_last_ts) "
                "VALUES(?, 0, 0, 1, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "  unaided_count = unaided_count + 1, unaided_last_ts = excluded.unaided_last_ts",
                (key, now),
            )
            self._conn.execute(
                "INSERT INTO practice_event(ts, key, kind, similarity) VALUES(?, ?, 'unaided', ?)",
                (now, key, similarity),
            )
        return True

    def unaided_events(self) -> list[tuple]:
        """All logged unaided events (ts, key, similarity), oldest first — the
        calibration audit trail. Read-only helper for an offline review tool."""
        return self._conn.execute(
            "SELECT ts, key, similarity FROM practice_event "
            "WHERE kind='unaided' ORDER BY id").fetchall()

    # --- lifecycle -------------------------------------------------------
    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "PracticeStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=5.0)
    # WAL = a second `rehearse` process reads/writes without "database is locked";
    # busy_timeout lets a brief lock resolve instead of erroring instantly.
    conn.execute("PRAGMA journal_mode=WAL")
    # Short on purpose: a write is on the per-turn path, so a (rare) competing
    # writer must degrade to a "not saved" warning fast, NOT stall the loop ~5s.
    conn.execute("PRAGMA busy_timeout=250")
    return conn


def _healthy(path: Path) -> bool:
    """True if `path` is a readable, non-corrupt SQLite DB WITH our schema. A
    junk/truncated file, or a foreign/old DB whose `sentence_stat` lacks our
    columns, returns False so the caller quarantines it instead of crashing on
    first use (load_stats assumes key/count/last_ts)."""
    try:
        c = sqlite3.connect(str(path))
        try:
            if c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                return False
            t = c.execute("SELECT name FROM sqlite_master WHERE type='table' "
                          "AND name='sentence_stat'").fetchone()
            if t:  # if our table exists it must carry our columns
                cols = {r[1] for r in c.execute("PRAGMA table_info(sentence_stat)")}
                if not _COLUMNS <= cols:
                    return False
            return True
        finally:
            c.close()
    except sqlite3.DatabaseError:
        return False
