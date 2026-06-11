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

SCHEMA_VERSION = 1
_COLUMNS = {"key", "count", "last_ts"}  # the schema load_stats/record assume


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
        self._conn = _connect(":memory:" if self._memory else str(self.path))
        self._migrate()

    # --- schema ----------------------------------------------------------
    def _migrate(self) -> None:
        ver = self._conn.execute("PRAGMA user_version").fetchone()[0]
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS sentence_stat ("
            "  key TEXT PRIMARY KEY,"
            "  count INTEGER NOT NULL DEFAULT 0,"
            "  last_ts REAL NOT NULL DEFAULT 0)"
        )
        # future additive migrations slot in here, guarded by `ver < N`.
        if ver < SCHEMA_VERSION:
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    # --- reads -----------------------------------------------------------
    def load_stats(self) -> dict[str, PracticeStat]:
        """Every sentence's persisted (count, last_ts), keyed by sentence.key —
        drop-in for the in-memory `stats` the loop used to start empty."""
        rows = self._conn.execute(
            "SELECT key, count, last_ts FROM sentence_stat").fetchall()
        return {k: PracticeStat(count=c, last_ts=t) for k, c, t in rows}

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
