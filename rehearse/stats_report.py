"""Offline practice-stats report (`rehearse-stats`).

Reads the cross-session practice DB (T-P2-2a) and prints a progress summary —
most/least-practiced sentences and, if shadow tracking has run (--enable-unaided,
T-P2-2b), the logged 'unaided production' events. Pure read, no mic, no models:
the between-sessions "how am I doing / what should I review" view, and the audit
trail for calibrating the unaided threshold.

`format_report` is pure (stats + events -> text) so it's unit-tested without a DB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rehearse.practice_store import PracticeStore, default_db_path

_TRIM = 56  # keep each line readable in a terminal


def _trim(text: str) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= _TRIM else text[: _TRIM - 1] + "…"


def _ago(ts: float, now: float) -> str:
    """Human relative time. `ts`<=0 (never practiced) -> 'never'."""
    if ts <= 0:
        return "never"
    d = now - ts
    if d < 0:
        return "just now"  # clock skew -> don't print a negative
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)}m ago"
    if d < 86400:
        return f"{int(d // 3600)}h ago"
    return f"{int(d // 86400)}d ago"


def format_report(stats: dict, events: list, now: float, db_label: str = "practice.db") -> str:
    """Render a progress summary from load_stats() output + unaided_events()."""
    practiced = [(k, s) for k, s in stats.items() if s.count > 0]
    if not practiced and not events:
        return (f"No practice history yet ({db_label}).\n"
                "Run a session (English practice) — stats persist here across sessions.")

    lines = [f"Practice stats — {db_label}"]
    total_reps = sum(s.count for _, s in practiced)
    lines.append(f"  {len(practiced)} sentences practiced, {total_reps} total reps")

    most = sorted(practiced, key=lambda kv: (-kv[1].count, kv[1].last_ts))[:5]
    if most:
        lines.append("  Most practiced:")
        lines += [f"    {s.count}x  {_trim(k)}  ({_ago(s.last_ts, now)})" for k, s in most]

    if len(practiced) > 5:  # only worth a second list when it differs from 'most'
        least = sorted(practiced, key=lambda kv: (kv[1].count, kv[1].last_ts))[:5]
        lines.append("  Least practiced (due for review):")
        lines += [f"    {s.count}x  {_trim(k)}  ({_ago(s.last_ts, now)})" for k, s in least]

    if events:
        lines.append(f"  Unaided production (shadow): {len(events)} events")
        for ts, key, sim in list(events)[-5:]:
            sim_s = f"{sim:.2f}" if sim is not None else "  ? "
            lines.append(f"    {sim_s}  {_trim(key)}  ({_ago(ts, now)})")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import time

    ap = argparse.ArgumentParser(prog="rehearse-stats",
                                 description="Show your cross-session practice progress.")
    ap.add_argument("--practice-db", default=None,
                    help="practice DB path (default: ~/.local/share/rehearse/practice.db)")
    args = ap.parse_args(argv)

    db_path = Path(args.practice_db) if args.practice_db else default_db_path()
    if not db_path.exists():  # don't create an empty DB just to report nothing
        print(f"No practice history yet ({db_path}).\n"
              "Run a session (English practice) — stats persist here across sessions.")
        return 0
    try:
        store = PracticeStore(db_path)
    except Exception as e:
        print(f"Could not open practice DB {db_path}: {e}", file=sys.stderr)
        return 1
    try:
        print(format_report(store.load_stats(), store.unaided_events(),
                            time.time(), db_label=str(db_path)))
    finally:
        store.close()
    return 0


def _cli() -> None:
    """Console-script entry; run_cli funnels every exit through fast_exit to skip
    the sentencepiece/abseil finalize SIGBUS (see rehearse._exit). main() stays
    pure for tests."""
    from rehearse._exit import run_cli
    run_cli(main)


if __name__ == "__main__":
    _cli()
