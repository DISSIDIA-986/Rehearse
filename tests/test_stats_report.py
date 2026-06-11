"""Tests for the offline practice-stats report (rehearse/stats_report.py)."""

from __future__ import annotations

from rehearse.practice_store import PracticeStore
from rehearse.session_seeder import PracticeStat
from rehearse.stats_report import _ago, _trim, format_report, main

NOW = 1_000_000.0


# --- _ago / _trim (pure) --------------------------------------------------

def test_ago_buckets():
    assert _ago(0, NOW) == "never"
    assert _ago(NOW, NOW) == "just now"
    assert _ago(NOW - 120, NOW) == "2m ago"
    assert _ago(NOW - 7200, NOW) == "2h ago"
    assert _ago(NOW - 2 * 86400, NOW) == "2d ago"


def test_ago_future_clock_skew_is_just_now():
    assert _ago(NOW + 500, NOW) == "just now"  # never prints a negative age


def test_trim_long_text():
    out = _trim("x" * 100)
    assert len(out) <= 56 and out.endswith("…")
    assert _trim("  hello   world  ") == "hello world"  # whitespace collapsed


# --- format_report (pure) -------------------------------------------------

def test_format_empty_is_friendly():
    out = format_report({}, [], NOW, db_label="x.db")
    assert "No practice history yet" in out and "x.db" in out


def test_format_counts_and_most_practiced():
    stats = {
        "alpha sentence": PracticeStat(count=5, last_ts=NOW - 3600),
        "beta sentence": PracticeStat(count=2, last_ts=NOW - 86400),
        "never one": PracticeStat(count=0, last_ts=0),  # count 0 -> not counted
    }
    out = format_report(stats, [], NOW)
    assert "2 sentences practiced, 7 total reps" in out  # the count==0 one excluded
    assert "5x  alpha sentence  (1h ago)" in out
    assert "Most practiced:" in out
    assert "Least practiced" not in out  # only 2 practiced -> no second list


def test_format_shows_least_when_many():
    stats = {f"s{i}": PracticeStat(count=i + 1, last_ts=NOW - i) for i in range(8)}
    out = format_report(stats, [], NOW)
    assert "Least practiced (due for review):" in out
    assert "1x  s0" in out  # least-practiced surfaces


def test_format_unaided_events_section():
    stats = {"k one": PracticeStat(count=1, last_ts=NOW)}
    events = [(NOW - 86400, "k one", 0.82), (NOW - 60, "k two", 0.71)]
    out = format_report(stats, events, NOW)
    assert "Unaided production (shadow): 2 events" in out
    assert "0.82  k one  (1d ago)" in out
    assert "0.71  k two  (1m ago)" in out


def test_format_unaided_handles_null_similarity():
    out = format_report({}, [(NOW, "k", None)], NOW)
    assert "Unaided production (shadow): 1 events" in out  # no crash on NULL sim


# --- main (e2e against a real DB) -----------------------------------------

def test_main_missing_db_is_friendly(tmp_path, capsys):
    rc = main(["--practice-db", str(tmp_path / "nope.db")])
    assert rc == 0
    assert "No practice history yet" in capsys.readouterr().out
    assert not (tmp_path / "nope.db").exists()  # must NOT create an empty DB


def test_main_reports_real_db(tmp_path, capsys):
    db = tmp_path / "practice.db"
    with PracticeStore(db) as s:
        s.record_practiced(["it just didn't appeal to me"], now=NOW - 3600)
        s.record_practiced(["how are you doing"], now=NOW - 7200)
        s.record_practiced(["how are you doing"], now=NOW - 60)  # count 2
        s.record_unaided("it just didn't appeal to me", 0.79, now=NOW - 30)
    rc = main(["--practice-db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2 sentences practiced, 3 total reps" in out
    assert "how are you doing" in out
    assert "Unaided production (shadow): 1 events" in out
