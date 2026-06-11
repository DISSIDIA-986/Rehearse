"""Unit tests for the per-turn latency budget tracer (rehearse/latency.py).

Pure logic — no audio, no mic, no models. Covers the formatting, the percentile
math (incl. edge cases the loop summary will actually hit), and the aggregator's
None-skipping (empty-transcript turns must not drag the felt summary to zero)."""

from __future__ import annotations

from rehearse.latency import LatencyAggregator, TurnTrace, _percentile


# ---- _percentile edge cases -------------------------------------------------

def test_percentile_empty_is_zero():
    assert _percentile([], 50) == 0.0
    assert _percentile([], 95) == 0.0


def test_percentile_single_value():
    assert _percentile([1.5], 50) == 1.5
    assert _percentile([1.5], 95) == 1.5


def test_percentile_p50_is_median_like():
    assert _percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_percentile_p0_and_p100_are_min_max():
    vals = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert _percentile(vals, 0) == 1.0
    assert _percentile(vals, 100) == 5.0


def test_percentile_interpolates():
    # two points: p50 sits exactly halfway between them
    assert _percentile([0.0, 10.0], 50) == 5.0


# ---- TurnTrace.one_line -----------------------------------------------------

def test_one_line_all_stages():
    t = TurnTrace(asr_s=0.31, ttft_s=0.62, tts_ttfa_s=0.28, tts_total_s=0.9, felt_s=1.42)
    line = t.one_line()
    assert "felt=1.42s" in line
    assert "asr=0.31" in line
    assert "ttft=0.62" in line
    assert "tts_ttfa=0.28" in line


def test_one_line_missing_stages_render_dash():
    t = TurnTrace(asr_s=0.30)  # empty-transcript turn: no LLM/TTS
    line = t.one_line()
    assert "asr=0.30" in line
    assert "ttft=—" in line
    assert "tts_ttfa=—" in line
    assert "felt=—" in line


# ---- LatencyAggregator ------------------------------------------------------

def test_aggregator_empty_summary_is_blank():
    agg = LatencyAggregator()
    assert agg.summary() == {}
    assert agg.summary_line() == ""
    assert len(agg) == 0


def test_aggregator_skips_none_columns():
    agg = LatencyAggregator()
    # one full turn + one empty-transcript turn (asr only)
    agg.add(TurnTrace(asr_s=0.3, ttft_s=0.6, tts_ttfa_s=0.2, felt_s=1.0))
    agg.add(TurnTrace(asr_s=0.4))  # no felt/ttft/tts
    s = agg.summary()
    assert s["felt_s"]["n"] == 1.0   # only the first turn had audio
    assert s["ttft_s"]["n"] == 1.0
    assert s["asr_s"]["n"] == 2.0    # both turns transcribed
    assert len(agg) == 2


def test_aggregator_percentiles():
    agg = LatencyAggregator()
    for f in (1.0, 2.0, 3.0):
        agg.add(TurnTrace(asr_s=0.3, ttft_s=0.6, tts_ttfa_s=0.2, felt_s=f))
    s = agg.summary()
    assert s["felt_s"]["p50"] == 2.0
    assert s["felt_s"]["p95"] >= 2.8  # near the top of [1,3]


def test_summary_line_mentions_turn_count_and_stages():
    agg = LatencyAggregator()
    agg.add(TurnTrace(asr_s=0.3, ttft_s=0.6, tts_ttfa_s=0.2, felt_s=1.0))
    line = agg.summary_line()
    assert "over 1 turns" in line
    assert "felt p50=" in line and "p95=" in line
    assert "asr p50=" in line
    assert "ttft p50=" in line
    assert "tts_ttfa p50=" in line


def test_summary_line_blank_when_only_empty_turns():
    # turns with no measured stage at all -> nothing to report
    agg = LatencyAggregator()
    agg.add(TurnTrace())
    assert agg.summary_line() == ""
