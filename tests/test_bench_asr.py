"""Unit tests for the ASR benchmark harness (rehearse/bench_asr.py).

Covers the pure decision logic — WER math + edge cases, WAV round-trip, sample
loading, backend run with a FAKE recognizer (no model, no GPU), report + verdict.
The real model calls and mic recording are the on-device manual step (like the
live loop) and aren't exercised here."""

from __future__ import annotations

import numpy as np

from rehearse.audio_io import ASR_SR
from rehearse.bench_asr import (
    Sample,
    format_report,
    load_samples,
    normalize_words,
    read_wav,
    run_backend,
    wer,
    write_wav,
)


# ---- normalize + WER --------------------------------------------------------

def test_normalize_lowercases_and_strips_punct():
    assert normalize_words("Hello, World!") == ["hello", "world"]


def test_normalize_keeps_apostrophes_and_digits():
    assert normalize_words("I'm at 3pm.") == ["i'm", "at", "3pm"]


def test_wer_perfect_match_is_zero():
    assert wer("the quick brown fox", "the quick brown fox") == 0.0


def test_wer_case_and_punct_insensitive():
    assert wer("Hello, world.", "hello world") == 0.0


def test_wer_one_substitution():
    # 1 wrong word out of 4
    assert wer("the quick brown fox", "the quick green fox") == 0.25


def test_wer_one_deletion():
    assert wer("the quick brown fox", "the quick fox") == 0.25


def test_wer_empty_ref_empty_hyp_is_zero():
    assert wer("", "") == 0.0


def test_wer_empty_ref_nonempty_hyp_is_one():
    assert wer("", "extra words here") == 1.0


def test_wer_empty_hyp_is_total_loss():
    assert wer("one two three", "") == 1.0


def test_wer_can_exceed_one_on_insertions():
    # ref 1 word, hyp 3 unrelated words -> 1 sub + 2 ins = 3 edits / 1 ref word = 3.0
    assert wer("hello", "a b c") == 3.0


# ---- WAV round-trip ---------------------------------------------------------

def test_wav_roundtrip_preserves_signal(tmp_path):
    t = np.linspace(0, 1, ASR_SR, endpoint=False, dtype=np.float32)
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    p = tmp_path / "a.wav"
    write_wav(str(p), sig)
    back = read_wav(p)
    assert back.shape == sig.shape
    assert np.max(np.abs(back - sig)) < 1e-3  # 16-bit quantization only


def test_read_wav_rejects_wrong_samplerate(tmp_path):
    import wave

    p = tmp_path / "bad.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(48_000)
        w.writeframes(np.zeros(100, dtype=np.int16).tobytes())
    try:
        read_wav(p)
        assert False, "expected ValueError on 48 kHz file"
    except ValueError as e:
        assert "Hz" in str(e)


def test_read_wav_rejects_stereo(tmp_path):
    import wave

    p = tmp_path / "stereo.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(ASR_SR)
        w.writeframes(np.zeros(200, dtype=np.int16).tobytes())
    try:
        read_wav(p)
        assert False, "expected ValueError on stereo file"
    except ValueError as e:
        assert "channel" in str(e).lower()


# ---- sample loading ---------------------------------------------------------

def test_load_samples_pairs_wav_and_txt(tmp_path):
    sig = np.zeros(ASR_SR // 2, dtype=np.float32)
    write_wav(str(tmp_path / "utt01.wav"), sig)
    (tmp_path / "utt01.txt").write_text("hello there", encoding="utf-8")
    # an unpaired wav is skipped, not crashed on
    write_wav(str(tmp_path / "orphan.wav"), sig)
    samples = load_samples(tmp_path)
    assert len(samples) == 1
    assert samples[0].name == "utt01"
    assert samples[0].ref == "hello there"


# ---- run_backend + report (fake recognizer — no model, no GPU) --------------

def _samples(names_refs):
    return [Sample(n, np.zeros(10, dtype=np.float32), r) for n, r in names_refs]


def test_run_backend_scores_each_utterance():
    samples = _samples([("u1", "the cat sat"), ("u2", "good morning")])

    class Perfect:
        def transcribe(self, audio, language="en"):  # noqa: ARG002
            # echo the matching ref by position
            return ref_iter.pop(0)

    ref_iter = ["the cat sat", "good morning"]
    res = run_backend(Perfect(), samples, backend="x", model="m")
    assert res.mean_wer == 0.0
    assert len(res.utts) == 2
    assert res.p50_wall >= 0.0


def test_run_backend_counts_errors():
    samples = _samples([("u1", "the cat sat on the mat")])

    class Wrong:
        def transcribe(self, audio, language="en"):  # noqa: ARG002
            return "the dog sat on the mat"  # 1/6 wrong

    res = run_backend(Wrong(), samples, backend="x", model="m")
    assert abs(res.mean_wer - 1 / 6) < 1e-9


def test_format_report_verdict_keep_when_no_win():
    from rehearse.bench_asr import BackendResult, UttResult
    a = BackendResult("whisper", "small.en", [UttResult("u1", "a b c", "a b c", 0.05, 0.3)])
    b = BackendResult("parakeet", "v3", [UttResult("u1", "a b c", "a b c", 0.05, 0.1)])
    report = format_report([a, b])
    assert "Δ mean WER" in report
    assert "KEEP the current backend" in report


def test_format_report_verdict_flags_real_win():
    from rehearse.bench_asr import BackendResult, UttResult
    a = BackendResult("whisper", "small.en", [UttResult("u1", "a b c d", "x b c d", 0.25, 0.3)])
    b = BackendResult("parakeet", "v3", [UttResult("u1", "a b c d", "a b c d", 0.0, 0.1)])
    report = format_report([a, b])
    assert "check felt-latency/contention before switching" in report
