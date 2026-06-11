"""Benchmark faster-whisper (current) vs parakeet-mlx on YOUR Mac and YOUR voice.

Why this exists: public WER tables (HF Open ASR Leaderboard etc.) can't decide an
ASR swap for THIS setup — your accent, your DJI mic, your room, and crucially the
Metal-GPU contention question (the shipped design keeps ASR on the CPU precisely so
it doesn't fight the LLM/TTS for the GPU; parakeet-mlx runs on Metal). So before
swapping anything, measure it locally. See docs/ASR-evaluation.md.

This is a DECISION TOOL. It does NOT touch the shipped pipeline. Promotion bar
(from the Codex joint review): a candidate must MATERIALLY beat whisper small.en on
transcript quality (WER on your own utterances) AND not regress p95 felt latency
under contention. Saving a few ASR milliseconds is not a reason to switch — for
turn-taking the felt budget is dominated by VAD end-of-utterance + LLM TTFT + TTS.

Run:
    uv sync --extra audio --extra parakeet     # adds parakeet-mlx
    # 1) record a handful of your own utterances against reference sentences:
    rehearse-bench-asr record --refs sentences.txt --out bench_audio/
    # 2) compare the backends on them:
    rehearse-bench-asr run --wav-dir bench_audio/
    # 3) (optional) measure GPU contention impact on coach TTFT:
    rehearse-bench-asr run --wav-dir bench_audio/ --contention

The pure logic here (WER, sample loading, aggregation, report) is unit-tested with
fake backends; the model calls and mic recording are the manual on-device step,
exactly like the live voice loop (the project's only manually-validated path).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rehearse.audio_io import ASR_SR

PARAKEET_DEFAULT = "mlx-community/parakeet-tdt-0.6b-v3"


# --------------------------------------------------------------------------- #
# Word Error Rate (pure Python, zero deps — no jiwer)                          #
# --------------------------------------------------------------------------- #

def normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation (keep intra-word apostrophes), split on space.
    The WER normalization both backends are scored under — keep it identical so the
    comparison is fair (no backend gets credit/blame for casing or commas)."""
    text = (text or "").lower()
    text = re.sub(r"[^\w\s']", " ", text)  # punctuation -> space; \w keeps digits
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate = word-level Levenshtein(ref, hyp) / len(ref words).

    Empty ref: 0.0 if hyp also empty, else 1.0 (any words are pure insertion).
    Can exceed 1.0 when hyp is much longer than ref (lots of insertions) — that's
    correct WER semantics, not a bug; we don't clamp it."""
    r = normalize_words(ref)
    h = normalize_words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # iterative DP over one row (O(len(h)) memory)
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev = dp[0]
        dp[0] = i
        for j, hw in enumerate(h, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1,        # deletion
                        dp[j - 1] + 1,    # insertion
                        prev + (rw != hw))  # match/substitution
            prev = cur
    return dp[len(h)] / len(r)


# --------------------------------------------------------------------------- #
# Parakeet wrapper — mirrors WhisperASR.transcribe(audio_16k_mono) -> str       #
# --------------------------------------------------------------------------- #

class ParakeetASR:
    """parakeet-mlx (NVIDIA Parakeet TDT, MLX/Metal) behind the WhisperASR API.

    Lazy import so the package imports without the optional `parakeet` extra.
    parakeet-mlx's transcribe() loads audio from a path, so we spill the 16 kHz
    float32 utterance to a temp WAV. NOTE: like tts.py's mlx-audio note, this API
    has shifted across versions — verified against parakeet-mlx's from_pretrained()
    + transcribe().text shape; adjust here if the package API moves."""

    def __init__(self, model_id: str = PARAKEET_DEFAULT):
        from parakeet_mlx import from_pretrained  # lazy; needs the parakeet extra

        self.model_id = model_id
        self._model = from_pretrained(model_id)

    def transcribe(self, audio_16k_mono: np.ndarray, language: str = "en") -> str:
        audio = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
            write_wav(tf.name, audio)
            result = self._model.transcribe(tf.name)
        # parakeet-mlx returns an object with .text (and segments); be defensive.
        return (getattr(result, "text", None) or str(result)).strip()


def write_wav(path: str, audio_16k_mono: np.ndarray) -> None:
    """Write 16 kHz mono float32 to a 16-bit PCM WAV (same encoding as _save_debug)."""
    pcm = (np.clip(np.asarray(audio_16k_mono, dtype=np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ASR_SR)
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> np.ndarray:
    """Read a 16-bit PCM mono WAV back to 16 kHz float32. Raises on rate mismatch
    (a stray 48 kHz file would silently inflate WER if we didn't catch it)."""
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != ASR_SR:
            raise ValueError(f"{path}: expected {ASR_SR} Hz, got {w.getframerate()} Hz")
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        if w.getnchannels() != 1:  # stereo would be read as interleaved -> garbage to ASR
            raise ValueError(f"{path}: expected mono, got {w.getnchannels()} channels")
        frames = w.readframes(w.getnframes())
    mono = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return mono


# --------------------------------------------------------------------------- #
# Samples                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    name: str
    audio: np.ndarray
    ref: str


def load_samples(wav_dir: str | Path) -> list[Sample]:
    """Load <name>.wav + sibling <name>.txt (reference transcript) pairs from a dir.
    Skips any .wav lacking a .txt (warns), sorted by name for stable reports."""
    wav_dir = Path(wav_dir)
    samples: list[Sample] = []
    for wav in sorted(wav_dir.glob("*.wav")):
        ref_path = wav.with_suffix(".txt")
        if not ref_path.exists():
            print(f"  skip {wav.name}: no sibling {ref_path.name}", file=sys.stderr)
            continue
        ref = ref_path.read_text(encoding="utf-8").strip()
        samples.append(Sample(wav.stem, read_wav(wav), ref))
    return samples


# --------------------------------------------------------------------------- #
# Run + report                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class UttResult:
    name: str
    ref: str
    hyp: str
    wer: float
    wall_s: float


@dataclass
class BackendResult:
    backend: str
    model: str
    utts: list[UttResult] = field(default_factory=list)

    @property
    def mean_wer(self) -> float:
        return sum(u.wer for u in self.utts) / len(self.utts) if self.utts else 0.0

    @property
    def p50_wall(self) -> float:
        return _pct([u.wall_s for u in self.utts], 50)

    @property
    def p95_wall(self) -> float:
        return _pct([u.wall_s for u in self.utts], 95)


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def run_backend(asr, samples: list[Sample], *, backend: str, model: str) -> BackendResult:
    """Transcribe every sample through `asr` (anything with .transcribe(audio)->str),
    timing each call and scoring WER vs the reference. Backend-agnostic so tests can
    drive it with a fake recognizer (no model download, no GPU)."""
    res = BackendResult(backend=backend, model=model)
    for s in samples:
        t0 = time.monotonic()
        hyp = asr.transcribe(s.audio)
        wall = time.monotonic() - t0
        res.utts.append(UttResult(s.name, s.ref, hyp, wer(s.ref, hyp), wall))
    return res


def format_report(results: list[BackendResult]) -> str:
    """Human-readable comparison: per-utterance WER table per backend + a summary
    line each, ending with the head-to-head WER delta and the promotion verdict."""
    lines: list[str] = []
    for r in results:
        lines.append(f"\n=== {r.backend} ({r.model}) ===")
        lines.append(f"{'utt':<16} {'WER':>7}  {'wall_s':>7}  hyp")
        for u in r.utts:
            lines.append(f"{u.name:<16} {u.wer:>7.3f}  {u.wall_s:>7.3f}  {u.hyp!r}")
        lines.append(f"  mean WER={r.mean_wer:.3f}  wall p50={r.p50_wall:.3f}s "
                     f"p95={r.p95_wall:.3f}s  (n={len(r.utts)})")
    if len(results) == 2:
        a, b = results
        d_wer = b.mean_wer - a.mean_wer
        lines.append(f"\nΔ mean WER ({b.backend} - {a.backend}) = {d_wer:+.3f} "
                     f"(negative = {b.backend} more accurate)")
        verdict = ("candidate WER is BETTER — now check felt-latency/contention before "
                   "switching") if d_wer < -0.005 else \
                  ("no material WER win — KEEP the current backend" if d_wer < 0.005 else
                   "candidate is WORSE — KEEP the current backend")
        lines.append(f"verdict: {verdict}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Mic recording (the user's on-device step) + contention probe — hardware       #
# --------------------------------------------------------------------------- #

def record_samples(refs_path: str | Path, out_dir: str | Path) -> int:  # pragma: no cover
    """Read reference sentences (one per line) from refs_path; for each, prompt the
    user to read it aloud, record from the default mic until Enter, save <n>.wav +
    <n>.txt into out_dir. This is the real-mic capture the benchmark needs."""
    import queue

    import sounddevice as sd

    from rehearse import audio_io

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs = [ln.strip() for ln in Path(refs_path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    if not refs:
        print("no reference sentences found", file=sys.stderr)
        return 1
    try:  # capture uses the INPUT device's default rate (may differ from output)
        in_sr = int(sd.query_devices(kind="input")["default_samplerate"])
    except Exception:
        in_sr = audio_io.DEFAULT_DEVICE_SR
    block = max(1, round(0.03 * in_sr))
    audio_q: queue.Queue = queue.Queue()
    recording = False

    def _cb(indata, frames, t, status):  # noqa: ARG001
        if recording:
            audio_q.put(audio_io.to_mono(indata).copy())

    print(f"Recording {len(refs)} utterances to {out}/ (Ctrl-C to stop early).\n")
    with sd.InputStream(samplerate=in_sr, channels=1, blocksize=block,
                        dtype="float32", callback=_cb):
        for i, ref in enumerate(refs, 1):
            input(f"[{i}/{len(refs)}] Press Enter, then read:\n    \"{ref}\"\n  (Enter to start) ")
            while not audio_q.empty():
                audio_q.get_nowait()
            recording = True
            input("  recording... Enter when done ")
            recording = False
            blocks = []
            while not audio_q.empty():
                blocks.append(audio_q.get_nowait())
            if not blocks:
                print("  (heard nothing — skipping)\n")
                continue
            utt = audio_io.resample(np.concatenate(blocks), in_sr, ASR_SR)
            stem = out / f"utt{i:02d}"
            write_wav(str(stem.with_suffix(".wav")), utt)
            stem.with_suffix(".txt").write_text(ref, encoding="utf-8")
            print(f"  saved {stem.name}.wav\n")
    return 0


def contention_probe(samples: list[Sample], *, coach_backend: str = "auto",
                     model: str | None = None, n: int = 5) -> str:  # pragma: no cover
    """Forced-overlap test (Codex's promotion gate): measure coach LLM TTFT alone vs
    while a parakeet transcription runs concurrently on the Metal GPU. If TTFT barely
    moves, the on-CPU rule can relax for ASR; if p95 TTFT jumps, keep ASR off the GPU."""
    import threading

    from rehearse.mlx_llm import resolve_coach_chat

    chat_fn, mid, backend = resolve_coach_chat(coach_backend, model)
    probe_msgs = [{"role": "user", "content": "Say one short friendly sentence."}]

    def _ttft() -> float:
        return chat_fn(probe_msgs, num_predict=8).ttft_s or 0.0

    para = ParakeetASR()
    audio = samples[0].audio if samples else np.zeros(ASR_SR, dtype=np.float32)

    baseline = [_ttft() for _ in range(n)]

    overlapped: list[float] = []
    for _ in range(n):
        stop = threading.Event()

        def _hammer():
            while not stop.is_set():
                para.transcribe(audio)

        t = threading.Thread(target=_hammer, daemon=True)
        t.start()
        overlapped.append(_ttft())
        stop.set()
        t.join()  # wait out the in-flight transcription so it can't bleed into the next trial

    return (f"contention probe (coach={backend}/{mid}, n={n}):\n"
            f"  TTFT alone:       p50={_pct(baseline,50):.3f}s p95={_pct(baseline,95):.3f}s\n"
            f"  TTFT + ASR(GPU):  p50={_pct(overlapped,50):.3f}s p95={_pct(overlapped,95):.3f}s\n"
            f"  Δ p95 TTFT = {(_pct(overlapped,95) - _pct(baseline,95)):+.3f}s "
            f"(large positive = GPU contention is real; keep ASR on CPU)")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="rehearse-bench-asr",
        description="Benchmark faster-whisper vs parakeet-mlx on your own utterances.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="record your own utterances against reference sentences")
    rec.add_argument("--refs", required=True, help="text file: one reference sentence per line")
    rec.add_argument("--out", required=True, help="output dir for <n>.wav + <n>.txt pairs")

    run = sub.add_parser("run", help="compare whisper vs parakeet on recorded utterances")
    run.add_argument("--wav-dir", required=True, help="dir of <name>.wav + <name>.txt pairs")
    run.add_argument("--whisper-model", default="small.en")
    run.add_argument("--parakeet-model", default=PARAKEET_DEFAULT)
    run.add_argument("--only", choices=["whisper", "parakeet"], default=None,
                     help="run just one backend (e.g. parakeet not installed yet)")
    run.add_argument("--contention", action="store_true",
                     help="also run the coach-TTFT-under-GPU-contention probe")

    args = ap.parse_args(argv)

    if args.cmd == "record":
        return record_samples(args.refs, args.out)

    samples = load_samples(args.wav_dir)
    if not samples:
        print(f"no <name>.wav + <name>.txt pairs in {args.wav_dir}", file=sys.stderr)
        return 1
    print(f"loaded {len(samples)} utterance(s) from {args.wav_dir}")

    results: list[BackendResult] = []
    if args.only != "parakeet":
        from rehearse.asr import WhisperASR
        whisper = WhisperASR(model_size=args.whisper_model)
        results.append(run_backend(whisper, samples, backend="whisper",
                                    model=args.whisper_model))
    if args.only != "whisper":
        try:
            para = ParakeetASR(args.parakeet_model)
        except ImportError:
            print("parakeet-mlx not installed — run: uv sync --extra audio --extra parakeet",
                  file=sys.stderr)
            if not results:
                return 1
        else:
            results.append(run_backend(para, samples, backend="parakeet",
                                        model=args.parakeet_model))

    print(format_report(results))

    if args.contention:
        try:
            print("\n" + contention_probe(samples))
        except ImportError:
            print("\ncontention probe skipped — needs the parakeet extra "
                  "(uv sync --extra audio --extra parakeet)", file=sys.stderr)
    return 0


def _cli() -> None:  # pragma: no cover
    """Console-script entry; run_cli funnels every exit through fast_exit to skip
    the sentencepiece/abseil finalize SIGBUS (see rehearse._exit). main() stays
    pure for tests."""
    from rehearse._exit import run_cli
    run_cli(main)


if __name__ == "__main__":  # pragma: no cover
    _cli()
