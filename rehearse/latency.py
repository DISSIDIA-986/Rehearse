"""Per-turn latency budget tracing for the live voice loop.

Why this exists: a model-swap question ("is Parakeet ASR worth it?") can't be
answered without knowing where the felt latency actually goes. For turn-taking
the budget ranks roughly: end-of-utterance detection -> LLM time-to-first-token
-> TTS time-to-first-audio -> ASR finalization. ASR is usually NOT the
bottleneck, so before changing any model you want to SEE the breakdown.

A turn's stages (all wall-clock seconds):
  eou        end-of-utterance: the VAD fired "end" and we have the utterance
  asr_s      ASR transcription  (eou -> transcript)
  ttft_s     LLM time-to-first-token  (prompt sent -> first token)
  tts_ttfa_s TTS time-to-first-audio  (reply ready -> first audio chunk synthesized)
  felt_s     eou -> first audio reaches the speaker (the number the user feels)

felt_s is measured at the loop level (it includes playback scheduling and the
LLM's full generate-to-first-sentence, not just ttft). The per-stage numbers come
from the turn primitive (pipeline.speak_turn). This module is pure (no audio/IO
deps) so it unit-tests without a mic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnTrace:
    """One turn's latency breakdown. Missing stages stay None (e.g. an empty
    transcript turn never reaches the LLM, so ttft_s/tts_ttfa_s are None)."""

    asr_s: float | None = None
    ttft_s: float | None = None
    tts_ttfa_s: float | None = None
    tts_total_s: float | None = None
    felt_s: float | None = None  # eou -> first audio out, set by the loop shell

    @classmethod
    def from_turn(cls, turn, felt_s: float | None = None) -> "TurnTrace":
        """Build a trace from a pipeline turn (TurnResult or SpokenTurn — both
        carry asr_s/ttft_s/tts_ttfa_s/tts_s). `felt_s` is the loop-measured
        eou -> first-audio time (the only stage the shell, not the primitive,
        can time). Dedups the identical construction across the loop modes."""
        return cls(asr_s=turn.asr_s, ttft_s=turn.ttft_s,
                   tts_ttfa_s=turn.tts_ttfa_s, tts_total_s=turn.tts_s,
                   felt_s=felt_s)

    def one_line(self) -> str:
        """Compact per-turn line for the live loop, e.g.
        'latency: felt=1.42s | asr=0.31 ttft=0.62 tts_ttfa=0.28'."""
        def f(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "—"

        felt = f"felt={f(self.felt_s)}s" if self.felt_s is not None else "felt=—"
        return (f"latency: {felt} | asr={f(self.asr_s)} ttft={f(self.ttft_s)} "
                f"tts_ttfa={f(self.tts_ttfa_s)}")


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile on a copy (no numpy dep here). `pct` in [0,100].
    Empty -> 0.0. Matches the simple semantics the loop summary needs."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    # linear interpolation between closest ranks (numpy 'linear' default)
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


class LatencyAggregator:
    """Collects TurnTraces across a session and reports p50/p95 per stage.

    Only turns that actually produced audio (felt_s is not None) count toward the
    felt summary — empty-transcript turns would skew it toward zero."""

    def __init__(self) -> None:
        self._traces: list[TurnTrace] = []

    def add(self, trace: TurnTrace) -> None:
        self._traces.append(trace)

    def __len__(self) -> int:
        return len(self._traces)

    def _column(self, attr: str) -> list[float]:
        return [v for t in self._traces if (v := getattr(t, attr)) is not None]

    def summary(self) -> dict[str, dict[str, float]]:
        """{stage: {'p50': x, 'p95': y, 'n': k}} for each stage with data."""
        out: dict[str, dict[str, float]] = {}
        for attr in ("felt_s", "asr_s", "ttft_s", "tts_ttfa_s", "tts_total_s"):
            col = self._column(attr)
            if col:
                out[attr] = {
                    "p50": _percentile(col, 50),
                    "p95": _percentile(col, 95),
                    "n": float(len(col)),
                }
        return out

    def summary_line(self) -> str:
        """Human one-liner for end-of-session, e.g.
        'latency over 12 turns: felt p50=1.40 p95=2.10 | asr p50=0.30 | ttft p50=0.60 | tts_ttfa p50=0.25'.
        Returns '' when nothing was recorded."""
        s = self.summary()
        if not s:
            return ""
        n = len(self._traces)
        parts: list[str] = []
        labels = {"felt_s": "felt", "asr_s": "asr", "ttft_s": "ttft",
                  "tts_ttfa_s": "tts_ttfa"}
        for attr, label in labels.items():
            if attr in s:
                st = s[attr]
                if attr == "felt_s":
                    parts.append(f"{label} p50={st['p50']:.2f} p95={st['p95']:.2f}")
                else:
                    parts.append(f"{label} p50={st['p50']:.2f}")
        return f"latency over {n} turns: " + " | ".join(parts)
