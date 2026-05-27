"""Voice-activity endpointing state machine (D2 / Codex golden params).

The state machine is pure: feed it a per-frame speech probability (0..1, e.g.
from Silero VAD) and it emits "start"/"end" events. Separating it from the model
makes the endpointing logic — the part most likely to have bugs — unit-testable
with synthetic probability sequences, no model download required.

Defaults follow the design's golden params: start after ~64ms voiced, end after
~300ms silence, hard cap ~8s (放宽, 让用户把话说完). frame_ms=32 matches Silero v5's
512-sample window at 16 kHz.

PRE-ROLL is the CAPTURE layer's job, not this state machine's: the mic loop must
keep a ~200ms ring buffer and prepend it when "start" fires, otherwise the first
frames of the user's speech (before the voiced threshold trips) are lost. This
detector intentionally holds no audio. (Phase E / main_loop responsibility.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

SILERO_FRAME = 512  # samples Silero v5 expects per call @ 16 kHz (= 32 ms)


class VadState(Enum):
    IDLE = "idle"          # waiting for speech to begin
    SPEAKING = "speaking"  # user is talking
    ENDED = "ended"        # utterance complete (caller should collect + reset)


@dataclass
class EndpointConfig:
    threshold: float = 0.5        # speech prob above which a frame counts as voiced
    frame_ms: int = 32            # Silero v5: 512 samples @ 16kHz = 32ms
    start_voiced_ms: int = 64     # consecutive voiced time to START (2 frames, in 60-90ms)
    end_silence_ms: int = 1_000   # trailing silence to END. 1s gives a non-native
                                  # speaker room to pause mid-sentence and think;
                                  # 300ms (native turn-taking) felt "too rushed".
    max_utterance_ms: int = 20_000  # don't cut off a slow/long practice answer


class EndpointDetector:
    """Drive with update(speech_prob) once per audio frame."""

    def __init__(self, config: EndpointConfig | None = None):
        self.cfg = config or EndpointConfig()
        self.reset()

    def reset(self) -> None:
        self.state = VadState.IDLE
        self._voiced_ms = 0
        self._silence_ms = 0
        self._speech_ms = 0

    def update(self, speech_prob: float) -> str | None:
        """Advance one frame. Returns 'start', 'end', or None."""
        cfg = self.cfg
        voiced = speech_prob >= cfg.threshold

        if self.state is VadState.IDLE:
            self._voiced_ms = self._voiced_ms + cfg.frame_ms if voiced else 0
            if self._voiced_ms >= cfg.start_voiced_ms:
                self.state = VadState.SPEAKING
                self._speech_ms = self._voiced_ms
                self._silence_ms = 0
                return "start"
            return None

        if self.state is VadState.SPEAKING:
            self._speech_ms += cfg.frame_ms
            self._silence_ms = 0 if voiced else self._silence_ms + cfg.frame_ms
            if self._silence_ms >= cfg.end_silence_ms:
                self.state = VadState.ENDED
                return "end"
            if self._speech_ms >= cfg.max_utterance_ms:
                self.state = VadState.ENDED
                return "end"
            return None

        return None  # ENDED: caller must reset() before reuse


class SileroVad:
    """Silero VAD v5 wrapper: 512-sample @16k frame -> speech probability 0..1.

    Lazy-imports silero-vad (the `vad` extra; pulls torch). Maintains the model's
    internal recurrent state across frames within an utterance; call reset()
    between utterances. Pair with EndpointDetector for start/end decisions.
    """

    FRAME = SILERO_FRAME

    def __init__(self):
        from silero_vad import load_silero_vad
        import torch

        self._torch = torch
        self._model = load_silero_vad()

    def prob(self, frame_16k: np.ndarray) -> float:
        f = np.asarray(frame_16k, dtype=np.float32).reshape(-1)
        if f.size != self.FRAME:  # pad/trim to the exact window Silero requires
            f = f[: self.FRAME] if f.size > self.FRAME else np.pad(f, (0, self.FRAME - f.size))
        with self._torch.no_grad():
            return float(self._model(self._torch.from_numpy(f), 16_000).item())

    def reset(self) -> None:
        self._model.reset_states()
