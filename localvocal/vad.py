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


class VadState(Enum):
    IDLE = "idle"          # waiting for speech to begin
    SPEAKING = "speaking"  # user is talking
    ENDED = "ended"        # utterance complete (caller should collect + reset)


@dataclass
class EndpointConfig:
    threshold: float = 0.5        # speech prob above which a frame counts as voiced
    frame_ms: int = 32            # Silero v5: 512 samples @ 16kHz = 32ms
    start_voiced_ms: int = 64     # consecutive voiced time to START (2 frames, in 60-90ms)
    end_silence_ms: int = 300     # trailing silence to END (250-350ms)
    max_utterance_ms: int = 8_000  # hard cap so a stuck mic can't hang forever


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
