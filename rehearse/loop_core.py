"""Testable glue between the audio frame stream and complete utterances.

Separated from main_loop's sounddevice/threading shell so the start/collect/end/
pre-roll logic — which has no automated coverage in the live loop — can be driven
deterministically with a fake VAD (Codex final-review gap #6 / D4).
"""

from __future__ import annotations

import numpy as np

from localvocal.audio_io import RingBuffer
from localvocal.vad import EndpointDetector, VadState

# Spoken end-of-session phrases (normalized: lowercased, trailing punctuation stripped).
_STOP = {
    "stop", "goodbye", "bye", "good bye", "exit", "quit",
    "that's all", "thats all", "i'm done", "im done", "let's stop", "lets stop",
}


def is_stop(text: str) -> bool:
    """True if the user asked to end the session."""
    return text.strip().lower().rstrip(" .!?,") in _STOP


class UtteranceAssembler:
    """Feed 16 kHz frames; get a complete utterance (pre-roll prepended) on 'end'.

    `vad` is any object with prob(frame)->float and reset(). Returns the assembled
    utterance ndarray when the endpoint fires "end", else None.
    """

    def __init__(self, vad, endpoint: EndpointDetector, preroll: RingBuffer):
        self.vad = vad
        self.endpoint = endpoint
        self.preroll = preroll
        self._buf: list[np.ndarray] = []

    def push(self, frame16: np.ndarray) -> np.ndarray | None:
        self.preroll.push(frame16)
        ev = self.endpoint.update(self.vad.prob(frame16))
        if ev == "start":
            self._buf = [self.preroll.get()]  # prepend pre-roll, no onset clip
        elif self.endpoint.state is VadState.SPEAKING:
            self._buf.append(frame16)
        elif ev == "end":
            self._buf.append(frame16)
            utt = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.float32)
            self._buf = []
            self.endpoint.reset()
            self.vad.reset()
            return utt
        return None
