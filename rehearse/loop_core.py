"""Testable glue between the audio frame stream and complete utterances.

Separated from main_loop's sounddevice/threading shell so the start/collect/end/
pre-roll logic — which has no automated coverage in the live loop — can be driven
deterministically with a fake VAD (Codex final-review gap #6 / D4).
"""

from __future__ import annotations

import queue

import numpy as np

from rehearse import audio_io
from rehearse.audio_io import RingBuffer
from rehearse.vad import EndpointDetector, VadState

# samples @16k that the loop reads per audio frame (Silero's 32ms window). The
# device captures at its native rate; `io_rates` maps that back to this window.
FRAME = 512


def io_rates(sd, frame: int = FRAME) -> tuple[int, int, int]:
    """Resolve (in_sr, out_sr, block) from the audio device.

    INDEPENDENT per-direction fallback: a failed output-rate query must not also
    discard a good input rate (and vice versa) — input drives VAD/ASR, output only
    playback. `block` is the device-rate blocksize that maps to one `frame`-sample
    @16k window. Pure over an injected sounddevice-like `sd` (testable, no mic)."""
    def _rate(kind: str) -> int:
        try:
            return int(sd.query_devices(kind=kind)["default_samplerate"])
        except Exception:
            return audio_io.DEFAULT_DEVICE_SR
    in_sr = _rate("input")
    out_sr = _rate("output")
    block = max(1, round(frame * in_sr / audio_io.ASR_SR))
    return in_sr, out_sr, block


def drain_utterance(audio_q, in_sr: int) -> np.ndarray | None:
    """Drain every queued mono block and resample the concatenation ONCE to 16k
    (resampling per block would add boundary artifacts). None if nothing queued.
    The shared tail of the manual / recall record step."""
    blocks: list[np.ndarray] = []
    try:
        while True:
            blocks.append(audio_q.get_nowait())
    except queue.Empty:
        pass
    if not blocks:
        return None
    return audio_io.resample(np.concatenate(blocks), in_sr, audio_io.ASR_SR)

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
