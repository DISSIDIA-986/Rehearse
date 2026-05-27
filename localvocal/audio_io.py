"""Audio I/O + resampling.

Codex blind-spot #2: keep ONE device stream open and resample ONCE per direction;
never reopen the device per turn. Sample rates in this pipeline:
  - device capture/playback: 48 kHz (typical Mac default)
  - ASR (faster-whisper):     16 kHz mono float32
  - TTS (Kokoro):             24 kHz mono float32 (resampled up to device on playback)

Resampling uses soxr (high quality, no torch) when available, with a numpy
linear-interpolation fallback so the pure logic is testable without the extra dep.
"""

from __future__ import annotations

import numpy as np

ASR_SR = 16_000
TTS_SR = 24_000
DEFAULT_DEVICE_SR = 48_000


def _scrub(a: np.ndarray) -> np.ndarray:
    """float32 + replace NaN/Inf with 0 so one bad buffer can't poison the pipeline."""
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse to mono float32.

    Expects sounddevice's layout: 1-D mono, or 2-D (frames, channels). We do NOT
    guess the channel axis — sounddevice always returns (frames, channels), so we
    average the LAST axis. (channels, frames) input would be wrong; callers must
    pass the sounddevice shape.
    """
    a = _scrub(audio)
    if a.ndim == 1:
        return a
    if a.ndim == 2:
        return a.mean(axis=1).astype(np.float32)
    raise ValueError(f"to_mono expects 1-D or 2-D audio, got shape {a.shape}")


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float32 audio.

    Uses soxr (anti-aliased, the declared `audio` dep). Without soxr we fall back
    to linear interpolation ONLY for upsampling (no aliasing risk); downsampling
    without soxr raises instead of silently aliasing high frequencies into the
    speech band (which would degrade ASR while looking like a model problem).
    """
    a = _scrub(audio)
    if sr_in == sr_out or a.size == 0:
        return a
    try:
        import soxr

        return np.asarray(soxr.resample(a, sr_in, sr_out), dtype=np.float32)
    except ImportError as e:
        if sr_out < sr_in:
            raise RuntimeError(
                f"downsampling {sr_in}->{sr_out} needs soxr (anti-aliasing); "
                "install the audio extra (uv sync --extra audio)"
            ) from e
        n_out = int(round(a.shape[0] * sr_out / sr_in))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, a.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, a).astype(np.float32)


class RingBuffer:
    """Fixed-duration rolling buffer of recent mono samples (pre-roll).

    The capture loop pushes every frame here while IDLE; when VAD fires "start",
    the caller prepends get() so the first ~200ms of speech (spoken before the
    voiced threshold tripped) is not clipped. Solves the Codex pre-roll gap.
    """

    def __init__(self, max_samples: int):
        self.max_samples = max(0, int(max_samples))
        self._arr = np.zeros(0, dtype=np.float32)

    def push(self, frame: np.ndarray) -> None:
        if self.max_samples == 0:
            return
        f = np.asarray(frame, dtype=np.float32).reshape(-1)
        if f.size:
            self._arr = np.concatenate([self._arr, f])[-self.max_samples:]

    def get(self) -> np.ndarray:
        return self._arr.copy()

    def clear(self) -> None:
        self._arr = np.zeros(0, dtype=np.float32)


def list_devices():  # pragma: no cover - hardware dependent
    """Return sounddevice's device table (for the manual mic test / smoke)."""
    import sounddevice as sd

    return sd.query_devices()


def default_output_samplerate() -> int:  # pragma: no cover - hardware dependent
    import sounddevice as sd

    try:
        return int(sd.query_devices(kind="output")["default_samplerate"])
    except Exception:
        return DEFAULT_DEVICE_SR
