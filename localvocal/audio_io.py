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


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse (n, channels) or (channels, n) to mono float32."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        return a
    # assume the smaller axis is channels
    ch_axis = 0 if a.shape[0] < a.shape[1] else 1
    return a.mean(axis=ch_axis).astype(np.float32)


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample mono float32 audio. soxr if installed, else linear interp."""
    a = np.asarray(audio, dtype=np.float32)
    if sr_in == sr_out or a.size == 0:
        return a
    try:
        import soxr

        return np.asarray(soxr.resample(a, sr_in, sr_out), dtype=np.float32)
    except ImportError:
        n_out = int(round(a.shape[0] * sr_out / sr_in))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, a.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, a).astype(np.float32)


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
