import sys

import numpy as np
import pytest

from rehearse.audio_io import resample, to_mono


def test_resample_identity():
    a = np.array([1, 2, 3], dtype=np.float32)
    assert np.array_equal(resample(a, 16000, 16000), a)


def test_resample_empty():
    assert resample(np.zeros(0, np.float32), 48000, 16000).size == 0


def test_resample_length_ratio_48k_to_16k():
    pytest.importorskip("soxr")  # real downsampling needs the audio extra
    a = np.zeros(4800, dtype=np.float32)  # 0.1s @ 48k
    out = resample(a, 48000, 16000)
    assert abs(len(out) - 1600) <= 1  # ~0.1s @ 16k


def test_resample_preserves_low_frequency():
    pytest.importorskip("soxr")  # real downsampling needs the audio extra
    sr_in, sr_out, f = 48000, 16000, 440.0
    t = np.arange(sr_in) / sr_in  # 1 second
    sig = np.sin(2 * np.pi * f * t).astype(np.float32)
    out = resample(sig, sr_in, sr_out)
    spec = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), 1 / sr_out)[int(np.argmax(spec))]
    assert abs(peak_hz - f) < 30  # dominant tone preserved


def test_to_mono_passthrough_1d():
    assert np.array_equal(to_mono(np.array([1.0, 2.0])), np.array([1.0, 2.0]))


def test_to_mono_averages_channels_last_axis():
    # sounddevice layout: (frames, channels)
    stereo = np.array([[0.0, 2.0], [1.0, 3.0], [4.0, 6.0]], dtype=np.float32)  # 3 frames, 2 ch
    out = to_mono(stereo)
    assert out.shape == (3,)
    assert np.allclose(out, [1.0, 2.0, 5.0])


def test_to_mono_rejects_3d():
    with pytest.raises(ValueError):
        to_mono(np.zeros((2, 2, 2), dtype=np.float32))


def test_scrubs_nan_inf():
    assert np.isfinite(to_mono(np.array([np.nan, 1.0, np.inf]))).all()
    assert np.isfinite(resample(np.array([0.0, np.nan, np.inf], np.float32), 16000, 16000)).all()


def test_resample_downsample_requires_soxr(monkeypatch):
    monkeypatch.setitem(sys.modules, "soxr", None)  # simulate soxr missing
    with pytest.raises(RuntimeError, match="soxr"):
        resample(np.zeros(4800, np.float32), 48000, 16000)


def test_resample_upsample_fallback_is_linear(monkeypatch):
    monkeypatch.setitem(sys.modules, "soxr", None)
    out = resample(np.zeros(1600, np.float32), 16000, 48000)  # upsample is safe w/o soxr
    assert abs(len(out) - 4800) <= 1
