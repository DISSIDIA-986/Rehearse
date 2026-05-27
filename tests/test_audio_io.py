import numpy as np

from localvocal.audio_io import resample, to_mono


def test_resample_identity():
    a = np.array([1, 2, 3], dtype=np.float32)
    assert np.array_equal(resample(a, 16000, 16000), a)


def test_resample_empty():
    assert resample(np.zeros(0, np.float32), 48000, 16000).size == 0


def test_resample_length_ratio_48k_to_16k():
    a = np.zeros(4800, dtype=np.float32)  # 0.1s @ 48k
    out = resample(a, 48000, 16000)
    assert abs(len(out) - 1600) <= 1  # ~0.1s @ 16k


def test_resample_preserves_low_frequency():
    sr_in, sr_out, f = 48000, 16000, 440.0
    t = np.arange(sr_in) / sr_in  # 1 second
    sig = np.sin(2 * np.pi * f * t).astype(np.float32)
    out = resample(sig, sr_in, sr_out)
    spec = np.abs(np.fft.rfft(out))
    peak_hz = np.fft.rfftfreq(len(out), 1 / sr_out)[int(np.argmax(spec))]
    assert abs(peak_hz - f) < 30  # dominant tone preserved


def test_to_mono():
    assert np.array_equal(to_mono(np.array([1.0, 2.0])), np.array([1.0, 2.0]))
    stereo = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)  # (n=2, ch=2)
    assert np.allclose(to_mono(stereo), [1.0, 2.0])
