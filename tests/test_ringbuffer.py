import numpy as np

from rehearse.audio_io import RingBuffer


def test_keeps_last_max_samples():
    rb = RingBuffer(5)
    rb.push(np.array([1, 2, 3], np.float32))
    rb.push(np.array([4, 5, 6], np.float32))
    assert np.allclose(rb.get(), [2, 3, 4, 5, 6])  # sample-accurate trim


def test_under_capacity():
    rb = RingBuffer(10)
    rb.push(np.array([1, 2], np.float32))
    assert np.allclose(rb.get(), [1, 2])


def test_zero_capacity_and_clear():
    assert RingBuffer(0).get().size == 0
    rb = RingBuffer(8)
    rb.push(np.array([1, 2, 3], np.float32))
    rb.clear()
    assert rb.get().size == 0
