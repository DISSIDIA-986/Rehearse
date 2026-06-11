import queue

import numpy as np

from rehearse import audio_io
from rehearse.audio_io import RingBuffer
from rehearse.loop_core import (
    UtteranceAssembler,
    drain_utterance,
    io_rates,
    is_stop,
)
from rehearse.vad import EndpointConfig, EndpointDetector


def test_is_stop():
    for t in ["stop", "Stop.", "  goodbye!", "Bye.", "I'm done", "quit"]:
        assert is_stop(t), t
    for t in ["I have no interest in it", "let's keep going", "stopwatch"]:
        assert not is_stop(t), t


# --- io_rates: device-rate resolution with independent fallback --------------

class _FakeSd:
    """Scriptable sounddevice stand-in. rates maps kind -> samplerate or an
    Exception instance to raise."""

    def __init__(self, rates):
        self.rates = rates

    def query_devices(self, kind):
        r = self.rates[kind]
        if isinstance(r, Exception):
            raise r
        return {"default_samplerate": r}


def test_io_rates_reads_both_directions():
    in_sr, out_sr, block = io_rates(_FakeSd({"input": 48000, "output": 44100}))
    assert (in_sr, out_sr) == (48000, 44100)
    assert block == max(1, round(512 * 48000 / audio_io.ASR_SR))  # 1536


def test_io_rates_output_failure_keeps_good_input():
    # independent fallback: a broken output query must NOT discard the input rate
    in_sr, out_sr, _ = io_rates(_FakeSd({"input": 32000, "output": RuntimeError("x")}))
    assert in_sr == 32000
    assert out_sr == audio_io.DEFAULT_DEVICE_SR


def test_io_rates_input_failure_falls_back_only_input():
    in_sr, out_sr, _ = io_rates(_FakeSd({"input": KeyError("k"), "output": 44100}))
    assert in_sr == audio_io.DEFAULT_DEVICE_SR
    assert out_sr == 44100


def test_io_rates_both_fail_use_defaults():
    in_sr, out_sr, block = io_rates(
        _FakeSd({"input": OSError(), "output": OSError()}))
    assert in_sr == out_sr == audio_io.DEFAULT_DEVICE_SR
    assert block >= 1


# --- drain_utterance: queue -> single-resample utterance ---------------------

def test_drain_utterance_empty_is_none():
    assert drain_utterance(queue.Queue(), 16000) is None


def test_drain_utterance_concatenates_in_order():
    q: queue.Queue = queue.Queue()
    q.put(np.array([1.0, 2.0], dtype=np.float32))
    q.put(np.array([3.0], dtype=np.float32))
    out = drain_utterance(q, 16000)  # in==out rate -> no resample, exact concat
    assert np.array_equal(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert q.empty()  # fully drained


def test_drain_utterance_resamples_once_to_16k():
    import pytest
    pytest.importorskip("soxr")  # downsample path needs the audio extra
    q: queue.Queue = queue.Queue()
    q.put(np.zeros(4800, dtype=np.float32))  # 0.1s @ 48k
    out = drain_utterance(q, 48000)
    assert abs(len(out) - 1600) <= 1  # ~0.1s @ 16k


class _FakeVad:
    """Scripted speech probabilities, one per frame."""

    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0
        self.resets = 0

    def prob(self, _frame):
        p = self.probs[self.i] if self.i < len(self.probs) else 0.0
        self.i += 1
        return p

    def reset(self):
        self.resets += 1


def test_assembles_one_utterance_with_preroll():
    cfg = EndpointConfig(frame_ms=32, start_voiced_ms=64, end_silence_ms=300)
    # 2 silence, 5 voiced (start fires on the 2nd voiced frame = index 3),
    # then 10 silence (end fires on the 10th). Frame k carries the value k so we
    # can prove pre-roll onset is prepended and ordering is preserved.
    probs = [0.0, 0.0] + [0.9] * 5 + [0.0] * 10
    vad = _FakeVad(probs)
    asm = UtteranceAssembler(vad, EndpointDetector(cfg), RingBuffer(16000))

    outs = []
    for k in range(len(probs)):
        res = asm.push(np.full(512, float(k), dtype=np.float32))
        if res is not None:
            outs.append(res)

    assert len(outs) == 1  # exactly one utterance over the whole stream
    out = outs[0]
    assert out[0] == 0.0  # pre-roll prepended: onset (frame 0) is at the front
    assert np.any(out == 6.0)  # a voiced frame is included
    assert np.any(out == 16.0)  # the end frame is included
    assert out.size >= 10 * 512  # pre-roll + speaking + trailing, not just voiced
    assert vad.resets == 1  # reset after end


def test_no_utterance_on_pure_silence():
    cfg = EndpointConfig(frame_ms=32, start_voiced_ms=64, end_silence_ms=300)
    vad = _FakeVad([0.0] * 20)
    asm = UtteranceAssembler(vad, EndpointDetector(cfg), RingBuffer(16000))
    for _ in range(20):
        assert asm.push(np.zeros(512, np.float32)) is None
