import numpy as np

from rehearse.audio_io import RingBuffer
from rehearse.loop_core import UtteranceAssembler, is_stop
from rehearse.vad import EndpointConfig, EndpointDetector


def test_is_stop():
    for t in ["stop", "Stop.", "  goodbye!", "Bye.", "I'm done", "quit"]:
        assert is_stop(t), t
    for t in ["I have no interest in it", "let's keep going", "stopwatch"]:
        assert not is_stop(t), t


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
