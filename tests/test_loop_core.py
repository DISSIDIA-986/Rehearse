import numpy as np

from localvocal.audio_io import RingBuffer
from localvocal.loop_core import UtteranceAssembler, is_stop
from localvocal.vad import EndpointConfig, EndpointDetector


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
    # 2 silence, 5 voiced (start at 2nd), 10 silence (end at 10th)
    probs = [0.0, 0.0] + [0.9] * 5 + [0.0] * 10
    vad = _FakeVad(probs)
    asm = UtteranceAssembler(vad, EndpointDetector(cfg), RingBuffer(16000))

    out = None
    for k in range(len(probs)):
        frame = np.full(512, 0.1, dtype=np.float32)
        res = asm.push(frame)
        if res is not None:
            out = res
    assert out is not None and out.size > 0  # exactly one utterance assembled
    assert vad.resets == 1  # reset after end


def test_no_utterance_on_pure_silence():
    cfg = EndpointConfig(frame_ms=32, start_voiced_ms=64, end_silence_ms=300)
    vad = _FakeVad([0.0] * 20)
    asm = UtteranceAssembler(vad, EndpointDetector(cfg), RingBuffer(16000))
    for _ in range(20):
        assert asm.push(np.zeros(512, np.float32)) is None
