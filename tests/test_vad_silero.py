"""Silero VAD model test using synthesized speech — no microphone.

Feeds Kokoro-generated speech frames through SileroVad and asserts speech is
detected, and that silence is not. Skipped if the vad/audio extras are absent.
"""

import numpy as np
import pytest

pytest.importorskip("silero_vad", reason="vad extra not installed")
pytest.importorskip("mlx_audio", reason="audio extra not installed")

from localvocal.audio_io import ASR_SR, resample  # noqa: E402
from localvocal.tts import KokoroTTS  # noqa: E402
from localvocal.vad import SILERO_FRAME, SileroVad  # noqa: E402


@pytest.fixture(scope="module")
def vad():
    return SileroVad()


def test_silence_is_not_speech(vad):
    vad.reset()
    assert vad.prob(np.zeros(SILERO_FRAME, dtype=np.float32)) < 0.5


def test_kokoro_speech_is_detected(vad):
    vad.reset()
    audio = resample(KokoroTTS().synth("Hello there, how are you doing today?"), 24_000, ASR_SR)
    probs = [
        vad.prob(audio[i : i + SILERO_FRAME])
        for i in range(0, len(audio) - SILERO_FRAME, SILERO_FRAME)
    ]
    assert probs, "no frames"
    assert max(probs) > 0.6  # speech clearly detected
    assert sum(p > 0.5 for p in probs) >= 0.3 * len(probs)  # voiced for a good chunk
