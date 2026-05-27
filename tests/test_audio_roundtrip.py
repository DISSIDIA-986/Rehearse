"""TTS -> resample -> ASR round-trip integration test (D4, no microphone).

Kokoro synthesizes a known sentence; soxr resamples 24k->16k; faster-whisper
transcribes it back. Asserts the words come through. This exercises the real
audio chain (the seams Codex flagged: sample-rate conversion, model wiring)
without any audio hardware. Skipped if the `audio` extra is not installed.
"""

import numpy as np
import pytest

from rehearse.audio_io import ASR_SR, resample

audio_deps = pytest.importorskip  # alias for readability
audio_deps("mlx_audio", reason="audio extra not installed (uv sync --extra audio)")
audio_deps("faster_whisper", reason="audio extra not installed")

from rehearse.asr import WhisperASR  # noqa: E402
from rehearse.tts import KokoroTTS  # noqa: E402


def _words(s: str) -> set[str]:
    return set(s.lower().replace(".", "").replace(",", "").split())


def _recall(phrase: str, text: str) -> float:
    """Fraction of the input words that survived to the transcript."""
    wp = _words(phrase)
    return len(wp & _words(text)) / max(1, len(wp))


@pytest.fixture(scope="module")
def models():
    return KokoroTTS(), WhisperASR()


@pytest.mark.parametrize(
    "phrase",
    [
        "The weather is really nice today.",
        "I have no interest in it.",
    ],
)
def test_tts_asr_roundtrip(models, phrase):
    tts, asr = models
    audio = tts.synth(phrase)
    assert audio.dtype == np.float32 and audio.size > 0
    assert np.max(np.abs(audio)) > 0.01  # non-silent
    a16 = resample(audio, tts.sr, ASR_SR)
    text = asr.transcribe(a16)
    assert text  # got a transcription
    # full recall: every input word (incl. negation "no") must survive the chain.
    # Catches dropped-word regressions that a loose set-overlap would mask.
    assert _recall(phrase, text) == 1.0, f"dropped words: {_words(phrase) - _words(text)}"
