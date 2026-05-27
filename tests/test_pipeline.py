"""Full conversation-turn integration test — real models, NO microphone (D4).

We synthesize the USER's speech with Kokoro, feed it through the real turn
pipeline (ASR -> LLM -> sanitize -> chunk -> TTS -> practiced scoring), and assert
each stage produced something sane. This proves the core loop works end-to-end
without any audio hardware. Skipped if the audio extra or Ollama is unavailable.
"""

import numpy as np
import pytest

from localvocal.audio_io import ASR_SR, resample

pytest.importorskip("mlx_audio", reason="audio extra not installed")
pytest.importorskip("faster_whisper", reason="audio extra not installed")

from localvocal.anki_loader import Sentence  # noqa: E402
from localvocal.asr import WhisperASR  # noqa: E402
from localvocal.llm_client import warmup  # noqa: E402
from localvocal.pipeline import respond  # noqa: E402
from localvocal.tts import KokoroTTS  # noqa: E402


def _ollama_up():
    try:
        warmup()
        return True
    except Exception:
        return False


_UP = _ollama_up()


@pytest.mark.skipif(not _UP, reason="Ollama / qwen3.5:4b not reachable")
def test_full_turn_pipeline():
    tts, asr = KokoroTTS(), WhisperASR()
    target = Sentence(
        text="I have no interest in it", translation=None,
        native="I have no interest in it", deck="d", card_index=0,
    )
    # synthesize the user's spoken turn (no mic)
    user_audio = resample(tts.synth("I have no interest in it."), tts.sr, ASR_SR)

    r = respond(user_audio, history=[], targets=[target], asr=asr, tts=tts)

    # ASR heard the user
    assert "interest" in r.user_text.lower()
    # LLM replied (short), TTS produced audio
    assert r.reply_text and r.reply_chunks
    assert r.reply_audio.dtype == np.float32 and r.reply_audio.size > 0
    assert np.max(np.abs(r.reply_audio)) > 0.01  # non-silent reply
    # D3: the user clearly used the target -> scored as practiced
    assert any(h.target == target.text for h in r.practiced)
    print(f"\n[turn] user={r.user_text!r}\n       reply={r.reply_text!r}\n"
          f"       ttft={r.ttft_s}s practiced={[round(h.similarity,2) for h in r.practiced]}")
