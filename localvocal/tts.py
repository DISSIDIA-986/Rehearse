"""Text-to-speech via Kokoro-82M on Apple GPU (mlx-audio).

Design D1: TTS runs AFTER the full short reply (serialized with the LLM on the
GPU). Kokoro outputs 24 kHz mono; the audio pipeline resamples up to the device
rate for playback. mlx-audio is imported lazily so the package imports without
the `audio` extra.

NOTE: mlx-audio's programmatic API has shifted across versions; `synth()` is kept
thin and is verified by the TTS->ASR round-trip integration test.
"""

from __future__ import annotations

import numpy as np

SR = 24_000
DEFAULT_MODEL = "prince-canuma/Kokoro-82M"
DEFAULT_VOICE = "af_heart"


class KokoroTTS:
    sr = SR

    def __init__(self, model_id: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE):
        from mlx_audio.tts.utils import load_model

        self._model = load_model(model_id)
        self.voice = voice

    def synth(self, text: str, speed: float = 1.0) -> np.ndarray:
        """Synthesize text to a 24 kHz mono float32 numpy array."""
        text = (text or "").strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        chunks: list[np.ndarray] = []
        for seg in self._model.generate(text=text, voice=self.voice, speed=speed):
            audio = getattr(seg, "audio", seg)
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)
