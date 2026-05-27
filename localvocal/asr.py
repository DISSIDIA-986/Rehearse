"""Speech-to-text via faster-whisper (CPU, int8) — English only.

Design: ASR runs on the CPU (small.en, int8) so it does NOT compete with the
LLM/TTS for the Metal GPU (Codex blind-spot #1). faster-whisper is imported
lazily so the package imports without the heavy `audio` extra installed.
"""

from __future__ import annotations

import numpy as np

from localvocal.audio_io import ASR_SR

DEFAULT_MODEL = "small.en"


class WhisperASR:
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        from faster_whisper import WhisperModel

        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_16k_mono: np.ndarray, language: str = "en") -> str:
        """Transcribe 16 kHz mono float32 audio to text."""
        audio = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ""
        segments, _info = self._model.transcribe(
            audio, language=language, beam_size=1
        )
        return "".join(seg.text for seg in segments).strip()


# Expose the sample rate the model expects, for the audio pipeline.
EXPECTED_SR = ASR_SR
