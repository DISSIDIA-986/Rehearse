"""Speech-to-text via faster-whisper (CPU, int8) — English only.

Design: ASR runs on the CPU (small.en, int8) so it does NOT compete with the
LLM/TTS for the Metal GPU (Codex blind-spot #1). faster-whisper is imported
lazily so the package imports without the heavy `audio` extra installed.
"""

from __future__ import annotations

import numpy as np

from rehearse.audio_io import ASR_SR

DEFAULT_MODEL = "small.en"
MIN_SAMPLES = int(0.1 * ASR_SR)  # <100ms: too short, whisper would hallucinate


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
        """Transcribe 16 kHz mono float32 audio to text.

        Tuned for accented speech on a close mic + speaker setup (DJI + Mac):
        - vad_filter strips internal silence/noise before decode (the assembled
          utterance keeps internal pauses; silence makes Whisper hallucinate/repeat).
        - beam_size=5 trades a little latency for accuracy on accented input.
        - condition_on_previous_text=False stops runaway repetition.
        """
        audio = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        if audio.size < MIN_SAMPLES:  # empty or too-short clip -> skip (no hallucination)
            return ""

        def _run(vad_filter: bool) -> str:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=vad_filter,
                vad_parameters={"min_silence_duration_ms": 350},
                condition_on_previous_text=False,
            )
            return "".join(seg.text for seg in segments).strip()

        text = _run(vad_filter=True)
        # vad_filter can wrongly strip a quiet/soft answer to nothing; if there's
        # clearly enough audio (>0.5s) but we got empty, retry without the filter.
        if not text and audio.size > ASR_SR // 2:
            text = _run(vad_filter=False)
        return text


# Expose the sample rate the model expects, for the audio pipeline.
EXPECTED_SR = ASR_SR
