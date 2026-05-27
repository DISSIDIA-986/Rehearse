"""One conversation turn: user audio -> reply audio (the testable core).

This is the heart of the loop, deliberately separated from mic/playback/VAD so it
can be exercised end-to-end with real models but NO microphone (feed it TTS-
generated "user speech"). Order follows D1 (full reply, then chunk for TTS) and
D3 (score 'practiced' from the user's transcript vs the active targets).

    user_audio_16k
       -> ASR transcribe            (faster-whisper, CPU)
       -> build messages            (system prompt weaves in targets)
       -> LLM chat                  (Ollama qwen3.5:4b, think:false, short)
       -> sanitize + chunk          (plain speech for TTS)
       -> TTS synth                 (Kokoro, 24k)
       -> score practiced           (nomic cosine vs targets)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from localvocal.anki_loader import Sentence
from localvocal.llm_client import ChatResult, chat
from localvocal.practiced_scorer import EmbedFn, PracticeHit, ollama_embed, score_practiced
from localvocal.prompt_builder import build_system_prompt
from localvocal.sentence_chunker import chunk_sentences
from localvocal.text_sanitize import sanitize_for_tts

ChatFn = Callable[..., ChatResult]


@dataclass
class TurnResult:
    user_text: str
    reply_text: str
    reply_chunks: list[str]
    reply_audio: np.ndarray
    practiced: list[PracticeHit] = field(default_factory=list)
    ttft_s: float | None = None


def respond(
    user_audio_16k: np.ndarray,
    history: list[dict[str, str]],
    targets: list[Sentence],
    *,
    asr,
    tts,
    chat_fn: ChatFn = chat,
    embed: EmbedFn = ollama_embed,
    system_prompt: str | None = None,
) -> TurnResult:
    """Run one full turn. `history` is prior [{role,content}] (NOT mutated here)."""
    empty = np.zeros(0, dtype=np.float32)
    user_text = asr.transcribe(user_audio_16k)
    if not user_text:
        return TurnResult("", "", [], empty)

    system = system_prompt if system_prompt is not None else build_system_prompt(targets)
    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": user_text}]
    result = chat_fn(messages)

    reply = sanitize_for_tts(result.text)
    chunks = chunk_sentences(reply)
    pieces = [tts.synth(c) for c in chunks] if chunks else []
    audio = np.concatenate(pieces) if pieces else empty

    practiced: list[PracticeHit] = []
    target_texts = [t.text for t in targets]
    if target_texts:
        try:
            practiced = score_practiced(user_text, target_texts, embed)
        except Exception:
            practiced = []  # scoring is a nicety; never break the turn

    return TurnResult(
        user_text=user_text,
        reply_text=reply,
        reply_chunks=chunks,
        reply_audio=audio,
        practiced=practiced,
        ttft_s=result.ttft_s,
    )
