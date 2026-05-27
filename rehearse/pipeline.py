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

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from rehearse.anki_loader import Sentence
from rehearse.llm_client import ChatResult, chat
from rehearse.practiced_scorer import EmbedFn, PracticeHit, ollama_embed, score_practiced
from rehearse.prompt_builder import build_system_prompt
from rehearse.sentence_chunker import chunk_sentences
from rehearse.text_sanitize import sanitize_for_tts

ChatFn = Callable[..., ChatResult]


@dataclass
class TurnResult:
    user_text: str
    reply_text: str
    reply_chunks: list[str]
    reply_audio: np.ndarray
    practiced: list[PracticeHit] = field(default_factory=list)
    ttft_s: float | None = None  # LLM time-to-first-token
    asr_s: float = 0.0  # ASR transcription wall time
    tts_s: float = 0.0  # TTS synthesis wall time
    practiced_error: str | None = None  # set if D3 scoring failed (surfaced, not swallowed)


@dataclass
class SpokenTurn:
    """One spoken round-trip, NO scoring — the shared primitive for every mode."""

    user_text: str
    reply_text: str
    reply_chunks: list[str]
    reply_audio: np.ndarray
    ttft_s: float | None = None
    asr_s: float = 0.0
    tts_s: float = 0.0


def speak_turn(
    user_audio_16k: np.ndarray,
    history: list[dict[str, str]],
    *,
    asr,
    tts,
    system_prompt: str,
    chat_fn: ChatFn = chat,
    num_predict: int | None = None,
) -> SpokenTurn:
    """ASR -> LLM -> sanitize -> chunk -> TTS. The mode-agnostic turn primitive
    (no scoring, no state) shared by english practice and markdown recall. Each
    mode does its own scoring on top of this. `history` is NOT mutated here.

    num_predict caps reply length in tokens (a hard cap is the only reliable way
    to keep a 4B model's replies short).
    """
    empty = np.zeros(0, dtype=np.float32)
    t0 = time.monotonic()
    user_text = asr.transcribe(user_audio_16k)
    asr_s = time.monotonic() - t0
    if not user_text:
        return SpokenTurn("", "", [], empty, asr_s=asr_s)

    messages = [{"role": "system", "content": system_prompt}, *history,
                {"role": "user", "content": user_text}]
    result = chat_fn(messages, num_predict=num_predict) if num_predict else chat_fn(messages)

    reply = sanitize_for_tts(result.text)
    chunks = chunk_sentences(reply)
    t1 = time.monotonic()
    pieces = [tts.synth(c) for c in chunks] if chunks else []
    audio = np.concatenate(pieces) if pieces else empty
    tts_s = time.monotonic() - t1
    return SpokenTurn(user_text, reply, chunks, audio, result.ttft_s, asr_s, tts_s)


def respond(
    user_audio_16k: np.ndarray,
    history: list[dict[str, str]],
    targets: list[Sentence],
    *,
    asr,
    tts,
    chat_fn: ChatFn = chat,
    embed: EmbedFn | None = ollama_embed,
    system_prompt: str | None = None,
    num_predict: int | None = None,
) -> TurnResult:
    """English-practice turn: a spoken turn + D3 'practiced' scoring.

    Behavior is unchanged from before the speak_turn() extraction (the Anki/
    English path stays isolated — markdown recall builds beside it, not on it).
    """
    system = system_prompt if system_prompt is not None else build_system_prompt(targets)
    st = speak_turn(user_audio_16k, history, asr=asr, tts=tts,
                    system_prompt=system, chat_fn=chat_fn, num_predict=num_predict)
    if not st.user_text:
        return TurnResult("", "", [], st.reply_audio, asr_s=st.asr_s)

    practiced: list[PracticeHit] = []
    practiced_error: str | None = None
    target_texts = [t.text for t in targets]
    if target_texts and embed is not None:  # embed=None -> scoring disabled (off critical path)
        try:
            practiced = score_practiced(st.user_text, target_texts, embed)
        except Exception as e:
            # never break the turn, but surface it (silently empty 'practiced'
            # would make the metric untrustworthy)
            practiced_error = f"{type(e).__name__}: {e}"

    return TurnResult(
        user_text=st.user_text,
        reply_text=st.reply_text,
        reply_chunks=st.reply_chunks,
        reply_audio=st.reply_audio,
        practiced=practiced,
        ttft_s=st.ttft_s,
        asr_s=st.asr_s,
        tts_s=st.tts_s,
        practiced_error=practiced_error,
    )
