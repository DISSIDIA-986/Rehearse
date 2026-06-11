"""Real-model end-to-end integration (no mocks): TTS-synthesize a known Anki
target as the user's 'speech', run the real ASR -> D3 practiced scoring (real
nomic-embed) -> SQLite persistence -> cross-session select_targets reorder.

Gated behind REHEARSE_LLM_TESTS=1 AND model/Ollama availability, so routine
`uv run pytest` stays fast and CI without models still passes. Run here with:
    REHEARSE_LLM_TESTS=1 uv run pytest tests/test_integration_real.py -q -s
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from rehearse.anki_loader import Sentence


def _ready() -> bool:
    if os.environ.get("REHEARSE_LLM_TESTS") != "1":
        return False
    try:
        import faster_whisper  # noqa: F401
        import mlx_audio  # noqa: F401

        from rehearse.embeddings import ollama_embed
        ollama_embed(["ping"])  # nomic-embed reachable
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ready(), reason="REHEARSE_LLM_TESTS!=1 or TTS/ASR/Ollama unavailable")


def _stub_chat(messages, **kw):
    """The coach reply is irrelevant to scoring/persistence — stub it so the test
    exercises the REAL ASR + TTS + nomic-embed path without the 2.5GB MLX load."""
    from rehearse.llm_client import ChatResult
    return ChatResult(text="Nice — tell me more about that.",
                      ttft_s=0.1, total_s=0.2, had_think_block=False)


def test_real_practiced_scoring_persists_and_reorders(tmp_path):
    from rehearse import audio_io
    from rehearse.asr import WhisperASR
    from rehearse.embeddings import ollama_embed
    from rehearse.main_loop import apply_practiced
    from rehearse.pipeline import respond
    from rehearse.practice_store import PracticeStore
    from rehearse.prompt_builder import build_system_prompt
    from rehearse.session_seeder import select_targets
    from rehearse.tts import KokoroTTS

    asr = WhisperASR()
    tts = KokoroTTS()
    target = Sentence("It just didn't appeal to me.", None, None, "deck", 0)
    other = Sentence("See you later tonight.", None, None, "deck", 1)
    targets = [target, other]

    # the user "says" the target sentence (TTS it -> real audio -> feed the pipeline)
    audio = audio_io.resample(tts.synth(target.text), tts.sr, audio_io.ASR_SR)

    r = respond(audio, [], targets, asr=asr, tts=tts, chat_fn=_stub_chat,
                embed=ollama_embed, system_prompt=build_system_prompt(targets))

    print(f"\nASR heard: {r.user_text!r}")
    print(f"practiced: {[(h.target, round(h.similarity, 2)) for h in r.practiced]}")
    assert r.user_text, "ASR returned nothing for synthesized target speech"
    assert r.practiced_error is None
    # the spoken target must score a real practiced hit (real nomic cosine >= 0.50)
    assert target.text in {h.target for h in r.practiced}

    # persist via the same path the loop uses, then 'reopen' (next session)
    db = tmp_path / "practice.db"
    stats: dict = {}
    keys: set = set()
    _, _, turn_keys = apply_practiced(r.practiced, len(targets), targets, stats, keys, time.time())
    assert turn_keys, "apply_practiced produced no keys for the hit"
    with PracticeStore(db) as s:
        s.record_practiced(turn_keys, time.time())
    with PracticeStore(db) as s:
        stats2 = s.load_stats()

    assert stats2[target.key].count >= 1  # the target's practice persisted across 'sessions'
    # least-practiced first -> the untouched 'other' now leads scheduling
    assert select_targets(targets, stats2, n=1)[0].key == other.key


def test_real_unaided_detection_with_nomic():
    """T-P2-2b with real nomic-embed: the user spontaneously produces a
    previously-learned sentence that is NOT a steered target this turn -> it is
    detected as unaided production (and a non-candidate active target is excluded)."""
    from rehearse import audio_io
    from rehearse.asr import WhisperASR
    from rehearse.embeddings import ollama_embed
    from rehearse.session_seeder import PracticeStat
    from rehearse.tts import KokoroTTS
    from rehearse.unaided import UNAIDED_THRESHOLD, detect_unaided, select_candidates

    asr = WhisperASR()
    tts = KokoroTTS()
    learned = Sentence("The deadline is next Friday.", None, None, "deck", 0)
    active = Sentence("How was your weekend?", None, None, "deck", 1)

    audio = audio_io.resample(tts.synth(learned.text), tts.sr, audio_io.ASR_SR)
    user_text = asr.transcribe(audio)
    print(f"\nASR heard: {user_text!r}")

    stats = {learned.key: PracticeStat(count=2)}  # 'learned' practiced before; 'active' never
    cands = select_candidates(stats, [learned, active], active_keys={active.key})
    assert [c.key for c in cands] == [learned.key]  # only prior-practiced, non-active

    hit = detect_unaided(user_text, cands, ollama_embed)
    print(f"unaided hit: {None if hit is None else (hit.key, round(hit.similarity, 2))}")
    assert hit is not None and hit.key == learned.key
    assert hit.similarity >= UNAIDED_THRESHOLD
