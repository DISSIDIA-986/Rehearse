"""Deterministic tests for the speak_turn() primitive + respond() (no models).

Proves the T1 refactor: speak_turn is the shared ASR->LLM->TTS primitive (no
scoring), and respond() = speak_turn + practiced scoring with unchanged behavior.
Uses fakes so it runs without Ollama/Kokoro.
"""

import numpy as np

from localvocal.anki_loader import Sentence
from localvocal.llm_client import ChatResult
from localvocal.pipeline import SpokenTurn, respond, speak_turn


class _FakeASR:
    def __init__(self, text):
        self.text = text

    def transcribe(self, _audio):
        return self.text


class _FakeTTS:
    sr = 24000

    def synth(self, _text):
        return np.ones(10, dtype=np.float32)


def _fake_chat(messages, num_predict=None):
    # returns markdown to also prove sanitize runs inside the primitive
    return ChatResult(text="**Hello** there!", ttft_s=0.1, total_s=0.2, had_think_block=False)


def _audio():
    return np.ones(16000, dtype=np.float32)


def test_speak_turn_basic():
    st = speak_turn(_audio(), [], asr=_FakeASR("hi there"), tts=_FakeTTS(),
                    system_prompt="sys", chat_fn=_fake_chat)
    assert isinstance(st, SpokenTurn)
    assert st.user_text == "hi there"
    assert st.reply_text == "Hello there!"  # markdown stripped by sanitize
    assert st.reply_chunks == ["Hello there!"]
    assert st.reply_audio.size > 0
    assert st.ttft_s == 0.1 and st.asr_s >= 0 and st.tts_s >= 0


def test_speak_turn_empty_user_short_circuits():
    st = speak_turn(_audio(), [], asr=_FakeASR(""), tts=_FakeTTS(),
                    system_prompt="sys", chat_fn=_fake_chat)
    assert st.user_text == "" and st.reply_text == "" and st.reply_audio.size == 0


def test_respond_no_scoring_when_embed_none():
    tgt = [Sentence("hello there", None, None, "d", 0)]
    r = respond(_audio(), [], tgt, asr=_FakeASR("hello there"), tts=_FakeTTS(),
                chat_fn=_fake_chat, embed=None, system_prompt="sys")
    assert r.user_text == "hello there"
    assert r.reply_text == "Hello there!"  # same pipeline as before
    assert r.reply_audio.size > 0
    assert r.practiced == [] and r.practiced_error is None


def test_respond_scores_with_embed():
    def fake_embed(texts):
        return [[1.0, 0.0]] * len(texts)  # all identical -> cosine 1.0 -> hit

    tgt = [Sentence("hello there", None, None, "d", 0)]
    r = respond(_audio(), [], tgt, asr=_FakeASR("hello there"), tts=_FakeTTS(),
                chat_fn=_fake_chat, embed=fake_embed, system_prompt="sys")
    assert any(h.target == "hello there" for h in r.practiced)


def test_respond_empty_user():
    tgt = [Sentence("x", None, None, "d", 0)]
    r = respond(_audio(), [], tgt, asr=_FakeASR(""), tts=_FakeTTS(),
                chat_fn=_fake_chat, embed=None, system_prompt="sys")
    assert r.user_text == "" and r.practiced == [] and r.reply_audio.size == 0
