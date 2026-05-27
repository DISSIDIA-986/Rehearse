"""Tests for the AnkiApp XML loader.

Deterministic tests run against committed synthetic fixtures (same FORMAT as the
user's real decks, fake content). A separate real-data smoke test runs only when
the private decks are present in data/ (skipped otherwise so the public CI passes).
"""

from pathlib import Path

import pytest

from rehearse.anki_loader import (
    Sentence,
    _clean,
    _has_cjk,
    load_sentences,
    parse_deck,
)

FIX = Path(__file__).parent / "fixtures"
RICH = FIX / "deck_richtext.xml"
TRANS = FIX / "deck_translation.xml"


# --- _clean ---------------------------------------------------------------

def test_clean_strips_html_and_entities():
    assert _clean("<p>The importance of education.</p>") == "The importance of education."
    assert _clean("a &amp; b") == "a & b"


def test_clean_collapses_whitespace_and_handles_none():
    assert _clean("  hello   world \n ") == "hello world"
    assert _clean(None) == ""
    assert _clean("   ") == ""


# --- _has_cjk -------------------------------------------------------------

def test_has_cjk():
    assert _has_cjk("我对它没什么兴趣")
    assert not _has_cjk("I have no interest")
    assert _has_cjk("mixed 中文 text")


# --- parse_deck: rich-text deck (lang lies, HTML present) -----------------

def test_parse_richtext_deck():
    s = parse_deck(RICH)
    # 3rd card is empty -> skipped
    assert len(s) == 2

    # card 0: Front == Back, both English -> no translation, no native
    assert s[0].text == "Worries oppress his spirit."
    assert s[0].translation is None
    assert s[0].native is None
    assert s[0].deck == "RichDeck"
    assert s[0].card_index == 0

    # card 1: Front is HTML-wrapped full sentence -> stripped; Back is a
    # different English fragment -> native. Back declared zh-CN but is English,
    # so it must NOT be treated as a translation.
    assert s[1].text == "The importance of education cannot be overemphasized."
    assert s[1].translation is None
    assert s[1].native == "emphasize the importance of education."


# --- parse_deck: translation deck (CJK + native rephrase) -----------------

def test_parse_translation_deck():
    s = parse_deck(TRANS)
    assert len(s) == 2
    assert s[0].text == "it just didn't appeal to me"
    assert s[0].translation == "我对它没什么兴趣"
    assert s[0].native == "I have no interest in it"


# --- load_sentences: dedup + field absorption -----------------------------

def test_load_dedup_absorbs_translation_and_native():
    # "Worries oppress his spirit." appears in BOTH decks. RichDeck (loaded
    # first) lacks translation/native; TransDeck's dup has both -> absorbed.
    s = load_sentences([RICH, TRANS], dedup=True)
    # RichDeck(2) + TransDeck(2) - 1 dup = 3
    assert len(s) == 3
    worries = next(x for x in s if x.key == "worries oppress his spirit")
    assert worries.translation == "忧虑压迫着他的精神。"
    assert worries.native == "He is weighed down by worry."


def test_load_no_dedup_keeps_all():
    assert len(load_sentences([RICH, TRANS], dedup=False)) == 4


def test_sentence_key_normalizes():
    assert Sentence("Hello, World! ", None, None, "d", 0).key == "hello, world"


# --- real-data smoke test (private decks; skipped if absent) --------------

DATA = Path(__file__).parent.parent / "data"
REAL = [DATA / "forgettable1105.xml", DATA / "GoogleNews.xml"]


@pytest.mark.skipif(
    not all(p.exists() for p in REAL),
    reason="private Anki decks not present in data/ (expected on public CI)",
)
def test_real_decks_load():
    s = load_sentences(REAL, dedup=True)
    # 718 + 1148 = 1866 cards; dedup (punctuation-normalized) -> ~1843 unique
    assert 1800 < len(s) < 1866
    # GoogleNews supplies CJK translations (1142) + native rephrases where the
    # "genocide" field differs from Text (~800; many cards never got rephrased)
    assert sum(1 for x in s if x.translation) > 1000
    assert sum(1 for x in s if x.native) > 700
    # every sentence has non-empty practice text
    assert all(x.text.strip() for x in s)
