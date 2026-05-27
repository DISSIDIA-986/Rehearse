from localvocal.anki_loader import Sentence
from localvocal.practice_item import PracticeItem, from_sentence


def test_key_normalizes_and_prefers_id():
    assert PracticeItem(id="Q1", prompt="why?").key == "q1"
    assert PracticeItem(id="", prompt="  Why  Fit? ").key == "why fit?"


def test_defaults_minimal_schema():
    it = PracticeItem(id="a", prompt="p")
    assert it.expected_points == [] and it.support_snippets == []
    assert it.section == "" and it.kind == "recall"


def test_from_sentence_includes_native_when_different():
    s = Sentence(text="it didn't appeal to me", translation="zh",
                 native="I have no interest in it", deck="GoogleNews", card_index=3)
    it = from_sentence(s)
    assert it.kind == "sentence"
    assert it.prompt == "it didn't appeal to me"
    assert it.expected_points == ["it didn't appeal to me", "I have no interest in it"]
    assert it.id == "GoogleNews#3" and it.section == "GoogleNews"


def test_from_sentence_drops_duplicate_native():
    s = Sentence(text="Hello there.", translation=None, native="hello there.",
                 deck="d", card_index=0)
    it = from_sentence(s)
    assert it.expected_points == ["Hello there."]  # native == text (case-insensitive) dropped


def test_from_sentence_no_native():
    s = Sentence(text="a garden pavilion.", translation=None, native=None,
                 deck="forgettable", card_index=2)
    assert from_sentence(s).expected_points == ["a garden pavilion."]
