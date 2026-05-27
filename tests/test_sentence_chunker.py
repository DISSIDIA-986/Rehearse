from rehearse.sentence_chunker import chunk_sentences


def test_basic_split():
    assert chunk_sentences("Hello world. How are you?") == [
        "Hello world.",
        "How are you?",
    ]


def test_abbreviation_not_split():
    assert chunk_sentences("Mr. Smith left early. Bye.") == [
        "Mr. Smith left early.",
        "Bye.",
    ]
    assert chunk_sentences("e.g. this works fine.") == ["e.g. this works fine."]


def test_no_trailing_punctuation():
    assert chunk_sentences("just a fragment") == ["just a fragment"]


def test_cap_long_punctuationless():
    text = "word " * 60  # 300 chars, no sentence punctuation
    out = chunk_sentences(text.strip(), max_chars=100)
    assert len(out) >= 3
    assert all(len(c) <= 100 for c in out)


def test_chinese_punctuation():
    assert chunk_sentences("你好。再见！") == ["你好。", "再见！"]


def test_closing_quote_after_punctuation():
    assert chunk_sentences('He said "Hi." Then left.') == [
        'He said "Hi."',
        "Then left.",
    ]


def test_decimal_not_split():
    assert chunk_sentences("It costs 3.14 dollars today.") == [
        "It costs 3.14 dollars today.",
    ]


def test_empty():
    assert chunk_sentences("") == []
    assert chunk_sentences("   ") == []
