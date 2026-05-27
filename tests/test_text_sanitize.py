from rehearse.text_sanitize import sanitize_for_tts


def test_emphasis_stripped():
    assert sanitize_for_tts("**hi** _there_") == "hi there"
    assert sanitize_for_tts("***bold italic***") == "bold italic"
    assert sanitize_for_tts("~~strike~~") == "strike"


def test_links_and_images():
    assert sanitize_for_tts("see [Google](http://g.com) now") == "see Google now"
    assert sanitize_for_tts("![a cat](x.png)") == "a cat"


def test_code():
    assert sanitize_for_tts("run `npm test` ok") == "run npm test ok"
    assert sanitize_for_tts("a ```\nblock\n``` b").replace("  ", " ") == "a b"


def test_headings_lists_quotes():
    assert sanitize_for_tts("# Title") == "Title"
    assert sanitize_for_tts("- item one\n- item two") == "item one\nitem two"
    assert sanitize_for_tts("1. first\n2. second") == "first\nsecond"
    assert sanitize_for_tts("> quoted") == "quoted"


def test_emoji_removed():
    assert sanitize_for_tts("hello 👋🎉 world") == "hello world"
    assert sanitize_for_tts("nice 😀") == "nice"


def test_html_tags_and_autolinks():
    assert sanitize_for_tts("<b>hi</b> there") == "hi there"
    assert sanitize_for_tts("see <http://example.com> now") == "see now"
    assert sanitize_for_tts("<span class='x'>kept</span>") == "kept"


def test_dash_normalized():
    assert sanitize_for_tts("yes — really") == "yes, really"
    assert sanitize_for_tts("10–20 items") == "10, 20 items"


def test_empty_and_plain():
    assert sanitize_for_tts("") == ""
    assert sanitize_for_tts(None) == ""
    assert sanitize_for_tts("just plain words") == "just plain words"
