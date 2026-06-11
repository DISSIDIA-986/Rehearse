"""Split an LLM reply into speakable chunks.

Used by the (D1 full-serialize) loop: after the complete short reply is in hand,
we feed it to TTS sentence-by-sentence so playback can start on the first chunk.
Avoids splitting inside common abbreviations; caps runaway punctuation-less text.
"""

from __future__ import annotations

import re

# Latin enders must be followed by whitespace/end (so "3.14" and "e.g." don't
# split), with optional trailing closing quote/bracket (He said "Hi." Then...).
# CJK enders (。！？…) are unambiguous boundaries on their own.
_CLOSERS = r"[\"'”’)\]]*"
_SENT_END_RE = re.compile(rf"[.!?]+{_CLOSERS}(?=\s|$)|[。！？…]+{_CLOSERS}")

_ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
    "etc.", "e.g.", "i.e.", "u.s.", "u.k.", "a.m.", "p.m.", "no.", "fig.",
    "inc.", "ltd.", "co.", "corp.", "dept.", "approx.", "jan.", "feb.",
    "mar.", "apr.", "jun.", "jul.", "aug.", "sep.", "sept.", "oct.",
    "nov.", "dec.", "ph.d.", "b.a.", "m.a.", "a.k.a.",
}


def chunk_sentences(text: str, max_chars: int = 200) -> list[str]:
    """Split text into TTS chunks. Never splits on an abbreviation period."""
    text = (text or "").strip()
    if not text:
        return []

    sentences: list[str] = []
    start = 0
    for m in _SENT_END_RE.finditer(text):
        candidate = text[start:m.end(0)]
        words = candidate.split()
        last_word = words[-1].lower() if words else ""
        if last_word in _ABBREV:
            continue  # keep going; this period is an abbreviation
        sentences.append(candidate.strip())
        start = m.end(0)
    if start < len(text):
        sentences.append(text[start:].strip())

    # Cap long punctuation-less chunks at a word boundary.
    out: list[str] = []
    for s in (s for s in sentences if s):
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            head = s[:cut].strip()
            if head:
                out.append(head)
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out
