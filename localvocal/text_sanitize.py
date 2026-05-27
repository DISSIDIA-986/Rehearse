"""Strip markdown / emoji so the TTS never reads symbols aloud.

The LLM is instructed to emit plain speech, but defense-in-depth: we sanitize
its output before synthesis. Keeps the spoken words, drops the markup.
"""

from __future__ import annotations

import re

# Emoji + pictographic ranges (not exhaustive, covers the common cases).
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flags)
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U00002190-\U000021FF"  # arrows
    "\U0000200D"             # zero-width joiner (emoji sequences)
    "\U000020E3"             # combining keycap
    "]",
    flags=re.UNICODE,
)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_AUTOLINK_RE = re.compile(r"<(?:https?|mailto):[^>\s]+>")  # <http://x> -> drop (don't speak URLs)
_DASH_RE = re.compile(r"\s*[—–]\s*")  # em/en dash -> comma pause for natural prosody
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) -> text
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")  # ![alt](url) -> alt
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_LIST_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_LIST_NUM_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3}|~{2})(.+?)\1", re.DOTALL)  # **b** _i_ ~~s~~
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{2,}")


def sanitize_for_tts(text: str) -> str:
    """Return speakable plain text: markdown removed, emoji removed."""
    if not text:
        return ""
    text = _AUTOLINK_RE.sub("", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)  # <b>hi</b> -> hi
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    # repeat emphasis pass twice to catch nested (***bold italic***)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _LIST_BULLET_RE.sub("", text)
    text = _LIST_NUM_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = _DASH_RE.sub(", ", text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n", text)
    return text.strip()
