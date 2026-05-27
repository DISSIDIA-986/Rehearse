"""A single thing to recall & elaborate, from ANY content source.

The neutral practice type shared by the markdown-recall mode (and adaptable from
Anki Sentences). Minimal schema per the fidelity-review (C4): no content_type, no
LLM-emitted anchors — anchors are derived deterministically at scoring time.

Privacy of the answer (C6): `expected_points` are for SCORING only and must NOT be
shown to the coach LLM; `support_snippets` are hints/scripts withheld until the
user explicitly stalls or asks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WS = re.compile(r"\s+")


@dataclass
class PracticeItem:
    id: str
    prompt: str  # the question / cue the user recalls against
    expected_points: list[str] = field(default_factory=list)  # scoring only, not shown
    support_snippets: list[str] = field(default_factory=list)  # hints, withheld until stall
    section: str = ""
    kind: str = "recall"  # "recall" (markdown) | "sentence" (anki adapter)
    source_title: str = ""

    @property
    def key(self) -> str:
        """Stable dedup/stats key (duck-types with Sentence.key for session_seeder)."""
        return _WS.sub(" ", (self.id or self.prompt)).strip().lower()


def from_sentence(s) -> PracticeItem:
    """Adapt an Anki Sentence into a PracticeItem (uniform recall type).

    Keeps the Anki/English path itself unchanged — this is only for code that
    wants to treat both sources through one type.
    """
    native = getattr(s, "native", None)
    points = [s.text]
    if native and native.strip().lower() != s.text.strip().lower():
        points.append(native)
    return PracticeItem(
        id=f"{s.deck}#{s.card_index}",
        prompt=s.text,
        expected_points=points,
        section=s.deck,
        kind="sentence",
        source_title=s.deck,
    )
