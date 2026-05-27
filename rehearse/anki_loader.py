"""Parse AnkiApp XML deck exports into a unified sentence bank.

AnkiApp exports a deck as XML shaped like:

    <deck name="GoogleNews">
      <fields>
        <text name="Text"        sides="11" lang="en-US"/>
        <text name="Translation" sides="01" lang="zh-CN"/>
        <text name="genocide"    sides="01" lang="en-US"/>
      </fields>
      <cards>
        <card>
          <text name="Text">it just didn't appeal to me</text>
          <text name="Translation">我对它没什么兴趣</text>
          <text name="genocide">I have no interest in it</text>
        </card>
        ...
      </cards>
    </deck>

Two real-world quirks this parser handles (verified against the user's decks):

1. Field elements are <text> OR <rich-text>; rich-text values may contain HTML
   (`<p>...</p>`, entities). We strip markup and collapse whitespace.

2. The <fields> `lang` attribute LIES. The "forgettable1105" deck declares its
   Back field `lang="zh-CN"` but fills it with English. So roles are inferred
   from CONTENT, not the declared lang:
     - practice sentence = the front-side field (sides[0] == "1")
     - translation       = a field whose CONTENT contains CJK characters
     - native rephrase    = another English field whose content differs from the
                            practice sentence (e.g. GoogleNews "genocide")
"""

from __future__ import annotations

import html
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[一-鿿぀-ヿ가-힯]")


def _clean(text: str | None) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _has_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _norm_key(text: str) -> str:
    """Normalized dedup key: lowercase, trailing punctuation/space stripped."""
    return _WS_RE.sub(" ", text).strip().lower().rstrip(".!?,; ")


@dataclass
class Sentence:
    """One practice sentence drawn from a card."""

    text: str  # the English sentence to practice (front side)
    translation: str | None  # zh translation, if the card had one
    native: str | None  # a more-natural English rephrase, if present
    deck: str  # source deck name
    card_index: int  # position within the deck (stable id within a deck)

    @property
    def key(self) -> str:
        return _norm_key(self.text)


def parse_deck(path: str | Path) -> list[Sentence]:
    """Parse a single AnkiApp XML export into Sentences."""
    path = Path(path)
    root = ET.parse(path).getroot()
    deck_name = root.get("name") or path.stem

    # Determine which field is the front/prompt field from the schema.
    front_field: str | None = None
    fields_el = root.find("fields")
    if fields_el is not None:
        for f in fields_el:
            name = f.get("name") or f.tag
            if front_field is None and (f.get("sides") or "")[:1] == "1":
                front_field = name

    sentences: list[Sentence] = []
    cards = root.find("cards")
    if cards is None:
        return sentences

    for idx, card in enumerate(cards):
        values: dict[str, str] = {}
        for el in card:
            name = el.get("name") or el.tag
            values[name] = _clean(el.text)

        # practice sentence: declared front field, else first non-empty field
        text = values.get(front_field or "", "")
        if not text:
            text = next((v for v in values.values() if v), "")
        if not text:
            continue  # skip cards with no usable content

        # translation: first field whose content is CJK
        translation = next(
            (v for v in values.values() if v and _has_cjk(v)), None
        )

        # native rephrase: another English field that differs from the sentence
        native = next(
            (
                v
                for v in values.values()
                if v
                and not _has_cjk(v)
                and _norm_key(v) != _norm_key(text)
            ),
            None,
        )

        sentences.append(
            Sentence(
                text=text,
                translation=translation,
                native=native,
                deck=deck_name,
                card_index=idx,
            )
        )
    return sentences


def load_sentences(paths: list[str | Path], dedup: bool = True) -> list[Sentence]:
    """Load and merge multiple decks. Dedups by normalized English text.

    On duplicate, the first occurrence wins but absorbs a translation/native
    from a later duplicate if the winner lacked one.
    """
    out: list[Sentence] = []
    seen: dict[str, Sentence] = {}
    for p in paths:
        for s in parse_deck(p):
            if not dedup:
                out.append(s)
                continue
            existing = seen.get(s.key)
            if existing is None:
                seen[s.key] = s
                out.append(s)
            else:
                if existing.translation is None and s.translation:
                    existing.translation = s.translation
                if existing.native is None and s.native:
                    existing.native = s.native
    return out


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Parse AnkiApp XML decks.")
    ap.add_argument("paths", nargs="+", help="XML deck files")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--sample", type=int, default=5, help="print N samples")
    args = ap.parse_args(argv)

    sentences = load_sentences(args.paths, dedup=not args.no_dedup)
    n_tr = sum(1 for s in sentences if s.translation)
    n_nat = sum(1 for s in sentences if s.native)
    print(f"loaded {len(sentences)} sentences "
          f"({n_tr} with translation, {n_nat} with native rephrase)")
    for s in sentences[: args.sample]:
        print(f"  [{s.deck}#{s.card_index}] {s.text!r}"
              + (f"  | zh={s.translation!r}" if s.translation else "")
              + (f"  | native={s.native!r}" if s.native else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
