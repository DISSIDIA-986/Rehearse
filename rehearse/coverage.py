"""Coverage scoring for markdown-recall mode — honest, NOT 'semantic vibes'.

A bullet counts as covered only when the user's answer is BOTH semantically close
(nomic cosine >= threshold) AND recalls the bullet's hard facts. "Hard facts" =
its anchors (numbers + acronyms/CamelCase) when it has any; for anchorless prose
bullets we fall back to a lexical content-word overlap so vague on-topic talk
can't fully cover a point on cosine alone (final-review fix). Pure cosine
over-credits; the fact gate is what makes coverage honest (Codex's top F2 risk).

Anchors are matched as whole tokens, NOT substrings: "17" does NOT match "2017"
(final-review HIGH fix). Coverage is sticky per bullet across the item's turns —
once recalled it stays recalled, so a long rambling answer can't un-cover an
earlier point, and we cap the cumulative answer so re-embedding stays bounded.
Threshold is its own knob, separate from the sentence scorer (C7).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from rehearse.embeddings import EmbedFn, cosine, ollama_embed
from rehearse.practice_item import PracticeItem

# Calibrated on real nomic-embed-text (2026-05-27): an answer that genuinely
# covers a bullet scores 0.89-0.93; vague-but-related 0.49-0.55; unrelated ~0.40.
# 0.62 sits well below "good" and above "vague"; the fact gate catches the rest.
DEFAULT_THRESHOLD = 0.62
CW_RATIO = 0.5          # anchorless bullets: fraction of content words that must be recalled
MAX_CUM_CHARS = 4000    # cap cumulative answer so re-embed cost/memory stays bounded

# numbers (17, 3.96, 99.9%, 10K) and acronyms/CamelCase (NPV, BM25, LangChain).
_NUM = re.compile(r"\d[\d.,:/%kKmMbB+-]*")
_ACRONYM = re.compile(r"\b(?:[A-Z]{2,}[a-z]?|[A-Za-z]*[A-Z][a-z]*[A-Z][A-Za-z]*)\b")
_WORD = re.compile(r"[a-z]{3,}")
_STOP = frozenset(
    "the a an and or but of to in on at for with by from as is are was were be been "
    "being it its this that these those they them their our your you i we he she his "
    "her not no do did does had has have will would can could should may might into "
    "over about than then them out up down off all any some most more very just so".split()
)


def extract_anchors(text: str) -> set[str]:
    """The hard facts a genuine recall must include: numbers + acronyms/CamelCase,
    normalised to lowercase tokens. Used on BOTH the bullet and the answer so
    matching is whole-token (no '17' inside '2017')."""
    out: set[str] = set()
    for m in _NUM.finditer(text):
        tok = m.group().strip(".,:/").lower()
        if any(c.isdigit() for c in tok):
            out.add(tok)
    for m in _ACRONYM.finditer(text):
        tok = m.group()
        if len(tok) >= 2 and not tok.islower():  # has uppercase signal
            out.add(tok.lower())
    return {a for a in out if len(a) >= 2}


def _content_words(text: str) -> set[str]:
    """Significant lowercase words (>=3 chars, minus stopwords) — the lexical gate
    for prose bullets that have no numeric/acronym anchors."""
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def has_substance(point: str) -> bool:
    """True if a point carries something recallable (a fact anchor or a content
    word). A point that is only stopwords/punctuation is NOT a key point — it has
    nothing to recall, so it must be filtered out rather than scored on cosine
    alone (final-review: no fact gate is possible for it). Shared by the extractor
    and the session so a content-free 'point' never reaches scoring."""
    return bool(extract_anchors(point) or _content_words(point))


_RANK = {"miss": 0, "partial": 1, "hit": 2}


@dataclass
class BulletScore:
    bullet: str
    similarity: float
    anchors_total: int   # facts to recall (anchors, or content words if anchorless)
    anchors_hit: int     # facts recalled so far
    status: str          # "hit" | "partial" | "miss"


@dataclass
class ItemCoverage:
    item_key: str
    section: str
    cumulative_answer: str
    bullets: list[BulletScore] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.bullets) and all(b.status == "hit" for b in self.bullets)


@dataclass
class Summary:
    total_bullets: int
    hit_bullets: int
    items_attempted: int
    items_total: int
    weak_sections: list[str]

    def __str__(self) -> str:
        pct = round(100 * self.hit_bullets / self.total_bullets) if self.total_bullets else 0
        s = (f"covered {self.hit_bullets}/{self.total_bullets} key points ({pct}%); "
             f"{self.items_attempted}/{self.items_total} items attempted")
        if self.weak_sections:
            s += f"; weak: {', '.join(self.weak_sections[:4])}"
        return s


class CoverageTracker:
    """Per-item, cumulative, sticky, bullet-level coverage. embed is injectable."""

    def __init__(self, items, *, embed: EmbedFn = ollama_embed,
                 threshold: float = DEFAULT_THRESHOLD):
        self.items = list(items)
        self.embed = embed
        self.threshold = threshold
        self._cum: dict[str, str] = {}
        # score only substantive points (content-free 'points' have nothing to recall)
        self._pts = {it.key: [p for p in it.expected_points if has_substance(p)] for it in self.items}
        self._anchors = {k: [extract_anchors(p) for p in pts] for k, pts in self._pts.items()}
        self._cwords = {k: [_content_words(p) for p in pts] for k, pts in self._pts.items()}
        self._bullet_vecs: dict[str, list] = {}  # lazy per-item cache
        self._best: dict[str, list[BulletScore]] = {}  # sticky best score per bullet
        self.results: dict[str, ItemCoverage] = {}

    def _bvecs(self, item: PracticeItem):
        if item.key not in self._bullet_vecs:
            pts = self._pts.get(item.key, [])
            self._bullet_vecs[item.key] = self.embed(pts) if pts else []
        return self._bullet_vecs[item.key]

    def _score_bullet(self, pt, bvec, anchors, cwords, user_vec, ans_anchors, ans_cwords) -> BulletScore:
        sim = cosine(user_vec, bvec)
        sem = sim >= self.threshold
        if anchors:  # hard facts gate
            hit = len(anchors & ans_anchors)
            total = len(anchors)
            facts_ok = hit == total
            some = hit > 0
        elif cwords:  # prose bullet: lexical overlap gate
            inter = len(cwords & ans_cwords)
            total = len(cwords)
            facts_ok = inter >= max(1, round(CW_RATIO * total))
            some = inter > 0
            hit = inter
        else:  # nothing to gate on (e.g. all-stopword bullet): cosine is all we have
            total = hit = 0
            facts_ok = some = sem
        if sem and facts_ok:
            status = "hit"
        elif sem or some:
            status = "partial"
        else:
            status = "miss"
        return BulletScore(pt, sim, total, hit, status)

    def score(self, item: PracticeItem, user_text: str) -> ItemCoverage:
        """Fold user_text into the item's cumulative answer and rescore each bullet,
        keeping the best status seen so far (sticky)."""
        cum = (self._cum.get(item.key, "") + " " + (user_text or "")).strip()
        cum = cum[-MAX_CUM_CHARS:]  # bound re-embed cost; sticky scores below keep prior hits
        self._cum[item.key] = cum
        pts = self._pts.get(item.key, [])
        if not pts:
            cov = ItemCoverage(item.key, item.section, cum, [])
            self.results[item.key] = cov
            return cov

        cum_anchors = extract_anchors(cum)
        cum_cwords = _content_words(cum)
        user_vec = self.embed([cum])[0]
        fresh = [
            self._score_bullet(pt, bvec, anchors, cwords, user_vec, cum_anchors, cum_cwords)
            for pt, bvec, anchors, cwords in zip(
                pts, self._bvecs(item),
                self._anchors[item.key], self._cwords[item.key])
        ]
        prev = self._best.get(item.key)
        if prev is None:
            best = fresh
        else:  # sticky: keep the higher-ranked status / better numbers per bullet
            best = [
                f if _RANK[f.status] >= _RANK[p.status] else
                BulletScore(p.bullet, max(p.similarity, f.similarity), p.anchors_total,
                            max(p.anchors_hit, f.anchors_hit), p.status)
                for p, f in zip(prev, fresh)
            ]
        self._best[item.key] = best
        cov = ItemCoverage(item.key, item.section, cum, best)
        self.results[item.key] = cov
        return cov

    def summary(self) -> Summary:
        total = sum(len(self._pts.get(it.key, [])) for it in self.items)
        hit = sum(1 for cov in self.results.values() for b in cov.bullets if b.status == "hit")
        sec_hit: dict[str, int] = defaultdict(int)
        sec_tot: dict[str, int] = defaultdict(int)
        for cov in self.results.values():
            for b in cov.bullets:
                sec_tot[cov.section] += 1
                if b.status == "hit":
                    sec_hit[cov.section] += 1
        weak = [s for s in sec_tot if sec_tot[s] and sec_hit[s] < 0.5 * sec_tot[s]]
        return Summary(total, hit, len(self.results), len(self.items), weak)
