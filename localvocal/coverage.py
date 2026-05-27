"""Coverage scoring for markdown-recall mode — honest, NOT 'semantic vibes'.

A bullet counts as covered only when the user's CUMULATIVE answer for the current
item is BOTH semantically close (nomic cosine >= threshold) AND mentions the
bullet's hard facts (anchors: numbers, acronyms / CamelCase). Pure cosine
over-credits vague talk; the anchor gate is what makes coverage honest (Codex's
top F2 risk). Threshold is its own knob, separate from the sentence scorer (C7).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from localvocal.embeddings import EmbedFn, cosine, ollama_embed
from localvocal.practice_item import PracticeItem

# Calibrated on real nomic-embed-text (2026-05-27): an answer that genuinely
# covers a bullet scores 0.89-0.93; vague-but-related 0.49-0.55; unrelated ~0.40.
# 0.62 sits well below "good" and above "vague"; the anchor gate catches the rest.
DEFAULT_THRESHOLD = 0.62

# numbers (17, 3.96, 99.9%, 10K) and acronyms/CamelCase (NPV, BM25, LangChain).
_NUM = re.compile(r"\d[\d.,:/%kKmMbB+-]*")
_ACRONYM = re.compile(r"\b(?:[A-Z]{2,}[a-z]?|[A-Za-z]*[A-Z][a-z]*[A-Z][A-Za-z]*)\b")


def extract_anchors(text: str) -> set[str]:
    """The hard facts a genuine recall must include: numbers + acronyms/CamelCase."""
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


@dataclass
class BulletScore:
    bullet: str
    similarity: float
    anchors_total: int
    anchors_hit: int
    status: str  # "hit" | "partial" | "miss"


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
    """Per-item, cumulative, bullet-level coverage. embed is injectable for tests."""

    def __init__(self, items, *, embed: EmbedFn = ollama_embed,
                 threshold: float = DEFAULT_THRESHOLD):
        self.items = list(items)
        self.embed = embed
        self.threshold = threshold
        self._cum: dict[str, str] = {}
        self._anchors = {it.key: [extract_anchors(p) for p in it.expected_points] for it in self.items}
        self._bullet_vecs: dict[str, list] = {}  # lazy per-item cache
        self.results: dict[str, ItemCoverage] = {}

    def _bvecs(self, item: PracticeItem):
        if item.key not in self._bullet_vecs:
            self._bullet_vecs[item.key] = self.embed(item.expected_points) if item.expected_points else []
        return self._bullet_vecs[item.key]

    def score(self, item: PracticeItem, user_text: str) -> ItemCoverage:
        """Fold user_text into the item's cumulative answer and rescore each bullet."""
        cum = (self._cum.get(item.key, "") + " " + (user_text or "")).strip()
        self._cum[item.key] = cum
        if not item.expected_points:
            cov = ItemCoverage(item.key, item.section, cum, [])
            self.results[item.key] = cov
            return cov

        cum_lower = cum.lower()
        user_vec = self.embed([cum])[0]
        scores: list[BulletScore] = []
        for pt, bvec, anchors in zip(item.expected_points, self._bvecs(item), self._anchors[item.key]):
            sim = cosine(user_vec, bvec)
            hit_anchors = sum(1 for a in anchors if a in cum_lower)
            sem = sim >= self.threshold
            anchors_ok = hit_anchors == len(anchors)  # vacuously true when no anchors
            if sem and anchors_ok:
                status = "hit"
            elif sem or hit_anchors > 0:
                status = "partial"  # semantically close but missing facts, or facts w/o the gist
            else:
                status = "miss"
            scores.append(BulletScore(pt, sim, len(anchors), hit_anchors, status))
        cov = ItemCoverage(item.key, item.section, cum, scores)
        self.results[item.key] = cov
        return cov

    def summary(self) -> Summary:
        total = sum(len(it.expected_points) for it in self.items)
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
