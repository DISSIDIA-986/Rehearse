"""Shadow-mode 'unaided production' detection (Phase 2, T-P2-2b).

D3 (practiced_scorer) scores PROMPTED hits — the user repeating/paraphrasing one
of THIS turn's injected targets (mastery ~7/10). The higher signal (10/10) is
*unaided* production: the user spontaneously using a previously-seen sentence
that was NOT being steered toward this turn.

This module detects that, but is wired in SHADOW MODE only (see main_loop's
--enable-unaided, default off): it records events/counts for later threshold
calibration and NEVER feeds select_targets, so a false positive cannot poison
the spaced-rep schedule. It also runs OFF the critical path (after playback), so
it adds no felt latency.

Anti-over-engineering (the 2000-scale embed cost that deferred T-P2-1): the
candidate set is bounded to PREVIOUSLY-PRACTICED sentences (count>0), excluding
this turn's active targets, ordered by due-ness and capped. A fresh user has zero
candidates -> zero cost. Threshold is stricter than D3's 0.50 (precision over
recall — an over-credit is worse than a miss here); at most one hit per turn.
"""

from __future__ import annotations

from dataclasses import dataclass

from rehearse.embeddings import EmbedFn, cosine

# Stricter than D3's 0.50: unaided credit must be a real reproduction, not a
# vaguely-related utterance. Deliberately conservative until calibrated on real
# session logs (practice_event table is the audit trail for tuning this).
UNAIDED_THRESHOLD = 0.65
MAX_CANDIDATES = 64  # bound the per-turn EMBEDDING + cosine work (the real cost)


@dataclass
class UnaidedHit:
    key: str
    text: str
    similarity: float


def select_candidates(stats, sentences, active_keys, cap: int = MAX_CANDIDATES):
    """Sentences eligible for an *unaided* hit this turn: previously practiced
    (stats[key].count > 0) and NOT one of this turn's active targets. Most-due
    first (low count, then oldest last_ts), capped. Pure.

    'count > 0' is the gate that makes the signal meaningful (the user can only
    *re-*produce something already seen) AND bounds the EXPENSIVE work: only the
    capped shortlist is ever embedded (a fresh deck yields an empty list — no
    embedding at all). The eligibility filter + sort here is plain in-memory CPU
    over the practiced subset (microseconds, off the critical path), not the
    2000-scale per-turn EMBEDDING that deferred T-P2-1."""
    active = set(active_keys)
    pool = [s for s in sentences
            if s.key not in active
            and (st := stats.get(s.key)) is not None and st.count > 0]
    pool.sort(key=lambda s: (stats[s.key].count, stats[s.key].last_ts))
    return pool[:cap]


def detect_unaided(user_text, candidates, embed: EmbedFn,
                   threshold: float = UNAIDED_THRESHOLD, cache: dict | None = None):
    """The single best unaided hit among `candidates` (each has .key and .text),
    or None. Returns at most ONE hit per turn (the highest similarity above
    `threshold`) so a transcript can't spray credit across near-duplicate cards.

    `embed` is the injectable EmbedFn; `cache` is an optional {key: vec} dict
    reused across turns so each candidate is embedded at most once per session.
    Pure over `embed`/`cache` -> fully headless-testable. Caller guarantees
    `candidates` already excludes this turn's active targets (select_candidates)."""
    user_text = (user_text or "").strip()
    if not user_text or not candidates:
        return None
    cache = cache if cache is not None else {}
    missing = [c for c in candidates if c.key not in cache]
    if missing:
        for c, vec in zip(missing, embed([c.text for c in missing])):
            cache[c.key] = vec
    user_vec = embed([user_text])[0]
    best: UnaidedHit | None = None
    for c in candidates:
        sim = cosine(user_vec, cache[c.key])
        if sim >= threshold and (best is None or sim > best.similarity):
            best = UnaidedHit(c.key, c.text, sim)
    return best
