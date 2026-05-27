"""Measure whether the user "practiced" a target sentence this turn (D3).

Codex review: "used in conversation" != "practiced", and if you can't measure it
it's theater. So after each user turn we embed the user's transcript and each
active target with nomic-embed (local, already installed) and count a hit when
cosine similarity clears a threshold (the user repeated or paraphrased it).

The scoring logic (cosine + threshold) is pure and injectable for tests; the
real embedder hits Ollama's local /api/embed. Cross-session "unaided production"
tracking is deliberately Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

# cosine/ollama_embed/EmbedFn live in the neutral embeddings module (C7); re-exported
# here so existing importers (pipeline, tests) keep working unchanged.
from localvocal.embeddings import EmbedFn, cosine, ollama_embed  # noqa: F401

# Calibrated against real nomic-embed-text on this project's sentences
# (2026-05-27): paraphrase/repeat pairs scored 0.55-1.0 (loosest real
# paraphrase 0.548), unrelated pairs maxed at 0.454. 0.50 sits in the clean
# gap — catches paraphrases, rejects unrelated. Re-run the calibration probe
# if the embed model changes.
DEFAULT_THRESHOLD = 0.50


@dataclass
class PracticeHit:
    target: str
    similarity: float


def score_practiced(
    user_text: str,
    targets: list[str],
    embed: EmbedFn,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[PracticeHit]:
    """Return targets the user repeated/paraphrased this turn, best first."""
    user_text = (user_text or "").strip()
    targets = [t for t in targets if t and t.strip()]
    if not user_text or not targets:
        return []
    vectors = embed([user_text, *targets])
    if len(vectors) != len(targets) + 1:
        raise ValueError(
            f"embedder returned {len(vectors)} vectors, expected {len(targets) + 1}"
        )
    user_vec, target_vecs = vectors[0], vectors[1:]
    hits = [
        PracticeHit(target=t, similarity=cosine(user_vec, tv))
        for t, tv in zip(targets, target_vecs)
    ]
    hits = [h for h in hits if h.similarity >= threshold]
    hits.sort(key=lambda h: h.similarity, reverse=True)
    return hits
