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

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

EmbedFn = Callable[[list[str]], list[list[float]]]

# Calibrated against real nomic-embed-text on this project's sentences
# (2026-05-27): paraphrase/repeat pairs scored 0.55-1.0 (loosest real
# paraphrase 0.548), unrelated pairs maxed at 0.454. 0.50 sits in the clean
# gap — catches paraphrases, rejects unrelated. Re-run the calibration probe
# if the embed model changes.
DEFAULT_THRESHOLD = 0.50
_OLLAMA_HOST = "http://localhost:11434"
_EMBED_MODEL = "nomic-embed-text"


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    result = dot / (na * nb)
    return result if math.isfinite(result) else 0.0  # guard nan/inf embeddings


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


def ollama_embed(
    texts: list[str], model: str = _EMBED_MODEL, host: str = _OLLAMA_HOST
) -> list[list[float]]:
    """Embed texts via the local Ollama /api/embed endpoint (nomic-embed-text)."""
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{host}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Ollama embed request failed: {e}") from e
    embs = data.get("embeddings")
    if not isinstance(embs, list) or len(embs) != len(texts):
        raise RuntimeError(
            f"unexpected Ollama embed response (model {model!r}): {str(data)[:200]}"
        )
    return embs
