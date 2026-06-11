"""Neutral local-embedding helpers (nomic-embed-text via Ollama) + cosine.

Shared by both the English "practiced" scorer and the markdown-recall coverage
scorer, so neither owns the other's threshold (C7). Pure cosine + a thin Ollama
/api/embed client.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from collections.abc import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]

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
    except urllib.error.HTTPError as e:  # 404 = model not pulled, 500, ...
        body = e.read().decode("utf-8", "replace")[:200] if e.fp else ""
        raise RuntimeError(
            f"Ollama embed HTTP {e.code} (model {model!r}): {body or e.reason}"
        ) from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Ollama embed request failed: {e}") from e
    embs = data.get("embeddings")
    if not isinstance(embs, list) or len(embs) != len(texts):
        raise RuntimeError(
            f"unexpected Ollama embed response (model {model!r}): {str(data)[:200]}"
        )
    return embs
