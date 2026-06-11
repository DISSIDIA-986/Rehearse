"""Tests for the neutral embedding helpers (cosine + the Ollama /api/embed client).

cosine is pure (unit). The HTTP-error path is exercised by faking urlopen so we
assert the friendly, specific message without a live Ollama.
"""

import io
import json
import urllib.error

import pytest

from rehearse.embeddings import cosine, ollama_embed


# --- cosine (pure) --------------------------------------------------------

def test_cosine_identical_is_one():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_empty_or_mismatched_len_is_zero():
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0, 2.0], [1.0]) == 0.0


def test_cosine_zero_vector_is_zero():
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_guards_nan_inf():
    assert cosine([float("nan"), 1.0], [1.0, 1.0]) == 0.0


# --- ollama_embed error specificity ---------------------------------------

def _http_error(code, body=b'{"error":"model not found"}'):
    return urllib.error.HTTPError(
        url="http://x/api/embed", code=code, msg="Not Found",
        hdrs=None, fp=io.BytesIO(body),
    )


def test_ollama_embed_http_error_is_specific(monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise _http_error(404)
    monkeypatch.setattr("rehearse.embeddings.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="HTTP 404"):
        ollama_embed(["hello"])


def test_ollama_embed_connection_refused_is_friendly(monkeypatch):
    def fake_urlopen(req, timeout=0):
        raise urllib.error.URLError("Connection refused")
    monkeypatch.setattr("rehearse.embeddings.urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="request failed"):
        ollama_embed(["hello"])


def test_ollama_embed_length_mismatch_raises(monkeypatch):
    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps({"embeddings": [[0.1, 0.2]]}).encode()
    monkeypatch.setattr("rehearse.embeddings.urllib.request.urlopen",
                        lambda req, timeout=0: _Resp())
    with pytest.raises(RuntimeError, match="unexpected Ollama embed response"):
        ollama_embed(["one", "two"])  # asked for 2, got 1
