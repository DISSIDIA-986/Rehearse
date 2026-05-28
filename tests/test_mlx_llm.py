"""MLX extraction backend shim. The real-generate test is gated (loads a ~2.5GB
MLX model); the shape/availability checks run everywhere (mlx_lm imports are lazy)."""

import os

import pytest

from rehearse import mlx_llm


def test_mlx_available_returns_bool():
    assert isinstance(mlx_llm.mlx_available(), bool)


def test_module_imports_without_mlx_lm():
    # importing the shim must NOT import mlx_lm (keeps `uv run pytest` green w/o the extra)
    import sys
    assert "mlx_lm" not in sys.modules or mlx_llm.mlx_available()  # only present if actually installed


@pytest.mark.skipif(not os.environ.get("REHEARSE_LLM_TESTS"),
                    reason="set REHEARSE_LLM_TESTS=1 for a real MLX generate (loads a model)")
def test_mlx_chat_returns_full_chatresult():
    if not mlx_llm.mlx_available():
        pytest.skip("mlx-lm not installed")
    r = mlx_llm.mlx_chat([{"role": "user", "content": "Reply with exactly: OK"}], num_predict=8)
    assert isinstance(r.text, str) and r.text
    assert r.ttft_s is not None and r.total_s >= 0 and r.had_think_block is False


# --- coach backend selection (mirror of resolve_extract_chat tests) -------

def test_resolve_coach_auto_prefers_mlx_when_available(monkeypatch):
    monkeypatch.setattr(mlx_llm, "mlx_available", lambda: True)
    fn, model, name = mlx_llm.resolve_coach_chat("auto")
    assert name == "mlx" and model == mlx_llm.MLX_COACH_MODEL
    # bound chat_fn calls mlx_chat with model pre-bound
    assert fn.func is mlx_llm.mlx_chat and fn.keywords.get("model") == mlx_llm.MLX_COACH_MODEL


def test_resolve_coach_mlx_unavailable_falls_back_to_ollama(monkeypatch, capsys):
    from rehearse.llm_client import chat as ollama_chat
    monkeypatch.setattr(mlx_llm, "mlx_available", lambda: False)
    fn, _model, name = mlx_llm.resolve_coach_chat("mlx")  # asked mlx, unavailable
    assert name == "ollama" and fn.func is ollama_chat
    assert "falling back to Ollama" in capsys.readouterr().out


def test_resolve_coach_explicit_ollama_even_if_mlx_present(monkeypatch):
    from rehearse.llm_client import chat as ollama_chat
    monkeypatch.setattr(mlx_llm, "mlx_available", lambda: True)
    fn, _model, name = mlx_llm.resolve_coach_chat("ollama")
    assert name == "ollama" and fn.func is ollama_chat


@pytest.mark.skipif(not os.environ.get("REHEARSE_LLM_TESTS"),
                    reason="set REHEARSE_LLM_TESTS=1 for a real MLX warm+probe (loads a model)")
def test_mlx_warm_and_probe_enforces_contract():
    if not mlx_llm.mlx_available():
        pytest.skip("mlx-lm not installed")
    ttft = mlx_llm.mlx_warm_and_probe(mlx_llm.MLX_COACH_MODEL)
    assert 0 < ttft < mlx_llm.MLX_PROBE_MAX_TTFT_S
