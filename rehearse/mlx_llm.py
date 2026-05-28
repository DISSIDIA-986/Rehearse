"""MLX (Apple-GPU) LLM backend for one-time markdown extraction.

Drop-in for `llm_client.chat`: same call signature (incl. `fmt`, which MLX has no
native schema mode for, so it's accepted and ignored — the tolerant JSON parser in
markdown_extractor handles output) and the same `ChatResult` return shape. Used only
for the throughput-bound extraction path (dogfooded ~2.6x faster than Ollama on this
Mac); the latency-bound voice loop and embeddings stay on Ollama.

`mlx_lm` is imported lazily INSIDE the calls so importing this module (and running the
test suite without the `audio` extra) never requires mlx-lm. The model loads on first
use and is cached per process — so a cache-hit `load_markdown` (no extraction) never
loads the ~2.5GB model.
"""

from __future__ import annotations

import importlib.util
import time

from rehearse.llm_client import DEFAULT_NUM_PREDICT, ChatResult, strip_think

MLX_EXTRACT_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"

_MODELS: dict = {}  # model_id -> (model, tokenizer), loaded once per process


def mlx_available() -> bool:
    """True if mlx-lm is importable (Apple Silicon + the extra installed)."""
    return importlib.util.find_spec("mlx_lm") is not None


def _load(model_id: str):
    if model_id not in _MODELS:
        from mlx_lm import load
        _MODELS[model_id] = load(model_id)
    return _MODELS[model_id]


def _render(tok, messages: list[dict[str, str]]) -> str:
    # Qwen3 family: enable_thinking=False is the structural equivalent of Ollama's
    # think:false (without it the model reasons and runs to the token cap).
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def mlx_chat(messages: list[dict[str, str]], *, model: str = MLX_EXTRACT_MODEL,
             num_predict: int = DEFAULT_NUM_PREDICT, temperature: float = 0.0,
             fmt=None, **_ignored) -> ChatResult:
    """Generate one reply on MLX. Signature matches llm_client.chat (fmt ignored)."""
    from mlx_lm import stream_generate
    mdl, tok = _load(model)
    prompt = _render(tok, messages)
    kwargs: dict = {"max_tokens": num_predict}
    if temperature and temperature > 0:
        from mlx_lm.sample_utils import make_sampler
        kwargs["sampler"] = make_sampler(temp=temperature)

    t0 = time.monotonic()
    ttft: float | None = None
    parts: list[str] = []
    for resp in stream_generate(mdl, tok, prompt, **kwargs):
        if ttft is None:
            ttft = time.monotonic() - t0
        parts.append(resp.text)
    total = time.monotonic() - t0
    text, had_think = strip_think("".join(parts))
    return ChatResult(text=text, ttft_s=ttft, total_s=total, had_think_block=had_think)
