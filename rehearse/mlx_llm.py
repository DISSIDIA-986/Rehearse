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
MLX_COACH_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"  # SAME id as extract -> _MODELS shares one load
MLX_PROBE_MAX_TTFT_S = 3.0  # warm TTFT contract for the coach probe

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


def resolve_coach_chat(backend: str = "auto", model: str | None = None):
    """Pick the LIVE COACH backend ONCE per session (never mid-turn): returns
    (chat_fn, model_id, backend_name). 'auto' uses MLX if mlx-lm is importable (saves
    ~0.7s/turn first-audio on this hardware), else Ollama. Each backend has its own
    model namespace (Ollama tag vs HF id), so the resolver owns per-backend defaults
    and `model` is interpreted in the chosen backend's namespace. Returns a chat_fn
    pre-bound to the resolved model id, drop-in for `chat`/`speak_turn`'s chat_fn."""
    import functools
    if backend in ("auto", "mlx"):
        if mlx_available():
            mid = model or MLX_COACH_MODEL
            return functools.partial(mlx_chat, model=mid), mid, "mlx"
        if backend == "mlx":
            print("  (mlx-lm unavailable — falling back to Ollama for the coach)")
    from rehearse.llm_client import DEFAULT_MODEL
    from rehearse.llm_client import chat as ollama_chat
    mid = model or DEFAULT_MODEL
    return functools.partial(ollama_chat, model=mid), mid, "ollama"


def mlx_warm_and_probe(model: str) -> float:
    """Eager preload + WARM-TTFT probe for the live coach (mirror of Ollama's
    warmup() + think_probe()). First call loads the model (cold/networked download if
    needed); the second is the warm-latency contract. Raises on empty reply, <think>
    leak, or warm TTFT above MLX_PROBE_MAX_TTFT_S. Returns the warm TTFT."""
    mlx_chat([{"role": "user", "content": "warm"}], model=model, num_predict=4)  # load
    r = mlx_chat([{"role": "user", "content": "Reply with the single word OK."}],
                 model=model, num_predict=8)
    if not r.text.strip():
        raise RuntimeError(f"MLX coach probe: empty reply (model={model})")
    if r.had_think_block:
        raise RuntimeError(f"MLX coach probe: model emitted <think> despite "
                           f"enable_thinking=False (model={model})")
    if r.ttft_s is None or r.ttft_s > MLX_PROBE_MAX_TTFT_S:
        raise RuntimeError(f"MLX coach probe: warm TTFT {r.ttft_s}s exceeds "
                           f"{MLX_PROBE_MAX_TTFT_S}s contract (model={model})")
    return r.ttft_s
