"""Probe v2: FAIR mlx-lm vs Ollama for markdown extraction.

v1 was unfair to MLX — MLX ran with thinking ON and no early stop, so it generated
~2.5x more text (hit max_tokens) and looked only ~1.1x faster on wall-clock, even
though its raw throughput was ~2.8x. This version disables thinking on the MLX side
(Qwen3 enable_thinking=False) so BOTH backends are non-thinking and stop at the JSON.

Same doc/chunks/prompt/parser; sequential; both warmed. Reports wall-clock AND
chars/sec (throughput) so the two effects are separated.

Run:  uv run python experiments/mlx_extract_probe.py /abs/doc.md [--chunks 3] [--all]
"""

from __future__ import annotations

import argparse
import subprocess
import time

from rehearse.markdown_extractor import (
    _EXTRACT_PROMPT,
    _ITEM_SCHEMA,
    _parse_json_obj,
    chunk_markdown,
)


def _prompt_for(chunk: str) -> str:
    return _EXTRACT_PROMPT.replace("{chunk}", chunk)


def _items(text: str) -> list[str]:
    try:
        data = _parse_json_obj(text)
        return [str(it.get("prompt", "")).strip()
                for it in (data.get("items") or []) if str(it.get("prompt", "")).strip()]
    except Exception:
        return []


def run(name: str, gen, chunks: list[str]) -> None:
    t0 = time.monotonic()
    prompts: list[str] = []
    out_chars = 0
    for i, ch in enumerate(chunks, 1):
        c0 = time.monotonic()
        text = gen(_prompt_for(ch))
        out_chars += len(text)
        got = _items(text)
        prompts += got
        print(f"  [{name}] chunk {i}/{len(chunks)}  {time.monotonic()-c0:5.1f}s  "
              f"{len(got)} items {'(PARSE FAIL)' if not got else ''}", flush=True)
    dt = time.monotonic() - t0
    print(f"== {name}: {dt:.1f}s / {len(chunks)} chunks ({dt/len(chunks):.1f}s/chunk), "
          f"{len(prompts)} items, {out_chars} chars, {out_chars/dt:.0f} chars/s ==")
    for p in prompts[:3]:
        print(f"     • {p[:80]!r}")


def ollama_gen(model: str):
    from rehearse.llm_client import chat  # already sends think:false + format
    return lambda prompt: chat([{"role": "user", "content": prompt}],
                               model=model, num_predict=4096, temperature=0.0,
                               fmt=_ITEM_SCHEMA).text


def mlx_gen(model_id: str):
    from mlx_lm import generate, load
    model, tok = load(model_id)

    def _template(prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:  # Qwen3 family: turn OFF thinking, same as Ollama's think:false
            return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           tokenize=False, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    def gen(prompt: str) -> str:
        return generate(model, tok, _template(prompt), max_tokens=4096, verbose=False)

    gen("warm up")  # exclude lazy init from timing
    return gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc")
    ap.add_argument("--mlx-model", default="mlx-community/Qwen3.5-4B-MLX-4bit")
    ap.add_argument("--ollama-model", default="qwen3.5:4b")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    text = open(args.doc, encoding="utf-8").read()
    all_chunks = chunk_markdown(text)
    chunks = all_chunks if args.all else all_chunks[:args.chunks]
    print(f"doc: {len(text)} chars, {len(all_chunks)} chunks total, timing {len(chunks)}.\n"
          f"Both non-thinking; MLX(4bit HF) vs Ollama(GGUF) — quant differs slightly.\n")

    print(f"--- Ollama: {args.ollama_model} (think:false + format) ---")
    run("ollama", ollama_gen(args.ollama_model), chunks)
    subprocess.run(["ollama", "stop", args.ollama_model], capture_output=True)

    print(f"\n--- MLX: {args.mlx_model} (enable_thinking=False) ---")
    run("mlx", mlx_gen(args.mlx_model), chunks)


if __name__ == "__main__":
    main()
