"""Probe: would the VOICE LOOP feel faster on MLX vs Ollama? (informs a future call)

The loop's perceived latency is time-to-first-AUDIO, which is gated by time-to-first-
SENTENCE (TTFS) — the coach must finish one sentence before sentence-chunked TTS can
start — not raw throughput. Replies are short (1-3 sentences). So we measure, per
backend, over several coach-style turns: TTFT (first token), est. TTFS (ttft + time to
finish the first sentence at the measured rate), total reply time, and throughput.

Both run non-thinking (Ollama think:false; MLX enable_thinking=False). Sequential.
Run:  uv run python experiments/mlx_loop_probe.py
"""

from __future__ import annotations

import re
import statistics as st

from rehearse.llm_client import chat
from rehearse.mlx_llm import MLX_EXTRACT_MODEL, mlx_chat

SYS = ("You are a friendly native English speaker having a casual voice chat. Keep replies "
       "to 1-2 short, warm sentences and end with a light question. Plain text only.")
TURNS = [
    "I just got back from a hike and my legs are wrecked but it was gorgeous.",
    "Work was rough today, too many meetings and not enough actual work.",
    "I'm trying to cook more at home lately instead of ordering takeout.",
    "I watched a really weird documentary last night about deep-sea fish.",
]
_SENT = re.compile(r"[.!?]")


def measure(name: str, gen) -> None:
    gen([{"role": "system", "content": SYS}, {"role": "user", "content": "warm up"}])  # warm
    ttfts, ttfs, totals, rates = [], [], [], []
    for u in TURNS:
        r = gen([{"role": "system", "content": SYS}, {"role": "user", "content": u}])
        body = max(r.total_s - (r.ttft_s or 0), 1e-3)
        rate = len(r.text) / body  # chars/sec after first token
        m = _SENT.search(r.text)
        first_sent = r.text[:m.end()] if m else r.text
        est_ttfs = (r.ttft_s or 0) + len(first_sent) / rate
        ttfts.append(r.ttft_s or 0); ttfs.append(est_ttfs); totals.append(r.total_s); rates.append(rate)
    print(f"== {name}: TTFT={st.median(ttfts):.2f}s  est.TTFS={st.median(ttfs):.2f}s  "
          f"total={st.median(totals):.2f}s  rate={st.median(rates):.0f} ch/s ==")
    print(f"     sample reply: {r.text[:80]!r}")


def main() -> None:
    import subprocess
    print("Coach-style short replies; both non-thinking. TTFS = when TTS could start.\n")
    measure("ollama qwen3.5:4b",
            lambda msgs: chat(msgs, model="qwen3.5:4b", num_predict=60, temperature=0.0))
    subprocess.run(["ollama", "stop", "qwen3.5:4b"], capture_output=True)
    measure(f"mlx {MLX_EXTRACT_MODEL}",
            lambda msgs: mlx_chat(msgs, model=MLX_EXTRACT_MODEL, num_predict=60, temperature=0.0))


if __name__ == "__main__":
    main()
