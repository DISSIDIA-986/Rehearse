# ASR / TTS / LLM model evaluation (2026-06)

Decision record for "are there better MLX-optimized models we should swap to?"
Driven by web research (HuggingFace, Open ASR Leaderboard, TTS Arena, MLX
community) cross-checked against an adversarial Codex high-effort review.

## TL;DR — keep all three shipped models

| Component | Shipped | Verdict |
|---|---|---|
| TTS | Kokoro-82M via `mlx-audio` (GPU) | **KEEP** — still the low-latency streaming sweet spot |
| LLM | `mlx-community/Qwen3.5-4B-MLX-4bit` (+ Ollama fallback) | **KEEP** — just migrated, works; no clearly-better option |
| ASR | faster-whisper `small.en` (CPU, int8) | **EVALUATE, don't blind-swap** — see Parakeet below |

Nothing is "clearly better AND low-risk" enough to swap on. The stack was well
chosen. Forcing a replacement on unverifiable public benchmarks would risk exactly
the regression we want to avoid.

## Why not swap TTS

Every "better" candidate (Qwen3-TTS, Voxtral-4B-TTS, Orpheus-3B, Sesame CSM-1B) is
much larger than Kokoro's 82M, contends for unified-memory/GPU with the LLM, and
lacks mature low-latency *streaming* (time-to-first-audio) in `mlx-audio`. For a
half-duplex turn-taking loop, Kokoro's tiny size + chunked streaming is the win.

## Why not swap LLM

The one switch pitched (Phi-4-mini for the coach) rested on a suspiciously precise
"18ms TTFT" number that we could not verify. Phi-4-mini is real
(`mlx-community/Phi-4-mini-instruct-4bit`) but weaker on general reasoning
(MMLU-Pro 52.8 vs Qwen 79.1, per Microsoft's own card) and positioned for
constrained/latency cases, not "better coach." We just migrated the coach and the
extractor to Qwen3.5-4B-MLX and it works — the bar to churn it is "clearly better,"
not "different."

## ASR — the one real candidate: NVIDIA Parakeet TDT 0.6B

Parakeet is legitimate and strong (HF Open ASR Leaderboard: v2 ~6.05 avg WER, v3
~6.34, very high RTFx). `parakeet-mlx` (senstella) is a real, pip-installable
Python package (PyPI 0.5.2, MLX/Metal). But it is **not** a safe auto-swap:

1. **MLX has no Neural Engine path.** MLX targets CPU + GPU only. The "no-contention
   ANE" route the research surfaced is FluidAudio, a **Swift SDK** — unusable from
   this Python `uv` project. `coremltools` could reach the ANE from Python, but
   that's a separate porting project, not a drop-in.
2. **The Python path runs on Metal.** `parakeet-mlx` uses the GPU, so adopting it
   reintroduces exactly the ASR/LLM/TTS GPU contention the design deliberately
   avoids by keeping whisper on the CPU. (Codex refined the rule: it's "don't
   *overlap* on Metal," not "ASR must be CPU forever" — if ASR is fully serialized
   before the LLM turn, a sub-50ms Parakeet burst may not hurt felt latency.)
3. **ASR is low-leverage on felt latency.** For turn-taking, the budget ranks:
   VAD end-of-utterance → **LLM TTFT** → **TTS time-to-first-audio** → ASR
   finalization. Swapping ASR mostly saves milliseconds the user never feels —
   *unless* whisper is producing bad transcripts that confuse the coach.
4. **It can't be decided in CI.** No mic here. The decision needs *your* voice,
   *your* DJI mic, *your* room.

### Promotion bar (Codex)

Parakeet replaces whisper **only if** it BOTH:
- materially improves transcript quality (WER on *your own* utterances), AND
- does not regress p95 felt first-audio latency under GPU contention.

Saving a few ASR milliseconds is not a reason to switch.

## How to actually decide — `rehearse-bench-asr`

A decision tool was built (it does **not** touch the shipped pipeline):

```bash
uv sync --extra audio --extra parakeet           # adds parakeet-mlx (eval only)
rehearse-bench-asr record --refs sentences.txt --out bench_audio/   # read aloud, real mic
rehearse-bench-asr run --wav-dir bench_audio/                       # whisper vs parakeet WER+latency
rehearse-bench-asr run --wav-dir bench_audio/ --contention          # coach TTFT alone vs + ASR on GPU
```

The report prints per-utterance WER, p50/p95 wall time per backend, the head-to-head
WER delta, and a KEEP/CHECK verdict. The contention probe measures whether running
Parakeet on the GPU regresses coach TTFT p95 (the forced-overlap test).

`--only whisper` runs the baseline without installing the parakeet extra.

## LLM backend (mlx-lm vs Ollama) — frozen 2026-06-10

Adversarial follow-up: is keeping mlx-lm over Ollama worth it, and should we push
further (LLM→TTS streaming, prompt/KV-cache reuse)? Measured on this M1 Max
(`experiments/mlx_loop_probe.py`, `experiments/mlx_extract_probe.py`):

| | TTFT | first-audio | total turn | decode |
|---|---|---|---|---|
| Ollama qwen3.5:4b | 0.39s | 0.82s | 1.52s | 127 ch/s |
| mlx-lm Qwen3.5-4B-4bit | 0.41s | 0.55s¹ | 0.82s | 357 ch/s (~2.8x) |

¹ The 0.55s assumes sentence-streamed TTS; **production does not stream** —
`speak_turn` generates the full reply, then synthesizes TTS (`pipeline.py`). So
real felt first-audio ≈ full turn (~0.82s) + first TTS chunk ≈ ~1.0s.

**The gain is decode throughput, NOT TTFT** (TTFT is flat). Extraction is ~2.6x but
cached (one-time), so per-session felt savings ≈ 0.

**Decision (frozen):** keep mlx-lm primary + **Ollama fallback (do not remove)**,
and do NOT pursue streaming or KV-cache. 3-way independent review (Codex
high-effort + 2 sub-agents) was unanimous STOP:
- felt latency ~1.0s is already 33% under the 1.5s goal — diminishing returns;
- streaming/KV-cache would touch the only manually-validated loop (a cross-thread
  race was already fixed once) and couple to the fast-churning mlx-lm API, for a
  ~0.1–0.3s imperceptible gain;
- **Ollama cannot be removed anyway** — it is the `nomic-embed-text` embeddings
  backend (`rehearse/embeddings.py`) used for coverage/practiced scoring.

Revisit only if daily use *feels* sluggish — the latency tracer below now surfaces
p50/p95 felt latency per session, so that would show up as data, not a guess.

## Side deliverable — per-turn latency instrumentation

Codex's "highest-leverage" change. All three live loops now record a per-turn
budget (`rehearse/latency.py`): ASR → LLM TTFT → TTS time-to-first-audio → felt
(end-of-utterance → first audio out), with a p50/p95 session summary. With
`--debug` it prints a one-line trace per turn. This is what tells you, on real
hardware, whether ASR is even on the critical path before you invest in swapping it.
(`speak_turn` now also reports `tts_ttfa_s` — time to the first audio chunk, the
only TTS latency a turn-taking user actually feels — separate from total synth time.)
