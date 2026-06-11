# Rehearse

> 本地、离线、免费、开源的语音练习教练 —— 练英语口语对话，也能凭记忆复述任意 Markdown 笔记。在 Apple Silicon Mac 上跑。
> A fully-local, offline, free, open-source voice coach for Apple Silicon Macs: practice English conversation, and recall any Markdown doc from memory — out loud.

**Status:** 🟢 v1 + markdown-recall mode shipped (159 tests pass, `--smoke` green). Approved architecture below.

## 这是什么 / What it is

练 **自然连贯的日常英语对话**——不是补词汇，而是把已经背过的高频句子（如 Anki 句库）织进真实对话里
做"对话式间隔复习"（conversational spaced repetition）。100% 本地离线，无云 API。

A low-latency voice loop (you speak → assistant speaks back) that weaves your existing high-frequency
sentences (e.g. an Anki sentence bank) into natural conversation, so you *practice them in context*
instead of passively reviewing flashcards. Runs entirely offline.

它还能做**凭记忆复述**：指定任意 Markdown 文件（简历、面试准备、术语表、演讲稿……），它把内容拆成要点来"考"你，
你出声复述，系统诚实地评估覆盖度。详见下方 [Markdown-recall mode](#markdown-recall-mode--凭记忆复述任意-markdown)。
It also does **recall practice**: point it at any Markdown file (resume, interview prep, a glossary, a
speech) and it interviews you to recall the key points from memory — see [Markdown-recall mode](#markdown-recall-mode--凭记忆复述任意-markdown) below.

## 目标硬件 / Target hardware

Mac Studio · Apple M1 Max · 32GB unified memory (Apple Silicon generally).

## 架构 / Architecture (approved)

Fork & adapt [`eauchs/speech-to-speech-pipeline`](https://github.com/eauchs/speech-to-speech-pipeline)
(Apple-Silicon / MLX, already has barge-in).

| Stage | Choice | Device | Why |
|---|---|---|---|
| ASR | faster-whisper `small.en` | **CPU** (int8) | English-only, keeps GPU free |
| LLM | **MLX** `Qwen3.5-4B-MLX-4bit`（非思考,Ollama 回退；embeddings 仍走 Ollama） | **GPU** | 实测整句回复约一半时间,每轮首音省 ~0.7s |
| TTS | Kokoro-82M via mlx-audio | **GPU**（与 LLM 串行） | RTF ≈ 0.03, natural enough, Apache-2.0 |
| VAD | Silero VAD | CPU | sub-second endpointing |
| Glue | asyncio + sentence-chunked streaming TTS + half-duplex | — | lowest perceived latency |

**关键设计取舍 / Key design decisions**（经 Codex 对抗审查加固，详见 `docs/DESIGN.md` 的 v2 Hardening）
- **不全用 MLX** — ASR 留在 CPU，避免 ASR+LLM+TTS 三者抢同一块 Metal GPU（尾延迟会涨 20-40%）。
- **GPU 上 LLM 与 TTS 串行**，不并发——并发恰恰是 TTFT 抖动最严重的时刻。
- **半双工默认**（TTS 播放时静音麦克风）根除外放回声；戴耳机可选全双工插话打断。
- **不显式纠错** — 自然对话伙伴，靠地道措辞和复述潜移默化（纯语音闭环无法评判发音，这是固有边界）。
- **保温 + 非思考（结构性）** — `OLLAMA_KEEP_ALIVE=-1`；经 Ollama 原生 `think:false` + 探针验证无 `<think>` 块；`deepseek-r1` 类不适合语音。
- **延迟靠实测** — 第一天埋 4 段时间戳出 p50/p95，`<1.5s` 是冲刺目标不是验收标准。

完整设计与盲区清单见 [`docs/DESIGN.md`](docs/DESIGN.md)。
Full design + risk/blind-spot list in [`docs/DESIGN.md`](docs/DESIGN.md).

## Run / 运行

Prereqs: `uv`, Ollama with `qwen3.5:4b` + `nomic-embed-text` pulled, your AnkiApp
XML decks in `data/`. First run downloads the Whisper / Kokoro / spaCy models
(needs network once; fully offline after).

```bash
uv sync --extra audio --extra vad        # install ASR/TTS/VAD (one time)
uv run rehearse --smoke                # health check, no mic (recommended first)
uv run rehearse --menu                 # interactive menu — pick a mode, no flags to memorize
uv run rehearse                        # the live voice loop — just start talking
uv run rehearse --full-duplex          # voice barge-in (use with AirPods/headphones)
```

**Shortcut:** add a `rehearse` alias so you can launch the menu from anywhere without
typing the long command (`rehearse` runs in a subshell, so your current directory is
unchanged). Run this **from the repo root** — it bakes in the repo's absolute,
quoted path:
```bash
cd /path/to/Rehearse   # the cloned repo (so $(pwd) below is correct)
echo "alias rehearse='(cd \"$(pwd)\" && uv run rehearse --menu)'" >> ~/.zshrc && source ~/.zshrc
rehearse   # opens the menu: English practice / Markdown recall / smoke / quit
```

**If it feels too rushed (cuts you off while thinking):**
```bash
uv run rehearse --end-silence-ms 1500  # wait longer for you to finish (default 1000)
uv run rehearse --manual-turns         # press Enter to start/stop each turn — zero time pressure
uv run rehearse --manual-turns --brief # + short (~15-word) replies: snappier, you talk more
```
**If recognition is inaccurate (accent / noise):** ASR now uses `vad_filter` +
`beam_size=5` by default. If still off, use a bigger model (slower, more accurate):
```bash
uv run rehearse --asr-model medium.en        # ~2.3s ASR vs ~0.9s for small.en
uv run rehearse --asr-model distil-large-v3  # most accurate
uv run rehearse --debug                # save each turn's audio+transcript to debug/ to inspect
```

Defaults: half-duplex (mute mic while the assistant speaks — works on the Mac
speaker without echo), continuous until you say "stop" or hit Ctrl-C, press Enter
to cut off a reply. On exit it prints latency p50/p95 and your practiced count.

**Spaced repetition persists across sessions.** Each sentence you practice is
saved to `~/.local/share/rehearse/practice.db` (SQLite), so the next session
keeps surfacing your least-practiced sentences first — real conversational
spaced repetition, not a per-run reset.
```bash
uv run rehearse --no-persist           # in-memory only (don't save/load history)
uv run rehearse --practice-db /path/db # use a specific stats file
```
A corrupt/unreadable DB is quarantined and recreated automatically; a save error
degrades to a warning and never interrupts a turn.

See your progress between sessions (offline, read-only):
```bash
uv run rehearse-stats                  # most/least-practiced sentences + totals
uv run rehearse-stats --practice-db /path/db
```

The LLM backend auto-picks **MLX** (Apple-GPU, faster) if `mlx-lm` is present, else
Ollama. Override per role with `--coach-backend {auto,mlx,ollama}` (live coach) or
`--extract-backend {auto,mlx,ollama}` (one-time markdown extraction). Ollama is still
required either way for embeddings and as the fallback backend.

### Markdown-recall mode / 凭记忆复述任意 Markdown

Point it at ANY markdown file (resume, interview prep, a terminology glossary, a
conference summary, a prepared speech) and it interviews you to recall the key
points **from memory**, out loud, tracking coverage honestly.

```bash
uv run rehearse --content markdown --path /abs/path/to/notes.md
uv run rehearse --content markdown --path notes.md --extract-backend ollama  # force Ollama
```

How it works: the local LLM extracts the doc into a recall agenda. This is
**one-time and cached** (written next to the file as `notes.md.recall.json` so you
can see exactly what will be drilled) — a long doc is one LLM call per chunk, so the
first run takes a few minutes with a `chunk N/total` progress line. Extraction runs
on **MLX** by default (Apple-GPU, dogfooded ~2.6x faster than Ollama: a 32KB doc
~3 min vs ~7.6), falling back to Ollama if mlx-lm isn't available; `--extract-backend
ollama` forces Ollama. (Ollama is still required for the live coach and the
embedding-based scoring — this only accelerates the one-time extraction.) A warm
coach then asks each item and you answer from
memory (manual turns — Enter to speak, Enter to send, so there's no time
pressure). Coverage is scored **honestly**: a point counts as recalled only when
your answer is both semantically on-topic AND contains its hard facts (the
numbers/acronyms, or enough of its key words) — vague talk doesn't get credit.
The coach never sees the expected answers, so it can't leak them; it offers a
hint only after you stall, then moves on so you're never stuck. It prints a
coverage summary at the end.

**Status:** v1 + markdown-recall mode implemented, 159 tests pass, `--smoke` green.
The live mic loop is the one path validated manually (the dev environment had no
mic; the full ASR→LLM→TTS chain is covered by automated TTS→ASR round-trip +
full-turn tests).

## 路线图 / Roadmap

- **Phase 1 (MVP):** ✅ done — full local loop (faster-whisper + qwen3.5:4b + Kokoro
  + Silero), continuous half-duplex conversation, Anki sentence injection + nomic
  "practiced" scoring, latency instrumentation, native smoke test.
- **Phase 2:** ✅ markdown-recall mode — recall any markdown doc from memory via
  local-LLM extraction + honest (cosine + fact-anchor) coverage scoring. Plus an
  interactive launcher menu (`--menu`) and a `rehearse` alias. The menu's *live status
  display* (running-session stats) is still deferred.
- **Later:** 用 `nomic-embed-text` + `bce-reranker` 做语义检索，按话题动态调相关句子 + 跨会话间隔复习排程（见 `TODOS.md`）。

## License

TBD (intended permissive). Components used are MIT / Apache-2.0.
