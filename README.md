# LocalVocal

> 本地、离线、免费、开源的英语口语对话练习助手 —— 在 Apple Silicon Mac 上跑。
> A fully-local, offline, free, open-source English speaking-practice voice assistant for Apple Silicon Macs.

**Status:** 🟢 v1 + markdown-recall mode shipped (145 tests pass, `--smoke` green). Approved architecture below.

## 这是什么 / What it is

练 **自然连贯的日常英语对话**——不是补词汇，而是把已经背过的高频句子（如 Anki 句库）织进真实对话里
做"对话式间隔复习"（conversational spaced repetition）。100% 本地离线，无云 API。

A low-latency voice loop (you speak → assistant speaks back) that weaves your existing high-frequency
sentences (e.g. an Anki sentence bank) into natural conversation, so you *practice them in context*
instead of passively reviewing flashcards. Runs entirely offline.

## 目标硬件 / Target hardware

Mac Studio · Apple M1 Max · 32GB unified memory (Apple Silicon generally).

## 架构 / Architecture (approved)

Fork & adapt [`eauchs/speech-to-speech-pipeline`](https://github.com/eauchs/speech-to-speech-pipeline)
(Apple-Silicon / MLX, already has barge-in).

| Stage | Choice | Device | Why |
|---|---|---|---|
| ASR | faster-whisper `small.en` | **CPU** (int8) | English-only, keeps GPU free |
| LLM | Ollama `qwen3.5:4b` (warm, non-thinking) | **GPU** | 4B 比 9B 首字更快；latency-first（实测为准） |
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
uv run localvocal --smoke                # health check, no mic (recommended first)
uv run localvocal --menu                 # interactive menu — pick a mode, no flags to memorize
uv run localvocal                        # the live voice loop — just start talking
uv run localvocal --full-duplex          # voice barge-in (use with AirPods/headphones)
```

**Shortcut:** add a `lv` alias so you can launch the menu from anywhere without
typing the long command (`lv` runs in a subshell, so your current directory is
unchanged). Run this **from the repo root** — it bakes in the repo's absolute,
quoted path:
```bash
cd /path/to/LocalVocal   # the cloned repo (so $(pwd) below is correct)
echo "alias lv='(cd \"$(pwd)\" && uv run localvocal --menu)'" >> ~/.zshrc && source ~/.zshrc
lv   # opens the menu: English practice / Markdown recall / smoke / quit
```

**If it feels too rushed (cuts you off while thinking):**
```bash
uv run localvocal --end-silence-ms 1500  # wait longer for you to finish (default 1000)
uv run localvocal --manual-turns         # press Enter to start/stop each turn — zero time pressure
uv run localvocal --manual-turns --brief # + short (~15-word) replies: snappier, you talk more
```
**If recognition is inaccurate (accent / noise):** ASR now uses `vad_filter` +
`beam_size=5` by default. If still off, use a bigger model (slower, more accurate):
```bash
uv run localvocal --asr-model medium.en        # ~2.3s ASR vs ~0.9s for small.en
uv run localvocal --asr-model distil-large-v3  # most accurate
uv run localvocal --debug                # save each turn's audio+transcript to debug/ to inspect
```

Defaults: half-duplex (mute mic while the assistant speaks — works on the Mac
speaker without echo), continuous until you say "stop" or hit Ctrl-C, press Enter
to cut off a reply. On exit it prints latency p50/p95 and your practiced count.

### Markdown-recall mode / 凭记忆复述任意 Markdown

Point it at ANY markdown file (resume, interview prep, a terminology glossary, a
conference summary, a prepared speech) and it interviews you to recall the key
points **from memory**, out loud, tracking coverage honestly.

```bash
uv run localvocal --content markdown --path /abs/path/to/notes.md
uv run localvocal --content markdown --path notes.md --extract-model qwen3.5:9b
```

How it works: the local LLM extracts the doc into a recall agenda (one-time,
cached, and written next to the file as `notes.md.recall.json` so you can see
exactly what will be drilled). A warm coach asks each item and you answer from
memory (manual turns — Enter to speak, Enter to send, so there's no time
pressure). Coverage is scored **honestly**: a point counts as recalled only when
your answer is both semantically on-topic AND contains its hard facts (the
numbers/acronyms, or enough of its key words) — vague talk doesn't get credit.
The coach never sees the expected answers, so it can't leak them; it offers a
hint only after you stall, then moves on so you're never stuck. It prints a
coverage summary at the end.

**Status:** v1 + markdown-recall mode implemented, 145 tests pass, `--smoke` green.
The live mic loop is the one path validated manually (the dev environment had no
mic; the full ASR→LLM→TTS chain is covered by automated TTS→ASR round-trip +
full-turn tests).

## 路线图 / Roadmap

- **Phase 1 (MVP):** ✅ done — full local loop (faster-whisper + qwen3.5:4b + Kokoro
  + Silero), continuous half-duplex conversation, Anki sentence injection + nomic
  "practiced" scoring, latency instrumentation, native smoke test.
- **Phase 2:** ✅ markdown-recall mode — recall any markdown doc from memory via
  local-LLM extraction + honest (cosine + fact-anchor) coverage scoring. Plus an
  interactive launcher menu (`--menu`) and a `lv` alias. The menu's *live status
  display* (running-session stats) is still deferred.
- **Later:** 用 `nomic-embed-text` + `bce-reranker` 做语义检索，按话题动态调相关句子 + 跨会话间隔复习排程（见 `TODOS.md`）。

## License

TBD (intended permissive). Components used are MIT / Apache-2.0.
