# HANDOFF — Rehearse autonomous build

> 交接给后续会话/agent。目标：**自主分阶段实现到底**，阶段间做对抗审查，无需人工确认。
> Last updated: 2026-05-27 by Claude (Opus 4.7). Repo: https://github.com/DISSIDIA-986/Rehearse

## ✅ STATUS: v1 + F2 markdown-recall mode shipped

v1 (English conversation practice, Phases A–G) built, each phase gated by a Codex
adversarial review + a final comprehensive review (ship-blocker fixed). Then **F2
markdown-recall mode** (recall any markdown doc from memory; honest cosine+anchor
coverage scoring; coach persona that can't leak answers) + an interactive launcher
menu (`--menu`) and `rehearse` alias — each also gated by Codex review (T3/T4 checkpoints
+ a final + a menu review, all findings fixed).
**156 tests pass; `uv run rehearse --smoke` PASS** (1843 sentences, qwen3.5:4b
non-thinking, TTS→ASR verbatim, nomic-embed + Silero up).
The ONLY remaining manual step is the real-microphone run (this environment has no
mic): `uv run rehearse` (or `rehearse`) — see README "Run". Everything mic-less is automated.
Deferred: the menu's live status display (running-session stats).

## 用户的执行指令（权威）

自主分阶段推进，**中途不要和用户确认、不需要人工干预，直到完成最后一个任务**：

1. 每个阶段实现完，进入下一阶段**之前**，先对**上一阶段**做对抗性审查（Codex），
   确认任务按规划如期完成、无遗漏/错误/drift。
2. 有问题就**立即修复**，修完**自动进入下一阶段**（不问用户）。
3. 所有阶段完成后，再做**一轮独立、全面的对抗性审查**。
4. 通过后做**冒烟测试**（已定：原生 macOS，非 Docker）。

唯一已确认的人工决策见下方"已锁决策"，后续不再问。

## 已锁决策（不要重新讨论）

- **D1 串行化**：全串行 LLM→TTS + 短回复上限（系统提示限 1-3 句）。不做 token overlap（Ollama 无法中途暂停）。
- **D2 打断**：半双工默认（TTS 播放时静音麦克风 + 150ms guard）+ 键盘中断；`--full-duplex`（AirPods）才开语音 barge-in。
- **D3 练过度量**：MVP 用 `nomic-embed-text` 余弦相似度（用户话 vs 活跃目标句，超阈值=命中）。跨会话"自发产出"归 Phase 2。
- **D4 测试**：WAV fixture 集成 + 确定性模块全单测 + 手动全闭环。真实本地模型（用户偏好 real tests，非 mock）。
- **冒烟 = 原生 macOS**（非 Docker — MLX/麦克风/宿主 Ollama 在容器内不可用）：
  TTS→ASR 自动往返集成测（无需麦克风）+ 启动健康检查（think 探针/模型加载/设备枚举）+ 交付一条真麦手测命令。

完整设计与盲区清单：`docs/DESIGN.md`（含 v2 Hardening + v3 Eng Review Lock + GSTACK REVIEW REPORT）。

## 环境（用户的 Mac Studio，darwin）

- Apple M1 Max, 32GB。uv 0.8.0, Python 3.13, Homebrew, Ollama（`qwen3.5:4b` 已拉，**用它，不是 9b**；非思考模式）。
- 关键坑：① 不全用 MLX（ASR 留 CPU 避免 GPU 争用）② 单音频设备 48k→16k/24k 一次性重采样，绝不每轮重开
  ③ `OLLAMA_KEEP_ALIVE=-1` 防冷启动 ④ think:false 必须经探针验证无 `<think>` 块 ⑤ `deepseek-r1` 不用（思考模型）。
- 跑测试：`uv run pytest`。跑解析器：`uv run python -m rehearse.anki_loader data/*.xml`。

## Anki 数据（私有，gitignored 在 `data/`）

AnkiApp XML 格式。`rehearse/anki_loader.py` 已解析（详见其 docstring）：
forgettable1105.xml(718) + GoogleNews.xml(1148) → 1843 unique，1136 带中文，810 带地道改写。
`lang` 属性会说谎，角色按内容判（CJK→translation，差异英文字段→native）。

## 进度（阶段 = 任务分组，每段一个 Codex 审查门）

- [x] **Phase A — anki_loader (T5)**：✅ 已完成并推送（commit 561f15c）。9 测试过。
- [x] **Phase B — 纯逻辑模块**：text_sanitize（剥 markdown/emoji）、sentence_chunker（切句, 标点/无标点 token 上限/缩写不误切）、
      session_seeder（按最久未练抽 2-4 句）、prompt_builder（注入目标句 + plain-speech 指令）、practiced_scorer（nomic 余弦, D3）。全单测。
- [x] **Phase C — LLM (T2)**：llm_client（Ollama `/api/chat`, think:false）+ 启动 think 探针（断言无 `<think>` + TTFT<阈值）+ 短回复上限。集成测打 qwen3.5:4b。
- [x] **Phase D — 音频/ASR/TTS (T3,T7)**：audio_io（单设备, 重采样数学单测）、vad（Silero 端点状态机, 起始60-90ms/收尾250-350ms, 合成概率序列单测）、
      asr（faster-whisper small.en, CPU）、tts（Kokoro via mlx-audio, GPU, 保温）。集成：**TTS→ASR 往返**（Kokoro 生成已知文本→whisper 转写≈原文，无需麦克风）。
- [x] **Phase E — 双工 + 主循环 (T4)**：duplex（半双工静音+guard / 键盘中断 / --full-duplex）、main_loop（全串行 asyncio：VAD→ASR→LLM→逐句TTS，常驻不退出直到 stop/Ctrl-C）。
- [x] **Phase F — 集成 harness (T8) + 最终全面对抗审查**：WAV fixture 集成测；然后独立全面 Codex 审查整个代码库。
- [x] **Phase G — 原生冒烟**：启动健康检查 + TTS→ASR 往返冒烟；交付 `scripts/mic_test`（一条命令的真麦手测）。

任务卡 T1-T8 详见 `docs/DESIGN.md` 的 Implementation Tasks；JSONL 在 `~/.gstack/projects/LocalVocal/tasks-eng-review-*.jsonl`。
注：T1 spike 已被实际实现吸收——直接自写（不 fork eauchs 那个 3-commit demo），在 Phase C/D 验证"非思考TTFT/TTS首音/无回声"。

## 每个 Phase 的收尾动作（固定流程）

1. 实现模块 + 写测试 → `uv run pytest` 全绿。
2. **Codex 对抗审查本阶段 diff**（read-only，挑遗漏/错误/drift）：
   `codex exec "<审查提示>" -C "$(pwd)" -s read-only --skip-git-repo-check -c 'model_reasoning_effort="medium"' < /dev/null`
   （medium 即可；high+websearch 太慢，曾卡 10min）。
3. 有发现→立即修→重测→重审直到 clean。
4. `git add` + commit（conventional, 末尾 Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>）+ `git push`。
5. 自动进入下一 Phase，**不问用户**。

## 不能自主完成的（交付时说明）

- **真麦克风全闭环 E2E**：本环境无麦克风/音频输出，无法自动驱动。用 TTS→ASR 往返覆盖音频链路；
  真麦手测留给用户跑 `scripts/mic_test`。
- **Docker**：不适用（已与用户确认走原生）。

## git 注意

- 全局 `~/.gitignore_global` 忽略 `*.md` 和 `.gitignore`；本仓库 `.gitignore` 已加 `!*.md` 反忽略。
  新 markdown 若没被追踪，用 `git add -f`。
- `data/` 是私有 deck，**永不提交**（公开仓库）。测试用 `tests/fixtures/` 的合成 fixture。
- commit 身份本仓库设为 `DISSIDIA-986` / `DISSIDIA-986@users.noreply.github.com`（不暴露真实邮箱）。
