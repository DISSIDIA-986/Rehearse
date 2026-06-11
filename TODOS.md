# TODOS — Rehearse

> 由 /plan-eng-review 生成（2026-05-27）。Phase 1 MVP 之后的延迟项，均已在 `docs/DESIGN.md` 决策过。

## Phase 2 — 增强（MVP 验证有用后再做）

### T-P2-1: 语义检索注入（nomic-embed + bce-reranker）
- **What:** 把全部 Anki 句子向量化，对话中按当前话题检索语义相关句子动态喂给助手。
- **Why:** 比"每轮随机抽 2-4 句"更贴话题，真正实现"找相关对话帮我复习"。
- **Pros:** 复用已装的 nomic-embed + bce-reranker；话题相关性更强。
- **Cons:** Codex 判定对 2000 句规模是过度工程；先让 MVP 的随机/间隔抽样证明"注入确实改变对话行为"再做。
- **Context:** MVP（D3）已用 nomic 做"练过"度量；本项是把 nomic 再用于检索侧。存向量可用 sqlite-vec 或纯文件。
- **Depends on:** MVP 跑通且证明注入有效。

### T-P2-2: 跨会话"无提示自发产出"追踪（SQLite 间隔复习）
- **2a [DONE 2026-06-11]** 跨会话持久化：`rehearse/practice_store.py`（stdlib sqlite3，零新依赖）
  把每句的 practice count + last_ts 存到 `~/.local/share/rehearse/practice.db`，启动时 `load_stats()`
  载入、每轮命中后 `record_practiced()` 落盘 → `select_targets` 现在跨会话挑"最久未练"，
  真正闭合 conversational spaced repetition。`--no-persist` 退回 v1 纯内存；`--practice-db` 指定路径。
  内存/磁盘逐字符一致、损坏库隔离重建、WAL、锁库快速降级、错误 schema 隔离——全部有测（25 个用例）。
  三轮 Codex 对抗审查 CLEAN。
- **2b [TODO]** "无提示自发产出"检测：最高级掌握信号（D3=A 的 10/10）。延迟——需更严阈值 + 排除当轮
  active 目标 + 事件日志，证明不污染排程后再接入。`practice_store` 的 schema 已留 `user_version` 迁移位。
- **Why:** MVP 的"练过"只到"复述/改写命中"（D3=A 的 7/10）；2b 补齐到 10/10。
- **Context:** D3 选 A 时明确把"自发产出"划到 Phase 2。2a 由 Codex+Claude 子代理联合评审选为 Phase 2 首做项。

### T-P2-3: 全双工外放 + 真 AEC
- **What:** 接 WebRTC AEC 子系统，实现裸机外放且随时语音打断。
- **Why:** 当前半双工外放不能语音打断（D2）；AirPods 才能全双工。
- **Pros:** 体验最好，外放也能自然插话。
- **Cons:** 独立子系统，复杂度高，与"别太复杂"冲突。
- **Context:** Codex 标为 BLOCKER 级体验项但 v1 不做；D2 选了键盘中断 + AirPods 语音打断绕开。
- **Depends on:** MVP 稳定。

### T-P2-4: 启动菜单的运行时实时状态显示（live status TUI）
- **What:** F2 已交付 `--menu` 选择式启动器 + `rehearse` alias；剩下 F1 设想的"运行中实时状态"——
  会话进行时显示延迟、覆盖度/练过计数、当前 item 等的 TUI 状态栏。
- **Why:** 用户每天在 Ghostty 里跑，想边练边看进度，不必等结束才看 summary。
- **Pros:** 复用现有 summary 数据；菜单已搭好入口。
- **Cons:** 真 TUI（curses/rich）会引入依赖，与"单二进制/零运行时依赖"取舍冲突；先证明菜单够用。
- **Context:** F2 阶段明确把 live-status 划为 deferred；当前菜单只做启动前选择。
- **Depends on:** 无（独立增强）。

## 启动前 Spike（Phase 1 第一步，不是延迟项）
见 `docs/DESIGN.md` v2 Hardening：fork 前先验证 ① 连续循环 ② 非思考 qwen 且 TTFT 低
③ TTS 首音可接受 ④ 半双工无回声。
