# Phase 3 分阶段实施计划 — 正确性/可用性/规模化闭合

> **归档状态（2026-08-29）**：Phase 3A/3B/3C 全部完成并通过 9/9 阶段门禁，
> 最终结项见 `runs/PHASE3_FINAL.md`、`runs/PHASE3_ARCHIVE_MANIFEST.json` 与
> `phase3-final` tag。本文保留实施计划，同时按最终验收口径校正 3C 的实际配置。

**目标**: 在 Phase 2 协议栈 (已交付, tag phase2-final) 基础上, 闭合三个层次:
3A 消费侧正确性 (0 混算进训练) → 3B 无人干预自动恢复 → 3C 规模化回归与审计。
每阶段: **严格门禁 → 文档 → git 提交(tag) 可回退** (沿用 Phase 2 流程)。

## 修订说明 (2026-08-27, 基于 Phase 2 实施经验)

| 原交接清单 | 修订后处置 |
|---|---|
| ① 自动恢复闭环 (watchdog+恢复计划+消费 gate) | **拆分**: 消费 gate 独立为 3A (正确性优先), 自动恢复为 3B (可用性) |
| ② TransferQueue 真实队列回滚 | **砍掉**: Manifest 去重 + 幂等重投递已覆盖语义 (2C Q0 恢复计划实证), 独立队列网关成本不划算 |
| ③ 8 卡 4+4 复测 | 保留降级为 3C 可选 (外部 GPU 窗口; 协议层与卡数无关已论证) |
| ④ 长程收敛 + Trace Runner 扩展 | 拆分: Trace Runner 自动化提前为 3C 必做 (3B 的测试面), 长程收敛为 3C 必做 |

新增: 多进程 CAS 验证 (2A 边界: `_seen_logical` 进程内 set)、恢复审计报告。

## 阶段划分

| 阶段 | 内容 | 门禁 (全部必须实证通过) | 回退点 |
|---|---|---|---|
| **3A 消费侧正确性** | Seal AUTO_FIX 就地修正: ABORTED 组返回权威 v1 值而非混合值 (RTX_SEAL_AUTO_FIX=1, 不碰训练循环) | G3A1: 注入实验训练消费侧 100% 样本 reward == v1 权威重算值; G3A2: 干净端到端运行 0 ABORTED/0 autofix，并以同一输入 AUTO_FIX on/off 配对断言逐样本一致; G3A3: 同 group-rm 模式 skew+AUTO_FIX vs clean 吞吐下降 <5% | tag phase2-final |
| **3B 可用性闭合** | 自动恢复链: 死亡检测 → 恢复计划自动执行 → 自动重启 → 接续 (容器 entrypoint 包装) | G3B1: 注入真实训练进程 kill -9，无人工干预自动继续完成 (恢复后 ≥10 步); G3B2: 恢复正确性 (loss 轨迹对照无崩溃在噪声内 + 已提交步不重复计); G3B3: 端到端恢复时延 <5 min。RM 异常 R1 被 fully-async 吞并表现为组丢弃，由 3A/2C 覆盖，不作为进程死亡门禁 | tag phase3a |
| **3C 规模化回归** | 注入回归自动化 + 0.5B 新配置 100 步长程 + 1.5B 20 步模型升级复验 + 多进程 CAS + 审计报告；8 卡复测为门禁外可选项 | G3C1: 回归套件全绿 (一条命令, 全切点断言); G3C2: 同 Seal 配置 100 步 loss 平均绝对差 <0.05，且 1.5B 协议语义一致; G3C3: 多进程并发同组 → 恰好 1 条权威记录 + 幂等 | tag phase3b |

## 3A 设计要点 (Seal AUTO_FIX)

- **实现位置**: `scripts/phase2_seal_rm.py`, 组完成时若版本向量不一致:
  该组**所有样本** reward = 权威 `_v1_reward` 重算值 (v1 样本值不变 = 零修改,
  v2 样本修正), 标记 `autofix=true` 写 seals.jsonl, rewards.jsonl 保留 old/new 双值
- **为什么不是包装训练循环**: slime 逐样本消费, "阻止组进入"需改消费路径
  (违背零 third_party 改动); 就地修正在 RM 层完成, 语义 = 检测 + 修正
- **正确性底座**: 2C 已验证 Replay 重算正确性 (80 样本 0 不一致) + 确定性 (纯函数)
- **开销**: ABORTED 组 8 样本重算 ~0.5s CPU; 干净组路径零改动

## 3B 设计要点 (自动恢复链)

- **前置探针 (风险前置, 半天)**: slime `--load` resume 可行性
  - 路线 A: `--load iter7` + num_rollout 匹配注入 (包装层) → 可行则走 resume
  - 路线 B: 冷启动 + 协议层重放未提交组 (已提交步丢弃重训) → 探针否定 A 时采用
  - **探针结果写入 3B 文档, 决定 3B 最终路线** (不预先承诺)
- **容器 entrypoint 包装**: 主进程退出/日志停滞检测 → `phase2_reconciler plan`
  自动执行 → 重启训练 (加载已提交步 + 重放未提交组) → 幂等防重
- **验证方法（最终采用）**: checkpoint iter9 落盘后真实 kill -9 训练进程 + 全程无人干预观察；
  `crm_crash` 的 RM 异常会被 fully-async 容错捕获并退化为组丢弃，因此归入 3A/2C 的
  Seal/Replay 正确性覆盖，不将它冒充进程死亡恢复

## 3C 设计要点 (规模化回归)

- `scripts/phase3_regress.sh`: 一条命令跑全切点 (R1/R2/R3/Q0/L0/L2 +
  Seal/Manifest/Reconciler/Replay 断言), 输出回归 JSON
- 长程最终验收: 0.5B 新配置 Seal 100 步 vs 历史同 Seal 配置 100 步，逐步 loss
  平均绝对差 0.0065 <0.05；另以 1.5B skew 20 步验证模型升级后的
  Seal/CAS/AUTO_FIX/checkpoint 语义。两项证据职责分离，不将 1.5B 20 步表述为长程对照
- 多进程 CAS: 2 个进程并发投递同一 (LogicalID,Epoch,Attempt) → 文件锁级恰好 1 条
- 审计报告: Reconciler 输出 Markdown/HTML 恢复报告 (ABORTED/修正/重放/已提交清单)
- 8 卡 4+4 复测: 外部 GPU 窗口就绪后执行 (可选门禁外)

## 阶段门禁流程 (每阶段强制, 同 Phase 2)

1. 实现 (协议包装层, 不改 third_party 协议/训练循环源码；模型配置兼容 patch 单独归档)
2. 门禁验证 (脚本化, 输出 gate 结果 JSON: PHASE3_GATE3{A,B,C}.json)
3. 阶段文档 (runs/PHASE3_3{A,B,C}.md: 设计/实现/验证/结论/边界)
4. git commit + tag (phase3a/b/c); 失败时 `git checkout <tag>` 回退

## 架构原则 (继承)

- **包装层注入**: 协议逻辑全部通过 custom-rm / env / entrypoint 包装器实现，不修改
  third_party 的协议/训练循环源码 (a6272da0/b83d1f40/8497a52a)。为运行 Qwen2.5-1.5B，
  对 slime 模型配置的 `rotary-base` 做了 10000→1000000 兼容修正；该单行 patch 独立归档，
  不属于 RewardTxn 协议实现
- **复用资产**: phase2_seal_rm.py (AUTO_FIX 挂载点), phase2_reconciler.py (3B 恢复链),
  phase2_trace_runner.py (3C 回归底座), day4_inject.sh (kill 注入)
- **预期对照**: 3A=Day5 g4 门禁 (0 Invalid Commit) 的真实消费侧闭合;
  3B=Day4 L2 死锁的自动恢复侧闭合; 3C=Phase 2 全部门禁的规模化复现

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| slime resume 死锁 (L2) 无法包装绕过 | 3B 前置探针先行, 路线 B 保底 (代价: 已提交步重训) |
| AUTO_FIX 误改干净样本 | G3A2 端到端 13,600 样本零误改 + 同输入 on/off 8/8 配对一致性断言 |
| 自动恢复循环崩溃 (恢复本身失败) | 幂等 + 恢复计数上限 (≥3 次失败转人工告警) |
| 外部 GPU 抢占 (8 卡/长程) | 3C 长程采用 4 卡 0.5B，并补 4 卡 1.5B 协议升级复验；8 卡列可选 |

## 验收出口 (Phase 3 完成定义)

1. 3A/3B/3C 全部门禁 PASS (PHASE3_GATE3*.json)
2. 一次无人干预的端到端故障恢复演示 (注入 → 自动恢复 → 训练完成)
3. 真实训练中 0 混算进入 (样本级证据)
4. 全切点回归一条命令可复现
5. 文档链完整: PHASE3_PLAN.md / 3{A,B,C}.md / 3 个 GATE JSON / git tag 3 个
6. 结项归档完整: PHASE3_FINAL.md / ARCHIVE_MANIFEST.json / `phase3-final` tag，
   portable evidence 可校验，大 checkpoint 有明确本地保留策略
