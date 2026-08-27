# Phase 2 分阶段实施计划 — 最小事务协议

**目标**: 实现 7 模块 (Seal/CAS/StepManifest/StepToken/Reconciler/SelectiveReplay/TraceRunner)
并完成 8 卡端到端评测。每阶段: **严格门禁 → 文档 → git 提交(tag) 可回退**。

## 阶段划分

| 阶段 | 内容 | 门禁 (全部必须实证通过) | 回退点 |
|---|---|---|---|
| **2A 数据面** | Group Seal + Reward CAS (custom-rm 包装层) | G2A1: R3 注入下混版本组 0 个进入训练; G2A2: 干净组 100% 放行 (无副作用); G2A3: 协议元数据开销 <5% | tag phase1-complete |
| **2B 持久化面** | StepManifest + StepToken (checkpoint 元数据包装) | G2B1: save 时 manifest+token 落盘; G2B2: kill 后重启能识别已提交步 (不重放); G2B3: 恢复路径无死锁 (Day4 死锁解除) | tag phase2a |
| **2C 恢复面** | Reconciler + Selective Replay + Trace Runner | G2C1: Q0/R1 崩溃窗口 Reconciler 恢复样本; G2C2: Selective Replay 实测节省 ≥30% (vs B5); G2C3: Trace Runner 全切点回归绿 | tag phase2b |
| **2D 端到端** | 8 卡 4+4 评测 (Qwen3-4B) | G2D1: 吞吐开销 <5%; G2D2: 恢复时延 <60s; G2D3: 收敛曲线与 Clean Oracle 对齐 | tag phase2c |

## 阶段门禁流程 (每阶段强制)

1. 实现 (协议层/包装层, 不改 third_party 锁定源码)
2. 门禁验证 (脚本化, 输出 gate 结果 JSON)
3. 阶段文档 (PHASE2_<阶段>.md: 设计/实现/验证/结论)
4. git commit + tag (phase2a/b/c/d); 失败时 `git checkout <tag>` 回退

## 架构原则 (继承 Day 2-4 教训)

- **包装层注入**: 全部逻辑通过 custom-rm / env 挂载 / 包装器实现, 不修改
  third_party 锁定源码 (a6272da0/b83d1f40/8497a52a)
- **复用资产**: day2_custom_rm.py (Seal 注入点), day4_inject.sh (kill 注入),
  rewards.jsonl instrumentation (CAS/Replay 判定输入)
- **模拟器对照**: 每个门禁以 DAY5 模拟器结论为预期 (0 Invalid / 2.5% 开销 /
  85-95% 节省), 实测复核
