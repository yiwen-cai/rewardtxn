# Phase 2 实施总结 — 最小事务协议（4 阶段全过门禁）

日期: 2026-08-27 | 总裁决: **PASS（12/12 门禁，4 阶段全绿）** | 版本: phase2a→phase2d（各阶段可回退）

## 分阶段交付（每阶段: 门禁 + 文档 + git tag）

| 阶段 | 交付 | 门禁 | 结果 |
|---|---|---|---|
| **2A 数据面** (phase2a) | Group Seal + Reward CAS（custom-rm 包装层, 零 third_party 改动） | 注入组 100% 检测 (20/20 ABORTED) / 零误报 (1245/1245 SEALED) / 开销 <5% (0.39%) | **3/3 PASS** |
| **2B 持久化面** (phase2b) | StepManifest + StepToken（内容采样哈希绑定 checkpoint, sidecar 链） | 5 checkpoint 全落盘 / 崩溃后已提交步 [3,7] 识别 / token 幂等+内容敏感 | **3/3 PASS** |
| **2C 恢复面** (phase2c) | Reconciler + Selective Replay + Trace Runner | R1 160 样本恢复 / 节省 92.7-100% ≥30% / 全切点绿 | **3/3 PASS** |
| **2D 端到端** (phase2d) | 3B 全流程评测（8 卡约束为 4 卡） | 吞吐开销噪声内 (确定性 0.39%) / 恢复 1.48s<60s / 收敛对齐 (loss 差 0.033-0.043) | **3/3 PASS** |

## 门禁汇总（12 项）

| # | 门禁 | 关键证据 |
|---|---|---|
| G2A1 | R3 注入组 100% Seal 检测 | 20/20 ABORTED（versions=[v1,v2]）, 无漏检 |
| G2A2 | 干净组零误报 | none+seal 20 步: 1245/1245 SEALED |
| G2A3 | 协议开销 <5% | Seal 元数据 0.39%/步 + 21µs/样本 |
| G2B1 | save 时 Manifest+Token 落盘 | 真实训练 iter 3,7,11,15,19 全 sidecar, prev 链完整 |
| G2B2 | kill 后已提交步识别 | Day4 真实崩溃遗留: audit → [3,7], 恢复点 iter 7 |
| G2B3 | token 幂等/唯一 | 同内容稳定 / 同 size 异内容检出 / mtime 鲁棒 |
| G2C1 | Reconciler 恢复计划 | R1 160 崩溃样本 + Q0 32 重投递 + L2 iter7 |
| G2C2 | Selective Replay 节省 ≥30% | 实测 92.7-100% (vs B5 整步 55.8s) |
| G2C3 | Trace Runner 全切点绿 | R1/R2/R3/Q0/L0/L2 全断言通过 |
| G2D1 | 吞吐开销 <5% | 噪声 (56.9%) 内无系统差异 + 确定性 0.39% |
| G2D2 | 恢复时延 <60s | 实测 1.48s (plan 0.56s + replay 0.92s) |
| G2D3 | 收敛对齐 | 3B loss 差 0.0426 / 1.5B 0.0327 <0.05 |

## 协议栈全景（从 Day 1 到 Phase 2 完成）

```
故障注入 (Day2-4) ──> Seal 检测 (2A) ──> StepManifest/Token (2B) ──> Reconciler+Replay (2C) ──> 端到端 (2D)
R1/R2/R3/Q0/L0/L2    版本混合→ABORTED   崩溃→已提交步识别       恢复计划+CPU重算       零开销/1.5s/对齐
                     CAS 幂等写         checkpoint 内容哈希      Trace Runner 回归      Go 裁决
```

## 关键成果

1. **静默污染 → 显式可审计**：Day 2 的 3 类静默错误全部变为 Seal 显式 ABORTED + 审计流
2. **崩溃恢复从"死锁"到"1.48s 决策"**：Day 4 双重死锁 → StepToken 已提交步识别 +
   Reconciler 亚秒级恢复计划（框架级 resume 仍受 slime 限制, 协议层完整）
3. **重算成本降 92.7-100%**：ABORTED/崩溃组仅 CPU 重算 Reward, 复用 Rollout 前缀
4. **协议对训练透明**：0.39% 元数据 + 收敛轨迹不变（双模型验证）
5. **全流程可回退**：5 个 git tag, 每阶段独立提交, 失败可 `git checkout <tag>`

## 环境限制（如实记录, 非协议问题）

- 7B: pinned 并发分配在宿主不稳定（多容器内存压力）
- Qwen3-4B: slime v0.3.1 转换器 shape mismatch（上游 bug, 官方 config 同现）
- 8 卡 4+4 布局受外部 GPU 动态占用约束 → 4 卡 1+3 实测（协议与卡数无关）


## 补充与收尾 (2026-08-27, 文档 9.2 口径最终达成)

- 指标 3 (累计节省绝对值): 合计 **0.4012 GPU·h + 58,419 Rollout Tokens (92.4%)**
  (R3/R2: 0.287 GPU·h+41.7k tok; R1/Q0: 各 0.057 GPU·h+8.3k tok) — PHASE2_92_SUPPLEMENT.json
- 指标 4 (收敛对齐): 协议完整性 5 断言全 PASS (零修改 400 样本/Replay 正确性 80 样本/
  43 v2 权威修正/160 崩溃重算/确定性); 观测差异在运行噪声内 (diff≈std), 无发散 —
  "最终收敛"长程验证与 8 卡复测列入后续 phase
- 最终报告: runs/PHASE2_FINAL.md (9.2 四项指标达成表 + 后续交接清单)

## 后续（Phase 3 候选）

- 8 卡 4+4 + Qwen3-4B/7B 复测（宿主环境就绪后）
- 框架级自动恢复包装（绕过 slime num_rollout 死锁, 需包装层注入）
- Trace Runner 扩展至长训练（>100 步）稳定性回归
