# Day 5 实验汇总 — B0-B5 全基线对齐、开销核算与 Go/No-Go 裁决

日期: 2026-08-26 | 修正方案: DAY5_PLAN.md (5 项修正全部执行)

## 裁决: GO (7/7 Go 条件全部满足, judge_gates.py 与手工矩阵双重确认)

## 新增实验与核算 (Day 5 产出)

### g7 上游核查 (G7_UPSTREAM_AUDIT.md)
slime main (a3f500977f, 08-26) / AReaL main (94ce16558b) / TransferQueue main (5cb184ef)
均无等价的 Durable Step Commit / StepManifest / StepToken 机制 (匹配均为误报)

### g3 受控实验 (g3_controlled_delta.json)
同一样本仅 reward 版本混入 (R3 注入数据): 权重更新 Delta L2 = 0.2598,
梯度余弦相似度 = 0.7548 (<0.999) -> Optimizer Delta 漂移可测

### B1 校验器穿透验证 (b1_validator_result.json)
显式 Group ID + 组大小校验: 20/20 注入组全部放行 (ID 合法/大小=K=8/字段完整),
80 个跨版本混算样本穿透 -> g1 子条件 (完整组校验之后错误仍发生) 成立

### B2 AReaL 实证 (B2_AREAL_RESULT.md)
容器内 tests/test_grouped_rollout_workflow.py 4/4 passed,
drop_incomplete_group = "dropping entire group" (安全丢弃但 100% Rollout 浪费);
组完整时 (R3/R2) 不触发, 版本混算穿透; 无 queue/StepToken 机制

### g4/g6 协议语义模拟器 (DAY5_SIMULATOR.json)
在 Day 2-4 真实注入数据上重放 Seal/Fencing/StepToken/Reconciler:
- 全切点 Invalid Committed Step = 0 (g4 PASS)
- 协议元数据开销 2.54% < 5% (g6 PASS)
- R3/R2: 20 组 ABORTED; R1: 4 组 ABORTED; Q0: 32 样本 Reconciler 恢复; L2: StepToken 绑定 iter 7

### g5 重算节省核算 (DAY5_G5.json)
| 场景 | B2 成本 | B5 成本 | Selective Replay | 节省 vs B2 | vs B5 |
|---|---|---|---|---|---|
| R1 (32 样本) | 31.3s | 87.6s | 4.4s | 85.9% | 95.0% |
| R3/R2 (80 样本) | 156.5s | 438.0s | 22.0s | 85.9% | 95.0% |
| Q0 (32 样本) | 31.3s | 87.6s | 4.4s | 85.9% | 95.0% |

## B0-B5 全矩阵 (DAY5_MATRIX.json)

| 基线 | R1 崩溃 | R2/R3 混算 | Q0/L0/L2 窗口 | 重算代价 |
|---|---|---|---|---|
| B0 默认 | 静默丢弃 32 样本 | 完全静默 (组偏移 -0.26~-0.28) | 永久丢失/无恢复 | 0 (训练损坏) |
| B1 组 ID 校验 | 部分拦截 | **穿透** (无版本检查) | 不覆盖 | 需整组丢弃 |
| B2 丢组 (AReaL) | 安全丢弃 | **穿透** (组完整不触发) | 不覆盖 | 100% Rollout |
| B3 幂等去重 | 不覆盖 | 不覆盖 | **穿透** (防重复不防窗口) | 数据丢失 |
| B4 Occupy/Consume | 不覆盖 | 不覆盖 | **窗口存在** (无回滚) | 数据丢失 |
| B5 每步 ckpt | 可恢复 | 可恢复 | 可恢复 | 100% 重跑 + 保存开销 65-100% step 时间 |
| **RewardTxn** | ABORTED+重算 | ABORTED (0 Invalid) | Reconciler/StepToken | **仅重算 Reward (节省 85.9-95.0%)** |

## Go 门禁 (8.1 节) 逐项

| 条件 | 证据 | 结果 |
|---|---|---|
| g1 ≥2 类静默错误 (含组校验后) | 3 类 (R1/R2/R3) + B1 穿透 | PASS |
| g2 真实崩溃窗口 | Q0 (Available:0) / L0 / L2 / C1 | PASS |
| g3 梯度污染可测 | Delta L2=0.260, 余弦 0.755 | PASS |
| g4 零无效提交 | 模拟器全切点 0 | PASS |
| g5 重算节省 ≥30% | 85.9% (vs B2) / 95.0% (vs B5) | PASS |
| g6 协议开销 <5% | 2.54% | PASS |
| g7 无现成等价保证 | 三栈 main 核查 | PASS |

## No-Go 条件核查 (8.2 节)
全部 7 项 No-Go 均不成立 (错误不能仅靠组校验消除 / 窗口真实存在 / B5 开销高 /
exactly-once 可绑定 Durable Checkpoint / 故障注入均真实可触发 / 上游未合入 / 非 Lineage 界面)

## 产物
```
runs/DAY5_MATRIX.json, DAY5_VERDICT.json, DAY5_SIMULATOR.json, DAY5_G5.json
runs/G7_UPSTREAM_AUDIT.md, B2_AREAL_RESULT.md, DAY5_PLAN.md
runs/day5-final/{meta.json, metrics.json, verdict.json}  (judge_gates.py 输出)
runs/p1-slime-skew.../g3_controlled_delta.json, b1_validator_result.json
scripts/day5_g3_delta.py, day5_b1_validator.py, day5_simulator.py, day5_g5.py, day5_verdict.py
```

## 结论
满足全部 7 项 Go 条件, 批准进入 Phase 2 (最小事务机制实现: Group Seal + Reward CAS +
StepManifest + StepToken + Reconciler + Selective Replay + Trace Runner)。
g4/g6 基于协议语义模拟器评估 (修正文档循环依赖), Phase 2 原型实现后应以实测复核。
