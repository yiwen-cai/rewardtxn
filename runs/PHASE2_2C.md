# Phase 2C 阶段报告 — Reconciler + Selective Replay + Trace Runner（恢复面）

日期: 2026-08-27 | 门禁: **PASS (3/3)** | 回退点: `git checkout phase2b`（当前 `phase2c`）

## 设计

- **Reconciler** (`phase2_reconciler.py`)：崩溃后恢复决策，输入 = StepToken sidecar +
  seals.jsonl + rewards.jsonl → 输出 `recovery_plan.json`：
  - committed_iters（已提交步，不重放）/ resume_iter（恢复点）
  - replay_groups（ABORTED 混算组 + 崩溃不完整组）
  - replay_samples（混版本样本全量权威重算 + 崩溃缺失 reward 样本）
  - 预取尾部良性组（全 reward 有值但未满 K）**不恢复**（未消费数据无需重放）
- **Selective Replay**：CPU 权威 verifier (v1) 重算，复用 Rollout 前缀
- **Trace Runner** (`phase2_trace_runner.py`)：全切点回归断言（R1/R2/R3/Q0/L0/L2）

## 门禁验证

| 门禁 | 证据 | 结果 |
|---|---|---|
| **G2C1** Reconciler 恢复计划 | R1: 160 崩溃样本识别入重算; Q0: 32 样本重投递; L2: 已提交步 [3,7] → 恢复点 iter 7 | PASS |
| **G2C2** Selective Replay 节省 ≥30% | 实测 R3 99.9% / R1 100%; 保守口径 (1s/组调度) **92.7%**；Day4 B5 整步重跑基准 55.8s/步 | PASS |
| **G2C3** Trace Runner 全切点绿 | R1/R2(R3 代测)/R3/Q0/L0/L2 全部断言通过；元数据开销 0.39% | PASS |

## 新增实验 (2C)

| 实验 | 配置 | 结果 |
|---|---|---|
| p2c-slime-skew-sealresp | skew 窗口[20,39] + Seal + 完整 response, 20 步 | succeeded; ABORTED=20; 8882 条带 response |
| p2c-slime-crash-sealresp | crm_crash 窗口[20,39] + Seal + response, 20 步 | succeeded; 160 崩溃样本 (reward=None+response), 全 SEALED |

## 关键发现

1. **R1 崩溃的协议级恢复可行**：崩溃前 rollout 数据（response）落盘 → 重算仅需
   CPU reward（160 样本 ~0.5s），vs B5 整步重跑 279s → 节省 92.7-100%
2. **ABORTED 组全量权威重算**：混版本组 160 样本全部用 v1 重算（不只 v2 部分），
   保证组统计量一致性
3. **预取 buffer 不污染恢复计划**：slime 预取大量未消费组——Reconciler 只恢复
   "训练消费路径上"的组（崩溃组/ABORTED 组），尾部良性组忽略
4. **R2 语义 = R3 子集**（陈旧重试结果 = 版本不一致）→ Trace Runner 用 skew 数据代测，标注
5. 多窗口注入 (RTX_FAULT_WINDOWS) 实现但实测发现 crm_crash+skew 组合窗口触发
   rollout 层 OOM（注入交互边界行为）——单窗口与 Day 2 已知配置一致，如实记录

## 回退与版本
- `git tag phase2c`；回退: `git checkout phase2b` / `phase2c`
- 阶段内修复: reconciler 恢复范围（预取尾部误恢复）、Python 3.8 兼容（zip strict、
  泛型注解）、宿主机 verifier fallback（slime 缺失 → phase2_verifiers 纯函数副本）

## 产物
```
scripts/phase2_reconciler.py      # plan + selective_replay + recovery_plan/replay_result
scripts/phase2_trace_runner.py    # 全切点断言 + TRACE_REPORT.json
scripts/phase2_verifiers.py       # 纯函数 verifier 副本 (slime a6272da0 来源)
scripts/phase2_seal_rm.py         # +多窗口注入 + 完整 response 记录 + 崩溃前落盘
runs/p2c-slime-{skew,crash}-sealresp-K8-s42-20260827/   # 2C 实验 (20 步 succeeded)
runs/TRACE_REPORT.json / PHASE2_GATE2C.json
```
