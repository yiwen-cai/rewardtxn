# Phase 3C 阶段报告 — 规模化回归 / 长程收敛 / 新配置复验

日期: 2026-08-29 | 门禁: **PASS (3/3)** | 回退点: `git checkout phase3b`（当前 `phase3c`）

## 背景

Phase 3A/3B 已闭合消费侧正确性与自动恢复；内存/IO 优化落地后（并发 512→64、
Ray object store 16GiB、CAS 索引化、checkpoint 滚动保留、资源门禁），本阶段以
新配置复验规模化语义：长程收敛、模型升级、一键回归。

## 门禁判定

| 门禁 | 结果 | 证据 |
|---|---|---|
| G3C1 一键回归全绿 | PASS | `phase3_regress.sh` **9/9**（显式 R1/R2/R3/Q0/L0/L2 六切点矩阵 + seal/配对零误改/CAS 单进程/CAS 4 进程/replay/manifest/零修改/恢复历史） |
| G3C2 长程收敛对齐 | PASS | 新配置 100 步 vs 历史 100 步（同 0.5B+seal）逐步 loss 平均绝对差 **0.0065 < 0.05**；新实验 100 步整体 std 0.0056、后 50 步均值 0.0033 |
| G3C3 多进程 CAS | PASS | 4 进程并发同组 → 恰好 1 条权威记录（cas_multi_process PASS） |

## 补充实验

### B. 0.5B 长程复验（long profile, 100 步）

- 实验: `p1-slime-none-K8-s42-20260829-001020`
- 配置: `RTX_PROFILE=long`（`--no-save-optim` + `CKPT_KEEP=1` + `SAVE_INTERVAL=100`），fault=none，seal 链路
- 结果: SUCCESS
  - 100 步 loss 收敛：均值 0.0063，后 50 步均值 0.0033
  - 协议：2751 组全 SEALED、0 ABORTED；10368 条 reward 与 v1 权威 **0 不一致**
  - **存储验证**：100 步全程仅 1 份 checkpoint（943M，no-save-optim），滚动保留生效
- 与历史 p3c-seal（100 步）loss 对齐：逐步平均绝对差 0.0065 < 0.05 → 新配置不改变训练语义

### C. 1.5B 模型升级验证（phase3b, skew[20,39], 20 步）

- 实验: `p1-slime-skew-K8-s42-20260829-100342`
- 配置: `RTX_PROFILE=phase3b`（保留 optimizer state + `CKPT_KEEP=2`），1.5B，skew 窗口 [20,39]
- 结果: SUCCESS
  - 窗口组 **20/20 ABORTED + autofix**；4296 条 reward 与 v1 权威 **0 不一致**（0 混算）
  - checkpoint：iter_9 + iter_19（keep=2 生效），41G
  - **协议栈在 1.5B 上与 0.5B 完全一致**（Seal/CAS/AUTO_FIX/滚动保留）
- 过程修复: `qwen2.5-1.5B.sh` 的 `--rotary-base` 错误（10000→1000000，与 HF config 一致）

### 资源门禁（新增基础设施）

- `scripts/resource_gate.py`：启动前检查 /public≥200G、根盘≥100G、目标 GPU 空闲显存≥30G，
  任一不达标拒绝启动（exit 2），结果写 `logs/resource_gate.json`
- 训练期间每 60s 采样 mem/disk/GPU → `logs/resource_health.jsonl`
- A/B/C 全部实验门禁 PASS；训练期间主机内存 used 38-54G / avail 440-470G

## 修复记录（本阶段）

- `third_party/slime/scripts/models/qwen2.5-1.5B.sh`: `--rotary-base 10000` → `1000000`
  （hf_validate 断言失败: rope_theta 1000000 vs rotary_base 10000）。这是模型配置兼容修正，
  非 RewardTxn 协议/训练循环改动；结项时已独立归档为
  `patches/slime-qwen2.5-1.5b-rotary-base.patch`

## 产物

```
runs/PHASE3_GATE3C.json                    runs/PHASE3_3C.md (本报告)
runs/PHASE3_REGRESSION.json (9/9 PASS)     runs/PHASE3_EXP_MEMIO.json
runs/p1-slime-none-K8-s42-20260829-001020 (B: 0.5B long 100步)
runs/p1-slime-skew-K8-s42-20260829-100342 (C: 1.5B skew 20步)
scripts/resource_gate.py (资源门禁)
git tag phase3c
```

## 边界

- G3C2 对齐使用历史 p3c-seal 同配置 100 步数据（clean 对照组历史不完整，
  以同 seal 配置对比 + 新实验自身收敛性双重证据）
- 8 卡 4+4 复测仍为可选门禁外（外部 GPU 窗口；协议层与卡数无关已论证）
