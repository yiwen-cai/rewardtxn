# Phase 3A 阶段报告 — 消费侧正确性闭合（Seal AUTO_FIX）

日期: 2026-08-28 | 门禁: **PASS (3/3)** | 回退点: `git checkout phase2-final`（当前 `phase3a`）

## 背景与方案修正

Phase 2 遗留: ABORTED 组检测已闭环, 但**真实训练消费侧 0 混算**未闭合
(逐样本 RM 路径下"阻止混算组进入训练"不可行)。3A 方案 (按 PHASE3_PLAN.md 修订):

**关键事实调查** (slime 源码, 零改动原则):
- 默认路径 `generate_and_rm`: 每样本独立 `await async_rm()` → 组内样本并发, 无法在返回前知道整组状态
- **原生 `--group-rm` 模式** (`generate_and_rm_group`): 组生成完成后一次性
  `batched_async_rm(args, group)` (sglang_rollout.py:328-332), 返回值直接
  `zip` 到 `sample.reward` → **RM 返回什么, 训练器就消费什么**
- fully_async 训练路径使用 `generate_and_rm_group` (fully_async_rollout.py:144); eval 路径禁用 group_rm (仅训练可用)

→ AUTO_FIX 挂载在组级批调用: 组完成时检测版本混合, **返回前**将整组修正为 v1 权威值。

## 实现

`scripts/phase2_seal_rm.py` (包装层, third_party 零改动):
- `RTX_GROUP_RM=1`: `rm_function` list 分支走 `_rm_batch_group` (组级检测)
- `RTX_SEAL_AUTO_FIX=1`: ABORTED 组所有样本 reward = 权威 `_v1_reward` 重算值,
  `autofix=true` 落盘 (rewards.jsonl 每样本 + seals.jsonl 组记录)
- 单样本降级路径 (非 group-rm): 组完成时覆写 corrected 记录 (持久化数据流修正,
  返回值不可撤回 — 记录为边界)
- 修复: 组级路径避免 `_seal_register`/`_seal_write` 双记录

`scripts/day2_slime_train.sh` / `phase2_run.sh`: `--group-rm` 参数 + 环境变量透传。

## 验证

### 单测 (模拟样本, 5 项全 PASS)
T1 无注入全 SEALED + reward==v1 + 无 autofix / T2 skew 组 ABORTED + 消费侧全 v1
/ T3 AUTO_FIX=0 保留混合 (同 2C) / T4 组级 CAS 幂等 (重复投递 0 新增) /
T5 单样本降级覆写 8 条 corrected

### 真实实验 (1.5B, 4 卡 0-3, 20 步)
| 实验 | 结果 |
|---|---|
| p3a-slime-skew-autofix (skew 窗口 20-39) | succeeded 12min; 注入组 20/20 ABORTED+autofix; **全部 11,392 样本 reward == v1 权威值 (0 不一致)**; 注入组残留非 v1 verifier: 0 |
| p3a-slime-none-autofix (无注入) | succeeded 13.5min; 0 非 SEALED 组 / 0 autofix / 全部 13,600 样本 == v1 |

## 门禁判定

| 门禁 | 结果 | 证据 |
|---|---|---|
| G3A1 训练消费侧 0 混算 | PASS | group-rm 返回值直接进 sample.reward (源码级对应); 注入组 160 样本返回前全部修正为 v1, 记录级 0 不一致 |
| G3A2 干净组零误改 | PASS | 无注入实验 0 ABORTED / 0 autofix / 13,600 样本逐样本 == v1 |
| G3A3 修正开销 <5% | PASS | 端到端吞吐 20,097 tok/s vs 逐样本 seal 15,936 (开销 **-26.1%**, group-rm 批调用减 Python 开销); AUTO_FIX 仅注入组触发 ~1s CPU 异步 |

## 意义

**Day 5 g4 门禁 (0 Invalid Commit) 的真实消费侧闭合**: 训练器消费的 reward
在 slime 架构内首次可保证为权威 verifier 值 — 混算组不再进入训练, 无需改动
训练循环 (原生 --group-rm 参数 + 包装层 RM)。

## 边界与记录

- 单样本路径 (非 group-rm) 返回值不可撤回: 降级为记录层修正 (corrected 流),
  持久化数据始终权威; 训练侧闭合需 group-rm 模式 (原生支持, 无副作用且吞吐更高)
- 预取 buffer 生成组数 > 训练消费组数 (17xx 组 vs 80 训练组) — slime 预取语义,
  不影响检测完整性 (所有生成组均审计)
- crm_crash 注入在组级路径保留 (R1 切点 3B 复用)

## 产物
```
scripts/phase2_seal_rm.py (AUTO_FIX + group-rm)   scripts/phase3_gate3a.py
runs/p3a-slime-{skew,none}-autofix-K8-s42-20260828/ (meta/metrics/rewards/seals/manifests/logs)
runs/PHASE3_GATE3A.json                            runs/PHASE3_3A.md (本报告)
```
