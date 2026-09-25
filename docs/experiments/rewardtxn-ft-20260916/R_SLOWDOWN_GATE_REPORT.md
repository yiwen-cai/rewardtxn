# R 臂修复后 30 步 F2 门控：**不通过**

日期：2026-09-23。非正式 pilot（`formal_sample=false`），不进入任何统计分母。
证据：`p3_evidence/ft1-gate-f2-s431-{r,a}-r1`、`p3_evidence/ft1-gate-f2-s431-disk.jsonl`；判定：`r-slowdown-gate-check.json`（`tests/ft/check_ft1_gate.py`）。
配置：seed 431，顺序 R→A，同 4 卡（GPU 4–7），30 步，F2 在第 12 次成功 optimizer 更新后、保存前 SIGKILL trainer，原生 launcher 重试 1 次。判据在运行前写入 `ft1-case.json`。

## 判据结果

| 判据（运行前冻结） | 结果 | 值 |
|---|---|---|
| R 墙钟 ≤ 2700 s | **不通过** | 2739.5 s；controller 在 2700 s 记 `method_observation: timeout`，此时 30 次提交已完成，但未到收尾（`final-native-state.json` 未写出） |
| R 每步间隔斜率 < 1 s/步 | **不通过** | 1.57 s/步（同进程内相邻提交间隔的最小二乘） |
| 恢复时全量哈希代数 = 1 | 通过 | `recovery_selected`：只哈希被加载的 head，耗时 8.0 s |
| 两臂验收写出 FV | **不通过** | A：`functional_verification_written`（分类 `safe_discard`）；R：`blocked_missing_final_native_state`（被 timeout 截断） |
| R 磁盘峰值 ≤ 45 GiB | 通过 | 34.7 GB（32.3 GiB） |

按方案 v2 §5 与 §6.2，任一不满足即停止，交用户决定；**未放宽上限，也未重跑**。

## 与修复前对比

| | 修复前（FT1 10 步试跑） | 本次（30 步） |
|---|---|---|
| R 每步增长 | 约 20 s/步 | 1.57 s/步 |
| 首步间隔 | 76 s | 54 s |
| 恢复额外哈希 | 约 2k+1 代 | 1 代（8 s） |
| 磁盘 | 每代 6.9 GB 全部保留 | 峰值 34.7 GB，旧代大分片已剪，pin 保留 |
| A 墙钟 | — | 1149 s（19.2 min） |

## 残余增长定位（R 自身事件，逐段）

| 段 | 趋势 |
|---|---|
| prepare→optimizer | 恒定约 1.1 s |
| optimizer→async_scheduled | 恒定约 14.5 s |
| scheduled→finalized（等 DCP） | 恒定约 10 s |
| finalized→committed（哈希＋快照） | 恒定约 13.5 s |
| **committed→下一 update_prepared** | **13.9 → 39 s，约 +1.25 s/步** |

增长全部集中在 committed→update_prepared，其主体是 R 独有的 `batch_taken→train_batch`（observer，7 → 28 s）。这就是审计 S2 标出、方案 v2 C1c 计划剖析的段落。修复前它被每步 20 s 的哈希增长掩盖；审计 M4 估计的元数据残余"远小于 1 s/步"偏低。

最可能的来源（未剖析，需验证）：批次身份桥与 RetainedLoader 在每步、每行上的 `_locked` 读取和整文件重写。`control.json` 随 attempts/accepted 线性增长，每步有数百次读、数十次带 fsync 的重写（`batch_identity.py:68,80,114`、`training_replay.py:155`、`validate_rows` 每行两次 `_current`、`authorize_attempt`/`accept_result`）。

## 其他观察
- 恢复间隙（故障前最后一次提交 → 恢复后第一次提交）282.7 s，含进程重启和 sglang 重连；不是正式 RTO。
- A 臂 F2 在第 12 步的分类为 `safe_discard`，与 FT1 的 F2-A r4 一致。
- 两个 pin（`recovery_loaded`、`first_commit_after_recovery`）按设计写出；剪枝事件正常。

## 需要用户决定
1. 是否批准执行 C1c 剖析，并据此做第二轮 R 实现修复（针对 committed→prepare 的线性项）。剖析可以先在 CPU 上用真实大小的 `control.json` 重放完成，不占 GPU。
2. 修复后是否再跑一次 30 步 F2 门控。FT1 预算需要再次批准。
