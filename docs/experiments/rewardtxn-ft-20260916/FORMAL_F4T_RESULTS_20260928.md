# F4′ 正式矩阵结果（2026-09-28）

结果由 `tests/ft/analyze_ft_f4t.py` 从逐对原始证据复算：

```
python3 tests/ft/analyze_ft_f4t.py --freeze docs/experiments/rewardtxn-ft-20260916/FORMAL_F4T_FREEZE_20260927.json \
  --out docs/experiments/rewardtxn-ft-20260916/FORMAL_F4T_RESULTS_20260928.json
```

统计器会核对：每个生效配对使用的冻结清单都属于原冻结及其修订链（A1→A2→A3，逐级哈希相连）；每个补做之前的尝试都是 `stopped_for_review`；每臂的来源核验、安全分类和 GPU 监测都通过。

## 设计（摘自 [冻结](FORMAL_F4T_FREEZE_20260927.md)）

单机 4×H100。对比对象：AReaL 原生恢复（A）与 AReaL＋RewardTxn（R）。模型为 Qwen2.5-0.5B-Instruct，每 run 10 次有效更新，K=8，U=4。故障 F4′：目标行 `2602/task 20` 的第 4 条评分返回、且该组 8 条都已进入评分后，SIGKILL trainer。

主量为：故障信号时已完成、尚未训练的物理评分中，以同一物理执行进入最终保留链的比例。逐对计算 `R 比例 − A 比例`，以配对为独立单位做双侧精确符号检验，阈值 p<0.05，并要求优势方向为 R。

## 主结果

| 指标 | 值 |
|---|---|
| 有效配对 | 10/10 `formal_pair_verified` |
| R − A 保留比例差 | 10 正、0 零、0 负；中位数 **0.974**，范围 0.933–0.986 |
| 双侧精确符号检验 | **p = 0.00195** < 0.05，`primary_claim_eligible = true` |
| R 臂 | 10/10 `correct_recovered`，保留 37/39 至 68/69 |
| A 臂 | 10/10 保留 0；其中 5 次 `safe_discard`、5 次 `safe_stop` |

在本配置下，预先登记的判据成立：当 trainer 在评分完成、尚未训练时被 kill，RewardTxn 能以同一物理执行保留绝大多数已完成的评分；AReaL 原生恢复一条都不保留，但每次都是安全的，没有错误提交。

## 逐对结果

| 对 | seed | 顺序 | A 分类 | A 保留 | R 保留 | 差值 | 故障后 token A / R | 故障后评分 A / R | GPU·h A / R |
|---|---|---|---|---|---|---|---|---|---|
| p01（redo2） | 9023 | A→R | safe_discard | 0/40 | 66/69 | 0.957 | 102,473 / 65,965 | 320 / 190 | 0.842 / 0.886 |
| p02 | 4602 | R→A | safe_stop | 0/43 | 37/39 | 0.949 | 0 / 65,004 | 0 / 187 | 0.534 / 0.880 |
| p03 | 526 | A→R | safe_discard | 0/43 | 42/43 | 0.977 | 100,713 / 65,253 | 320 / 182 | 0.840 / 0.854 |
| p04 | 3515 | R→A | safe_stop | 0/37 | 42/44 | 0.955 | 0 / 63,358 | 0 / 182 | 0.533 / 0.880 |
| p05 | 1556 | A→R | safe_stop | 0/39 | 68/69 | 0.986 | 0 / 65,994 | 0 / 188 | 0.537 / 0.877 |
| p06（redo1） | 7299 | R→A | safe_stop | 0/37 | 41/42 | 0.976 | 0 / 63,821 | 0 / 183 | 0.547 / 0.839 |
| p07（redo1，全量保留） | 5778 | A→R | safe_stop | 0/41 | 44/45 | 0.978 | 0 / 62,607 | 0 / 180 | 0.543 / 0.886 |
| p08 | 3216 | R→A | safe_discard | 0/47 | 68/70 | 0.971 | 102,720 / 65,792 | 320 / 188 | 0.880 / 0.933 |
| p09 | 5509 | A→R | safe_discard | 0/45 | 42/45 | 0.933 | 101,003 / 63,307 | 320 / 182 | 0.875 / 0.872 |
| p10 | 8400 | R→A | safe_discard | 0/44 | 51/52 | 0.981 | 98,662 / 60,837 | 320 / 173 | 0.882 / 0.855 |

## 次要观察（描述性，不做统计推断）

- **重复计算**：A 臂在 `safe_discard` 时，故障后要重新生成约 9.9–10.3 万 token、重新评分 320 次；R 为 6.1–6.6 万 token、173–190 次评分，约少 35% 的 token 和 42% 的评分。
- **A 的 `safe_stop`（10 次中 5 次）**：kill 落在 AReaL `recover_handler` 写恢复检查点期间，这一时刻主线程的最后一个事件是 `recover_handler_dump_start`。AReaL 先写完 `step_info.json` 等描述文件，DCP 分片还没写完。重启时 `recover_handler.load` 报 `EOFError`，重试后仍失败，故障后成功更新 0 次。这说明 AReaL 的恢复检查点不是原子写入。这是安全停止（没有用损坏状态继续训练），但会丢掉整个 run 的后续进度。在本 F4′ 切点下，这种情况约占一半。
- **R**：10 次都经新进程独立加载最终状态、验证保留链后通过，没有 `invalid_commit`，也没有 `safe_stop`。
- **墙钟**：R 臂每 run 755–840 s；A 臂在 `safe_discard` 时 756–794 s，在 `safe_stop` 时约 480–493 s（提前停止）。RTO 在 A 丢弃目标工作时无定义，按冻结规则只作描述，不计算节省比例。

## 修订与技术无效

| 修订 | 内容 | 文档 |
|---|---|---|
| 补做规则 | 外部原因造成的技术无效不占用补做次数 | [FORMAL_F4T_REDO_AMENDMENT_20260927.md](FORMAL_F4T_REDO_AMENDMENT_20260927.md) |
| A1 | kill 落在 train_batch 与 optimizer_end 之间时，chain/input 验收放行 | [FORMAL_F4T_CHAIN_AMENDMENT_20260927.md](FORMAL_F4T_CHAIN_AMENDMENT_20260927.md) |
| A2 | 原生恢复无法继续的有效命中记为 `safe_stop`（保留 0） | [FORMAL_F4T_SAFE_STOP_AMENDMENT_20260927.md](FORMAL_F4T_SAFE_STOP_AMENDMENT_20260927.md) |
| A3 | GPU 监测器记住自有 PID，修复误判 | [FORMAL_F4T_MONITOR_AMENDMENT_20260928.md](FORMAL_F4T_MONITOR_AMENDMENT_20260928.md) |

每次修订都没有改动 seed、顺序、目标、主量或统计方法。A1、A2 都是在发现某个故障时机后，放宽验收或补充分类；两者对两臂适用同一规则，而且都不是根据某一臂的收益才决定的。

技术无效尝试共 4 次，原件全部保留：

| 尝试 | 失败臂 | 原因 |
|---|---|---|
| p01 原始 | R | 宿主 runner 随 Codex 会话被杀（host lease expired） |
| p01 redo1 | A | 真实外部 GPU 进程（其他用户的评测） |
| p06 原始 | A | 监测器误判（被 kill 的自有 trainer），见 A3 |
| p07 原始 | A | 同上 |

## 资源

- 有效配对：15.77 GPU·h；含全部技术无效尝试共 18.70 GPU·h，上限 25 GPU·h。
- 通过的非预留 run 已删除大分片；p07（预留）的两臂以及所有 `safe_stop` 臂保留全量证据。

## 范围与限制

- 结论只适用于这一配置：10 次更新、0.5B 模型、单机、F4′ 切点。不能外推到长训练、大模型或其他故障类型。
- A 臂的一半结果是 `safe_stop`。如果论文要讨论"同步保存时被 kill"这一时机，应把它作为 AReaL 恢复路径的独立发现单独呈现，不要与 R 的保留优势混在一起。
- 2026-09-24 的 F2／无故障结果属于旧实现版本，单独报告，不合并。F1 后移试点只作为工程观察。
