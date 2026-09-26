# 性能门控结果（阶段 1 + B，2026-09-26）

冻结：[PERF_GATE_FREEZE_20260926.json](PERF_GATE_FREEZE_20260926.json)（工程样本，不计入正式结果）。两对均 `formal_pair_verified`（该状态名由正式 runner 沿用）。

## 功能
- F2（s763）：R `correct_recovered` 32/32，`native_state_loaded.exact_match=true`；A `safe_discard` 0/32。分类与 2026-09-24 冻结版一致。
- 无故障（s1199）：两臂 `no_fault_verified`。
- 磁盘峰值：R F2 34.7 GB、无故障 20.8 GB，与旧版持平。

## 性能（无故障对，`profile_r_overhead.py`，见 [PERF_GATE_PROFILE_20260926.json](PERF_GATE_PROFILE_20260926.json)）

| 指标 | A | R | R/A |
|---|---:|---:|---:|
| 训练循环 s | 157.6 | 209.2 | **1.33** |
| 总墙钟 s | 489.0 | 510.4 | 1.04 |
| batch_taken→update_prepared（每步中位） | — | 9.6 | 旧版 19.6 |
| async_finalized→committed（每步中位） | — | 1.95 | 旧版 12.7 |

旧版同类数据（p11–p13 中位）：循环 A 211 s / R 507 s = 2.40；总墙钟 1.60。

**判定**：按预先登记的规则，训练循环比 1.33 > 1.15，**进入阶段 2**。

注意：A 的循环本次只有 158 s（旧版约 211 s），主要是每步等批时间从 9.7 s 降到 0.85 s。n=1，属于跨 run 波动，不作为 A 的性能结论。

## 探针（R 进程累计，10 步）

| 段 | 墙钟 s | 说明 |
|---|---:|---|
| rlvr.io（I/O 线程） | 95.3 | 3,328 次，线程 CPU 33.8 s，不在主线程 |
| r.native_snapshot（hasher 线程） | 63.4 | 与主线程并行；线程 CPU 仅 0.6 s，主要在等 D2H/默认流 |
| r.settle_finalize（屏障） | 47.2 | **关键路径**，每步约 4.7 s，等 DCP 写盘 |
| r.settle_join（屏障） | 17.6 | **关键路径**，每步约 1.8 s，等 hasher（快照与 D2H） |
| state.prehash（hasher） | 19.9 | 分块摘要 |
| state.flock_wait | 15.0 | 10,626 次，争用 |
| r.prune（屏障） | 7.8 | **关键路径**，每步约 0.8 s，锁内删除大文件 |
| state.json_parse | 6.5 | 1,644 次（缓存生效，旧版每步约 1,730 次） |

**剩余差距**：R 每步约比 A 多 5 s。
- 屏障约 7.5 s：finalize、join、prune，加上 commit 0.2 s；
- `batch_taken→prepared` 仍有约 9.6 s：训练主线程与并发 rollout 的 I/O 线程、flock 争用，A 同段约 2.4 s。

阶段 2（lag=1 流水提交）能消除屏障里的等待；第二项需要另行处理，例如把 prune 移出锁、降低 rollout I/O 线程上的 CPU。
