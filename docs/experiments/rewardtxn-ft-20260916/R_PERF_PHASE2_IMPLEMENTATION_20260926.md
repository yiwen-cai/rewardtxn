# R 性能阶段 2 实现与 CPU 验证（2026-09-26）

依据 [阶段 2 方案 v1.1](R_PERF_PHASE2_PLAN_20260926.md)（[审计](R_PERF_PHASE2_PLAN_AUDIT.md)有条件通过，用户批准）。修改前备份为 `*.bak-20260926-phase2`。

## 改动
- **state**
  - 新增 `_unresolved`、`_pending_parent`、`_parent_ok`。
  - `prepare_generation`：允许一个本 execution 的未决前驱 P，其 parent 可为 head 或指向 head 的 pending 形式；新 intent 的 parent 须为 `{generation: P, intent_sha256: sha(P 的 intent.json 字节)}`；谱系、consumed、逻辑更新去重都相对 P；已有两个未决代时拒绝。
  - `commit_generation`：按 `_parent_ok` 对两种 parent 形式做 CAS；token 的 parent 仍取 head。
  - `select_recovery`：至多两个未决代，且必须构成链；按顺序逐代提升；某代失败则它和后续代都写 abandoned；返回值新增 `promoted`。
- **运行时**
  - B2 在 `save` 开头执行：finalize → join hasher → commit。
  - `save` 时提前落盘 optimizer evidence 与 `policy.json`；finalize 包装在 finalize 正常返回后，按 schedule 时登记的 call_id→代 写 finalize evidence 与 `writer_gate(False)`。
  - 快照：主线程按值规范化 RNG 与全部非张量，张量在 hasher 线程中哈希并发布 `native-state.json`。B1 在 optimizer 包装开头等待发布完成，失败时由主线程补算。
  - `loader.consumed` 只增不减；prune 由后台单线程执行（删除仍在锁内）；`train` 返回前和 `close` 时都会 join。
  - 恢复进程对 `promoted` 中的每一代写 `committed` 事件，带 `via='recovery'`。
- **验收工具**
  - `offline_generation.parent_matches` 同时支持两种 parent 形式；`check_ft1_chain`、`check_training_fault`、`check_training_integration` 改用它。
  - `check_training_fault` 的单 pid 断言对 `via='recovery'` 的提交放宽。
  - `finalize_ft1_fault` 未改：端点仍取"首个使目标行进入保留链的 committed 事件"。在 lag-1 下这个时点会晚一步，属于如实计量。
- **测试**
  - `test_training_hooks` 的替身 runtime 增加 `join_snapshot`，并补丁 `train`；
  - `test_deferred_commit` 改写为 11 项 lag-1 测试；
  - `test_state_perf` 新增 pending parent、恢复组合与离线 parent 校验。

## CPU 验证
| 检查 | 结果 |
|---|---|
| 宿主 `test_state*` | 全部通过（62+ 项，含新增） |
| 容器 `test_deferred_commit` | 11/11：lag-1 顺序与单 writer 槽；集合通信与捕获只在主线程；F2 式崩溃提升第 0 代、放弃第 1 代；快照未发布时崩溃则放弃；finalize 之前崩溃则放弃；finalize 失败不写 evidence；RNG 与 scheduler 按值捕获；B1 补算；writer 挂起时不死锁；后台 prune 与 prepare 并发 |
| 容器全量回归，与 `9825fcd` 对照 | 仅 `test_training_hooks` 因安装契约变化需要更新替身，更新后通过；其余模块结果一致 |
| `check_ft1_chain` 在保留的旧格式正式证据 p06 两臂上回归 | 均 `verified` |

**未覆盖**：新格式证据上的端到端验收工具。需要 GPU run，由门控的 F2 对完成。

## 门控
冻结 `PERF_GATE2_FREEZE_20260926.json`：无故障 2 对（s9225、s6707）+ F2 1 对（s2504）。判据见冻结中的 `gate_rule`。
