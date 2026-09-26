# F4/F1 试点方案 + 统一 RTO 口径（2026-09-26，待审计）

用户决定（2026-09-26）：
- 同意统一两臂 RTO 口径；
- F4/F1 **先做试点**，暂不跑 20+10 对的完整矩阵；
- R 使用当前 HEAD（lag-1 版本，见 [阶段 2 门控](PERF_GATE2_RESULT_20260926.md)）。

试点只用于工程验证与参数冻结，不计入正式样本。

## 1. 统一 RTO 定义

**RTO = 故障信号时刻 → 受影响工作首次被正确且持久地纳入该臂恢复权威的时刻**（均为宿主 monotonic 时钟）。

| 臂 | 恢复权威 | "持久纳入"时刻 |
|---|---|---|
| A+R | R 的已提交链（token） | 首个使目标行全部进入保留链的 `committed` 事件，含 `via=recovery`。现有实现不变 |
| A | AReaL 原生 recover checkpoint（DCP + recover_info） | 故障后首个在训练中消费了目标行的更新，其 recover checkpoint 的异步写盘被 finalize 成功返回的时刻；如果 recover_info 写得更晚，取两者较晚者 |

**目标行**
- F4/F1：冻结的 `source_row_id=5518` 组（与现有契约一致）；
- F2：被杀更新的 4 个 source row（现有推导）。

**A 端需要补的观测**
- 在两臂共用的观测器（`areal_pilot_hooks.install_hooks`）里，对 `AsyncCallsQueue.schedule_async_request` / `maybe_finalize_async_calls` 做类级包装，发出 `dcp_scheduled`（call_id、pid）和 `dcp_finalized`（call_ids、pid）。
- R 在实例上另有一层包装，调用会穿过类级包装，两臂事件一致。
- 每个 `checkpoint_save_start` 按同一进程内的顺序配对到其后第一个 `dcp_scheduled`。
- 目标行到更新的映射沿用 `finalize_ft1_fault` 中 F2 的方法：`batch_taken` 的批次指纹 → 生成指纹 → source row。

**`finalize_ft1_fault` 的变化**
- A 分支的端点从"最终 checkpoint 上界"改为上表口径，`persistence_time_kind` 两臂都记为 `durable_commit`，`formal_rto_eligible=true`；
- 任一臂缺观测时，记为不可判定并停跑，不回退到上界。

**影响说明**
- lag-1 下 R 的端点晚一步，这是如实计量，不做修正；
- A 的 finalize 可能被推迟到下一次保存时的非阻塞 reap，这同样如实计入。

## 2. 最小流程接入 F1/F4

- `run_ft_minimal.minimal_config`：
  - 接受 `F1`/`F4`，重试次数与现有 FT1 故障配置一致（`retries: 1`）；
  - F1 仍复制 `ft1-f1-target.json`（`run_training_fault` 已支持）。
- 故障钩子沿用 `ft1_fault_hooks.contract`：
  - F4：目标组 8 条回答全部生成后，第 4 次评分执行开始时杀掉评分进程；
  - F1：生成引擎在至少一条回答完成后被杀。
- 未命中切点时记为 `fault_cut_missed`，属技术无效，按冻结规则最多补做一次。
- `run_ft_formal` 的 `perf_gate` 分支改为通用的 `pilot` 类型：非正式样本，场景 ⊆ {F1, F2, F4, no_fault}。

## 3. 试点冻结草案

| 顺位 | 场景 | 对数 | 顺序 |
|---|---|---:|---|
| 1–2 | F4 | 2 | A→R、R→A |
| 3–4 | F1 | 2 | R→A、A→R |

- 种子：`random.Random(20260927)` 从未用过的种子中抽 4 个。
- 每 run 10 次更新，其余训练参数与 2026-09-24 正式冻结相同。
- 预算约 8 GPU·h。
- 试点判据：
  1. 切点命中（`valid_hit`）；
  2. 两臂验收通过，分类可判定；
  3. 两臂 RTO 都能按 §1 测出；
  4. F4 两臂对目标组的"重生成 / 重评分"次数可从证据中计数，作为恢复代价的分解依据。
- 试点**不做**统计检验，也不报告 R 对 A 的节省比例，只报告能否测量以及数值量级，用于决定正式矩阵的规模与参数。

## 4. 需要审计的风险
1. **F4 可能被原生评分重试完全吸收**（严格评分每个 wrapper 最多 2 次尝试）：此时两臂都只重算 1 条评分，RTO 差异趋于零。这属于如实结果，但会影响正式矩阵是否值得跑，试点要能识别出这种情况。
2. **F1 的最小流程接入**：F1 会杀生成引擎，AReaL 需要重启推理服务。10 步内能否恢复并训练到目标行，还要看 900 秒观察窗口是否够用。
3. **A 的 finalize 观测只在主进程内做类级包装**：要确认 DCP 在 fork 出的 writer 中不会走另一条路径，也不会被 R 的实例包装绕过。
4. **测量口径不对称**：R 的端点是 token 写入，A 的端点是 finalize 返回，两者都是本臂恢复所依赖的最后一步持久化，但开销组成不同，报告中要分开列出。

## 5. 流程
本方案 → 独立审计 → 用户批准 → 实现与 CPU 验证 → 试点冻结 → GPU 试点（约 8 GPU·h）→ 报告后再决定正式矩阵。
