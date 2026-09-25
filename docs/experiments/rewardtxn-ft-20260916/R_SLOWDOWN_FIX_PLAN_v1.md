# R 臂逐步变慢修复方案（草案，待审计）

日期：2026-09-23。状态：**方案，未实现、未运行**。适用：`scripts/ft/state.py`、`scripts/ft/training_adapter.py`、`scripts/ft/training_replay.py`、`tests/ft/check_ft1_chain.py`。

## 1. 现象（已测）

证据：`p3_evidence/ft1-smoke-s401-{a,r}-r1`，无故障 10 步，同 seed。

每步间隔（`observer-pilot` 的 `batch_taken` 相邻差，秒）：

| 步 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| A | 23 | 13 | 4 | 17 | 4 | 6 | 20 | 5 | 8 |
| R | 76 | 84 | 108 | 129 | 149 | 173 | 189 | 210 | 234 |

R 自身事件（`rewardtxn/events.jsonl`，monotonic，秒），k = 之前已提交代数：

| 区段 | k=0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| `async_scheduled → async_finalized`（等待 DCP 完成） | 11.8 | 10.5 | 10.5 | 10.4 | 9.4 | 9.1 |
| `async_finalized → committed` | 38.0 | 50.4 | 65.5 | 79.3 | 92.0 | 104.3 |
| `committed → 下一 update_prepared`（与 rollout 等待重叠） | 20.3 | 29.8 | 37.5 | 45.3 | 57.1 | 62.4 |

斜率约 +21.5 s/步；外推 30 步 R run ≈ 3 h 墙钟，超过方案 §6 的 45 分钟上限，正式矩阵 R 臂将全部判为 timeout。

## 2. 根因（源码）

`state._head()`（state.py:361）对**每个已提交 generation** 调用 `_validate_token()`，后者调用 `_inventory(checkpoint)`，对整个 checkpoint（约 6.9 GB/代，120 文件/10 代）逐文件 SHA-256 并 fsync。单线程实测约 13 s/代（≈0.53 GB/s）。

每个训练步里的调用：

| 调用点 | 全量哈希次数 |
|---|---|
| `prepare_generation` → `_head` | k |
| `commit_generation` → `_head` | k |
| `commit_generation` → `_complete` → `_inventory`（新代） | 1 |
| `commit_generation` 同步后再次 `_inventory` 复核（新代） | 1 |
| `Runtime.save` 提交后 `_head`（training_adapter.py:224） | k+1 |

合计约 3k+3 次 6.9 GB 哈希，随步数线性增长，整 run 呈平方增长。部分时间和 rollout 等待重叠，所以观测斜率低于理论值。

恢复路径同样受影响：`select_recovery → _head`（k）+ `_validate_token(head)`（1）+ `RetainedLoader.__init__ → _head`（k）。30 步后恢复要额外哈希约 61×6.9 GB，按 13 s/代估约 13 分钟，直接计入 R 的 RTO。

常数部分（k=0 时 finalized→committed 38 s）：两次新代全量哈希约 26 s，外加 `native_snapshot`（GPU 状态逐张量哈希，写 `native-state.json`，供 R 加载后精确比对）约 12 s。

## 3. 第二个硬约束：磁盘

2026-09-23 现场：`/public` 剩 193G（98%），`/` 剩 377G（95%）。R 保留全部代 checkpoint，30 步 ≈ 207 GB/run。不改保留策略，**一次 30 步 R run 就可能写满 `/public`**。

## 4. 修复设计

原则：只改 R 的**在线完整性检查成本与保留策略**，不改提交语义、恢复语义和配对配置。

### L1（必做）：热路径不再重哈希历史代

- `_head(owner, control, *, verify_content=False)`：
  - 每代仍校验 `token.json`↔`manifest.json`↔`intent.json` 的哈希链、run_nonce/config、父子链唯一性（全部是小 JSON，成本可忽略）。
  - 未剪枝的代：做**文件集合 + 大小**的 stat 校验（与 manifest `files` 比），不读内容。
  - 只在 `verify_content=True` 时对 head 做全量内容哈希。
- `select_recovery()`：对**即将加载的 head** 做全量内容哈希（`verify_content=True`）；祖先只走元数据 + stat。
- `RetainedLoader.__init__`、`prepare_generation`、`commit_generation`、`Runtime.save` 提交后：只走元数据 + stat。
- 正确性论证：恢复**只加载 head**；当前实现中 head 任何损坏都会直接 `StateError`，不会回退到父代，所以祖先 checkpoint 的**内容**从未参与恢复。消费权威（consumed/pending/drawn）在 manifest/intent 中，受 token 哈希链保护，L1 不削弱这一点。
- 被削弱的只有一件事：**不被加载的祖先 checkpoint**，其同大小内容篡改不再在线发现。改由离线审计（`check_ft1_chain.py` 全量哈希）覆盖，并在方法说明中披露。

### L2（必做）：新代只哈希一次，且并行

- `commit_generation` 的第二次 `_inventory` 复核改为 stat 身份比较：`(size, mtime_ns, ctime_ns, ino)` 在第一次哈希时记录（同时已有 before/after fstat 检查）；任何写入都会改变 ctime。
- `_inventory` 按文件并行哈希（ThreadPoolExecutor；hashlib 对大缓冲释放 GIL），线程数上限 8（训练容器 `--cpus=32`）。结果字典与现有格式逐字节一致。

### L3（必做，受磁盘约束）：运行中剪枝旧代权重

- 提交第 k 代后，在 owner 锁内对 k−2 及更早的代：
  1. 先原子写 `pruned.json`（含 manifest_sha256、剪枝时间、owner epoch），fsync；
  2. 再删除该代 `checkpoint/`；intent/manifest/token/receipts **全部保留**。
- 中途崩溃：有 marker 未删完 → 下次调用幂等续删；无 marker → 文件完整。不存在"文件缺一半且无 marker"的状态。
- `_head`：对有 `pruned.json` 的代跳过 stat；head 与 parent 不允许被剪枝（被剪则 `StateError`）。
- 保留 head + parent（A 臂原生只保留 1 份），峰值约 3 代 ≈ 21 GB/R run。
- `check_ft1_chain.py`：已剪枝代验证 marker↔manifest 哈希和消费链；内容哈希只做保留代。旧证据（未剪枝）仍按原逻辑。

### 不改的部分（明确排除）

- `wait_async_saves()` 让 DCP 在提交前完成（约 10 s/步）：这是 R "先持久再授权消费" 的方法语义，属真实成本，保留并照常计入。
- `native_snapshot` 写 `native-state.json`：R 加载后的精确比对依赖它，属方法逻辑，保留。
- 观测器（`install_load_observer`）两臂相同，不动。

## 5. 预期效果（估计，需实测确认）

- 线性项消失：每步额外开销与 k 无关。
- 常数项：去掉第二次哈希（−13 s），首次哈希并行化（13 s → 预计 3–5 s）。
- R 每步 ≈ 76 − 13 − 9 ≈ 55 s；30 步 ≈ 28 分钟 + 启动/恢复，预计 35–40 分钟，落在 45 分钟上限内但余量不大。
- R 恢复额外哈希从约 61 代降到 1 代（并行后约数秒）。

若实测 30 步 R 仍超过 45 分钟，**不在看过比较结果后放宽上限**；先停下交用户决定（例如在两臂同时改上限并重新冻结）。

## 6. 验证计划

CPU（隔离容器 CPU profile，无 GPU）：
1. 原有测试全部重跑：`test_state`（21）、`test_state_fork_lock`、`test_training_replay`、`test_writer_recovery`、P2 state 21 / oracle 25、`p2_second_fault_probe`（6）、`test_p2_schedule_driver`。
2. 新增：
   - 每步全量哈希次数恒为 1，与链长无关（打桩计数 `_inventory`，链长 1/5/30）；
   - 多代链中 head 内容损坏 → `select_recovery` 拒绝；
   - 祖先缺文件/大小变化 → 拒绝；祖先同大小内容篡改 → 在线不报（记录为已知边界），离线 `check_ft1_chain` 报错；
   - 剪枝：marker 先于删除、崩溃在两者之间可恢复、head/parent 不可剪、剪后 `select_recovery`/`RetainedLoader` 正常；
   - 并行 `_inventory` 与串行结果逐字节相同。
3. 合成基准：30 代 × 200 MB，记录 commit 耗时随 k 的斜率（目标 < 0.1 s/代）。

GPU（冻结后，已在简化方案中计划）：
4. 1 次 R 无故障 10 步 smoke：每步间隔斜率 < 1 s/步；320 输入重评分、chain、load 验收照常通过。
5. 1 对 30 步 F4 试跑：R 墙钟 ≤ 45 分钟，磁盘峰值与预估相符；不进正式分母。

## 7. 版本与合规

- 这是 R 实现修改：按方案 §6"实现错误需要修复时结束当前版本批次"，此前所有 FT1 试跑保留为修复前版本，不与修复后结果合并。
- 修改前备份被改文件（`.bak-<时间戳>`），新 freeze 覆盖修改后哈希。
- 方法描述中披露 L1 的在线检查边界与 L3 的保留策略。

## 8. 需用户决定

1. L3 剪枝保留 head + parent（推荐）还是只保留 head。
2. 正式矩阵每 run 验收通过后，是否删除终点 checkpoint（只留 manifest/哈希/日志；失败或非 `correct_recovered` 的 run 与随机 10% 保留全量）。按当前磁盘，80 run 全量保留约需 0.8–1 TB，不可行。
