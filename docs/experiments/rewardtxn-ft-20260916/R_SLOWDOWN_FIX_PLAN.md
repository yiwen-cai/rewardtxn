# R 臂逐步变慢修复方案 v2（待用户决定后实现）

日期：2026-09-23。状态：**方案，未实现、未运行**。
v1：`R_SLOWDOWN_FIX_PLAN_v1.md`；独立审计：`R_SLOWDOWN_FIX_PLAN_AUDIT.md`（有条件通过）。本版逐条吸收审计 S1–S4、M1–M6，对应关系见 §9。

**动机声明**：本次修改的动机是**可行性**（FT-v1 §6 的 30 步 45 分钟上限、磁盘容量），不是改善 RTO。不根据任何 FT1 RTO 结果在修复变体之间做选择（FT-v1 §6"不得看过收益后调优"）。

## 1. 现象（已测，数字经审计复算）

无故障 smoke（`p3_evidence/ft1-smoke-s401-{a,r}-r1`，同 seed）每步间隔（observer `batch_taken` 相邻差，秒）：

| 步 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| A | 23 | 13 | 4 | 17 | 4 | 6 | 20 | 5 | 8 |
| R | 76 | 84 | 108 | 129 | 149 | 173 | 189 | 210 | 234 |

R 每步增长约 20.3 s。合并 observer 与 R 事件（同一 monotonic 时钟）后的构成：

| 区段 | 随 k 的斜率 | 内容 |
|---|---|---|
| `async_finalized → committed` | 13.2 s/k（截距 38.3 s） | `commit_generation` 的 `_head`（k）＋ `Runtime.save` 提交后的 `_head`（k+1） |
| `train_batch → update_prepared` | 7.9 s/k | `prepare_generation` 的 `_head`（k） |
| `committed → 下一 batch_taken` | 1.6 → 4.5 s | 残余元数据 O(k)（§2.3） |
| `batch_taken → train_batch` | 不随 k，7.6–16 s | **R 独有**（A 约 0.7 s），来源未剖析 |

## 2. 根因

### 2.1 线性项（主因）
`state._head()`（state.py:361）对每个已提交代调用 `_validate_token()` → `_inventory()`，对整个 checkpoint 做 SHA-256。每代 6,918,086,199 B，其中 `native/__0_0.distcp` 一个文件占 99.995%，其余 11 个小文件合计约 350 KB。**单代哈希约 6.6 s**（约 0.8–0.95 GB/s，受 CPU 限制；单文件无法并行）。

每步全量哈希次数：

| 调用点 | 次数 |
|---|---|
| `prepare_generation → _head`（state.py:387） | k |
| `commit_generation → _head`（state.py:527） | k |
| `_complete → _inventory`（state.py:498，新代） | 1 |
| 提交前复核 `_inventory`（state.py:550，新代） | 1 |
| `Runtime.save` 提交后 `_head`（training_adapter.py:224） | k+1 |
| **合计** | **3k+3** |

恢复路径：`select_recovery → _head`（k）＋ `_validate_token(head)`（state.py:597，1）＋ `RetainedLoader.__init__ → _head`（training_replay.py:23，k）；如有候选代提升，另加 `commit_generation` 的 `_head`＋`_complete`＋复核。按第 12 次更新附近注入，修复前恢复约 25 次哈希 ≈ 2.8 分钟，全部计入 R 的 RTO。

### 2.2 常数项
k=0 时 `finalized→committed` 为 38.3 s，其中 3 次单代哈希约 19.8 s，其余约 **18.5 s** 主要是 `native_snapshot`（GPU 状态逐张量哈希，写 `native-state.json`）以及 publish、证据写入和 fsync。

故障 run 另有 observer（`areal_ft1.py:49-51, 96-99`）在**两臂**每次保存时再做一次 `native_snapshot`：A 从约 7 s/步升到约 18.5 s/步，R 每步多约 6–11 s。

### 2.3 残余 O(k)（修复后仍存在，量级小）
- `control.json` 随 attempts/accepted 线性增长：10 步时 470 KB，解析 4.3 ms、编码 5.5 ms。每步有数百次 `_locked` 读、数十次带 fsync 的整文件重写。
- `_head` 仍按 O(k) 读取并哈希每代约 150 KB 的 manifest/intent。

30 步累计约 1–2 分钟。本版不修，只测量，另立优化项。

## 3. 硬约束：磁盘

2026-09-23 现场：`/public` 剩 193 GB（98%），`/` 剩 377 GB（95%）。R 保留全部代时，30 步 ≈ 207 GB/run，一次就可能写满。

## 4. 修复设计

原则：只改 R 专用路径的**在线校验成本**和**权重保留策略**。提交语义、恢复语义、保存频率、A 的执行路径和两臂共用的 observer 都不动。

### L1 热路径不再重哈希历史代
- `_head(owner, control, *, verify_head_content=False)`：
  - 每代照旧校验：`token`↔`manifest`↔`intent` 的哈希链、run_nonce/config、父子链唯一；
  - 每代做文件集合＋大小的 stat 比对（已按 L3 剪枝的文件，按 marker 列表豁免）；
  - 只有 `verify_head_content=True` 时，才对 head 做全量内容哈希。
- `select_recovery`：用 `verify_head_content=True`，并与 state.py:597 的 `_validate_token(head)` 合并，保证**恢复时 head 只哈希 1 次**。候选代提升路径也改为 O(1) 次哈希（提升代 1 次＋复核改 stat）。
- `prepare_generation`、`commit_generation`、`Runtime.save` 提交后、`RetainedLoader.__init__`：只做元数据＋stat。
- 正确性：恢复只加载 head（training_adapter.py:113, 280-289）；head 损坏直接 `StateError`，不回退父代；消费集合只取自 head 的 manifest，受 token 哈希链保护。审计确认 `writer_recovery/replay/rlvr_replay/oracle` 和故障钩子都不读祖先 checkpoint 的内容。
- **披露边界**：不被加载的祖先 checkpoint 发生同大小内容篡改时，在线不再发现，改由离线审计全量哈希覆盖。FT-v1 §5 故障模型不含磁盘损坏；§5.1(5) 规定离线哈希不进入恢复决策，与 L1 相容。

### L2 新代只全量哈希一次
- 删掉 state.py:550 的第二次全量 `_inventory`。改为 stat 身份复核：**重新枚举整棵目录树**，比较名称集合、`is_symlink`、`S_ISREG` 和 `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns, st_mode, st_nlink)`，与首次哈希时记录的值必须完全一致。
- 不引入线程池（单文件无收益，而且会给 fork 奖励 worker 的进程增加线程；见 state.py:164）。
- **披露边界**：内核 5.4/ext4 的时间戳按 jiffy 粗粒度更新，"首次哈希完成后、同一 tick 内的同大小覆写"可能不改 mtime/ctime。现有的逐文件 before/after 检查（state.py:336-343）本来就有同样的弱点。前提是 writer 已 join、有 finalize 收据、pending gate 成立，在此前提下接受这个边界。

### L3 只剪大权重分片，并设 pin
- **剪什么**：只删 `checkpoint/native/*.distcp`。`policy.json`、`native-state.json`、`recover/*`、`native/.metadata`、`common.pt`、`metadata.json` 以及 intent/manifest/token/receipts 全部保留，每代约 350 KB。空间仍省 99.995%，逐代的 policy/step/小文件哈希审计照常可做。
- **剪哪代**：在 owner 锁内，**按 token 派生的链**定位（不用可能陈旧的 `control['head']`），只剪链上位置 ≤ k−2 的已提交代。abandoned 代、无 token 代、候选代一律不剪（midwrite 残留是 `check_training_fault.py:115` 的证据）。
- **pin（永不剪）**：
  1. 每次 `select_recovery` 返回的 generation（故障前最后可恢复状态）；
  2. 每次恢复后的第一次提交（首次恢复 checkpoint）；
  3. 终点 head。

  pin 由 R 自己的 state 记录（`pins.json`，确定性追加），外部 harness 不在运行中动 R 的目录。这样满足 FT-v1 §9 的保留要求。
- **marker**：`pruned.json` 只含确定性内容（manifest_sha256、被删文件列表及其 manifest 中的 size/sha256），不含时间和 epoch。顺序是先写 marker 并 fsync，再删文件。崩溃后续删时读取已有 marker 并核对一致，**不重写**；marker 与期望不一致则 `StateError`。
- **校验**：缺失文件集合必须等于 marker 列表，且该列表必须是 manifest 中 distcp 分片的子集。**没有 marker 却缺文件**的，判为损坏或"不可复验"，绝不当作已剪枝。
- **空间预算**：head＋parent＋写入中的 1 代＋pin（1 次故障 2–3 个）＋abandoned 残留，每个 R 故障 run 峰值约 **35–41 GB**。两个 4 卡槽并行时约 80 GB，外加 A 臂。
- **存储成本披露**：R 保留 2–3 代，A 原生保留 1 份，差额按 FT-v1 §6 计费。pin 单列为证据保留成本，不计入方法成本。

### 不改的部分
- `wait_async_saves()`（约 10 s/步）：R"先持久、再授权消费"的方法语义，保留并计入成本。
- observer 用的 `native_snapshot`（两臂共用）：不动，以免改变 A 的计时。
- `Runtime.prepare` 使用陈旧 `control['head']` 导致的活性问题（在"token 已写、control 未写"窗口崩溃后，永久报 parent mismatch）：是修复前就有的问题，**另立修复**，不混入本批。
- §2.3 的残余 O(k)：只测量，不修。

### 可选 C1（需用户决定，见 §8）：削减 R 的常数项
- C1a：为 R 单独复制一份 `native_snapshot`，把逐张量哈希并行化，输出逐字节一致；observer 仍用原函数。
- C1b：在 `wait_async_saves` 之后，让新代首次 `_inventory` 与 R 的 `native_snapshot` 并发执行。两者都在 hashlib 内释放 GIL；提交前记录 stat 身份，在锁内复核。
- C1c：先剖析 R 独有的 `batch_taken → train_batch`（7.6–16 s），**只剖析不改**，结果交用户决定。

## 5. 预期效果（按故障 run 拟合，审计复算）

修复后每步 = 无哈希常数 C ＋ 1 次新代哈希：

| run | 单代哈希 (s) | C (s) | 修复后每步 (s) | 30 步训练段 (min) |
|---|---|---|---|---|
| smoke s401 R | 6.8 | 49.0 | 55.7 | 27.9 |
| smoke s409 R | 6.9 | 39.7 | 46.5 | 23.3 |
| F4 s417 R | 6.9 | 54.5 | 61.5 | 30.7 |
| F1 s421 R r2 | 6.5 | 57.9 | 64.4 | 32.2 |

加上固定开销（启动/恢复，4.9–10.4 分钟）和残余 O(k)（1–2 分钟），**R 的 30 步故障 run 预计 38–43 分钟，距 45 分钟上限只有约 2–7 分钟余量**。启用 C1 后预计每步再省约 5–10 s（未测）。

超限处理（**在任何正式结果之前冻结**）：若门控试跑中 R 超过 45 分钟，不单方面放宽，只能在以下两项中选：
- (a) 实施 C1 后重测；
- (b) 两臂同时修改上限并重新冻结。

两者都须用户批准。

## 6. 验证计划

### 6.1 CPU（隔离容器 CPU profile，无 GPU）
1. **预期会改变的既有断言**，先列出并附理由，更新对应合同文档；旧证据的历史判定不变：

   | 位置 | 现状 | 更新后 |
   |---|---|---|
   | `p2_schedule_driver.py:1628-1644`（`F3.b01.i09` 等 `*.i09`） | 祖先同大小篡改 → 恢复拒绝 | 改为篡改 **head** → 拒绝；另加"祖先同大小篡改：在线不报、离线报" |
   | `test_p2_second_fault_probe.py:114-138` | 逐代读 checkpoint 字节 | 3 代链不触发剪枝（k−2 仅第 0 代，且可能是 pin）；按实际 pin 调整期望 |
   | `check_ft1_chain.py:87-94, 114` | 每代文件集合＝manifest＋全量哈希；每代 policy step | 已剪分片按 marker 豁免；policy/step 照常逐代检查 |
   | `check_ft1_chain.py:96, 133, 138` | 写死 10 代/320 样本 | 按 run 配置参数化（30 步） |
   | `check_training_fault.py:92-96`、`check_training_integration.py:54-58` | 每代全量哈希 | 按 marker 豁免已剪分片 |

   其余既有测试必须**原样通过**：`test_state`（21）、`test_state_fork_lock`、`test_training_replay`、`test_writer_recovery`、P2 state 21 / oracle 25 中未列入上表者、`p2_second_fault_probe`（6）、`test_p2_schedule_driver`。
2. **哈希次数计数**（打桩 `_inventory`，链长 1/5/30）：每步恰好 1 次；恢复恰好 1 次（head）；候选代提升路径为 O(1)。
3. **L2 复核**：在首次哈希与复核之间增删文件、rename 替换（ino 变）、截断后重写、新增 symlink，都应被拒绝。同 tick 同大小覆写记为已知边界，并附测试说明。
4. **L3 剪枝**：
   - 崩溃点：marker 前、marker 后删除前、删除一半，都能正确续删，且不重写 marker；
   - `control.head` 陈旧时仍按 token 链定位；
   - abandoned、候选代、pin 都不被剪；
   - 无 marker 缺文件 → 拒绝或不可复验；
   - 剪后 `select_recovery`、`RetainedLoader` 正常。
5. **旧证据回归**：用新版 `check_ft1_chain` 只读检查 `ft1-f1-s421-r-r2`（10 代完整），结论须与原判定一致。对已人工删除 checkpoint 的旧目录（如 `ft1-smoke-s401-r-r1`），结论须为"不可复验"。
6. **合成基准**：每代 1 个大文件＋11 个小文件；control.json 每步增加 32 个 attempts；使用真实大小的 manifest。测 prepare、commit、save 各段随 k 的斜率，目标：哈希相关斜率约 0，元数据残余 < 0.1 s/k。

### 6.2 GPU 门控（冻结后，属非正式 pilot）
- **1 对 30 步 F2**（A 与 R，同 seed、同 4 卡）。F2 会重启 trainer，覆盖恢复路径；F4 的 R 只有一个 trainer 进程，测不到恢复路径。
- **事先确定的判据**，全部满足才算通过：
  1. R 墙钟 ≤ 45 分钟；
  2. R 每步间隔对 k 的回归斜率 < 1 s/步；
  3. 恢复时全量哈希次数 = 1（由事件计数确认）；
  4. 两臂都完成现有验收链（fault/chain/input/load/finalize）；
  5. R 磁盘峰值 ≤ 45 GB，pin 与 marker 经新版 chain 审计核对一致。
- 任一不满足即停止，交用户决定（§5 超限处理）。
- 记录 `native_snapshot` 段耗时、各段斜率、磁盘峰值（含 abandoned 与 pin）。

## 7. 版本与合规
- 这是 R 的实现修改。按 FT-v1 §6，结束当前版本批次：此前全部 FT1 试跑保留为修复前版本，不与修复后结果合并。
- A 臂也 import `training_adapter`（areal_ft1.py:74），**两臂的源码冻结都会变**，新批次中 A 也要在新冻结下重跑。
- 修改前备份被改文件（`.bak-<时间戳>`）；新 freeze 覆盖修改后的全部哈希；方法说明中披露 L1、L2 的边界和 L3 的保留策略。
- 公平性：改动只涉及 R 专用路径和审计工具；A 的执行路径、observer、保存频率都不变（FT-v1 §5.1(5)、§6）。

## 8. 需用户决定
1. **C1 常数削减**：是否实施 C1a/C1b（不改语义，只削减 R 自身的哈希常数）？C1c 的剖析是否执行？
   - 不做：余量约 2–7 分钟，超时风险较高；
   - 做：多一轮实现与测试。
2. **FT1 预算**：FT-v1 规定 GPU 预检最多 16 次，现有约 14 次。§6.2 的 30 步 F2 门控 1 对（2 次）会达到或超过上限，而且 30 步 run 不在 FT1 原定义内。是否批准，并登记为非正式 pilot？
3. **证据保留修订**：正式矩阵每 run 验收通过后，是否删除大分片（只留小文件、manifest、哈希、日志；失败或非 `correct_recovered` 的 run 以及随机 10% 保留全量）？这与 FT-v1 §9 冲突，只能作为**FT-v1 的正式修订**，两臂同样适用，并在正式结果之前冻结。不修订时，80 run 按 pin 保留约需 1.5–2 TB，现有磁盘放不下。
4. **活性问题**（陈旧 `control['head']`）：是否批准另立一个小修复？建议在同一冻结前完成，但与本方案分开提交和测试。

## 9. v1 → v2 修改对照

| 审计项 | v2 处理 |
|---|---|
| S1 单代 6.6 s、并行无效 | §2.1 更正；L2 删除线程池 |
| S2 估算低估、需故障 run 门控 | §5 按故障 run 重估 38–43 分钟，先写好超限处理；§6.2 改用 30 步 F2 门控并事先定判据 |
| S3 与 §9 冲突 | L3 增加 pin；§8-3 改为 FT-v1 正式修订 |
| S4 工具与测试影响未列全 | §6.1-1 列出预期会改变的断言；只剪 distcp |
| M1 marker 幂等 | 内容确定性，续删不重写 |
| M2 陈旧 head、非提交代 | 按 token 链定位；排除 abandoned/无 token/候选；活性问题另立（§8-4） |
| M3 stat 复核写全＋披露 | L2 完整身份元组＋jiffy 边界 |
| M4 残余 O(k) | §2.3 如实写出；基准改用真实分布 |
| M5 恢复路径计数 | L1 合并 state.py:597；候选代提升改为 O(1) |
| M6 预算与 A 重跑 | §7、§8-2 |
