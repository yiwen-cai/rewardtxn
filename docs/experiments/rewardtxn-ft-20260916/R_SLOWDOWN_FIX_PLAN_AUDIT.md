# R_SLOWDOWN_FIX_PLAN 独立审计

日期：2026-09-23。审计对象：`R_SLOWDOWN_FIX_PLAN.md`（未实现）。方式：只读源码与证据，未运行训练/GPU/容器。只做了两项本机只读 CPU 测量：`control.json` 解析耗时、内存 SHA-256 吞吐。

## 结论：有条件通过

- 根因成立：`_head()` 每次都对全部已提交代重新做全量内容哈希。调用点与次数（3k+3）逐一核对无误，§1 表格数字可复算。
- L1 的正确性论证成立：恢复只加载 head，head 损坏不回退父代，消费权威只在 intent/manifest 里，受 token 哈希链保护；FT-v1 §5 故障模型也不含磁盘损坏。
- 以下几项必须先改方案再实现：
  - L2 的"并行哈希"实际上没有收益：单个文件占 99.99% 字节。
  - §5 的时间估算建立在错误的单代哈希耗时和无故障 smoke 上，低估了 45 分钟超时的风险。
  - L3 与 FT-v1 §9 的证据保留规则冲突；它对审计工具和既有测试的影响也没有列全。
  - §6 称"原有测试全部重跑通过"不成立。

---

## 一、严重问题（实现前必须改方案）

### S1. L2 的并行哈希无效；单代哈希耗时约 6.6 s，不是 13 s

- 证据：`ft1-smoke-s401-r-r1/rewardtxn/state/generations/*/manifest.json` 每代 12 个文件，共 6,918,086,199 B。其中 `native/__0_0.distcp` 一个文件就有 6,917,735,465 B（99.995%），其余 11 个文件合计约 350 KB。
- SHA-256 对单个流无法并行，按文件分线程最多省几毫秒。§4 L2 的"13 s → 3–5 s"和 §5 里的"−9 s"不成立。要让单文件变快，只能改摘要格式（如分块哈希），这与"结果逐字节一致"矛盾，也要同步改离线审计。
- 单代哈希耗时的实际值：
  - `async_finalized→committed` 区段里有两个随 k 增长的 `_head`：`state.py:527` 和 `training_adapter.py:224`。每步多一代，就多两次单代哈希。复算得斜率 13.19 s/k，所以单代约 6.6 s。
  - 下一次 `prepare` 里的 `_head`（`state.py:387`）斜率为 7.86 s/k。
  - 整体每步斜率约 20.3 s ≈ 3 次单代哈希，不需要"和 rollout 等待重叠"来解释。
- 其他佐证：
  - 离线全量审计 `chain-verification-r2.json` 用 69.18 GB 耗时 85.9 s，约 0.8 GB/s。
  - 本机 hashlib 内存吞吐 0.95 GB/s（1–16 MB 块结果相同，受 CPU 限制）。
  - 方案里"单线程实测 13 s/代（0.53 GB/s）"在仓库中找不到出处，很可能把"每 k 两次哈希"的斜率误当成了一次。
- 建议：
  - L2 只保留"去掉第二次全量哈希、改为 stat 身份比较"（细节见 M3）。
  - 删掉线程池：没有收益，还给会 fork 奖励 worker 的进程多引入一组线程（`state.py:164` 的注释说明这里已经处理过 fork 与锁的交互）。
  - 如果还想缩短常数项，可以在 `wait_async_saves` 之后，让首次 `_inventory` 与 `native_snapshot` 并发执行：两者都在 hashlib 里释放 GIL。做法是提交前记录 stat 身份，在锁内复核。

### S2. §5 的估算与 45 分钟风险被低估

对 §1 的复算与归因（observer 与 R 事件都用系统 monotonic 时钟，可以合并时间线）：

- §1 三行数字在 k=0..5 上完全一致。补充 k=6..9 的 finalized→committed：115.4 / 129.0 / 144.7 / 157.9 s。
- 第三行（committed→下一 update_prepared）的构成：
  - committed→下一 batch_taken 只有 1.6–4.5 s；
  - R 独有的 batch_taken→train_batch 为 7.6–16 s（A 约 0.7 s）；
  - 其余是 prepare 里的 `_head`（train_batch→update_prepared：0.5, 11.1, 19.3 … 69.6 s）。
  - 所以这一行不是"与 rollout 等待重叠"。
- 常数项：finalized→committed 截距 38.3 s。k=0 时这一段已含 3 次单代哈希：`_complete`、复核、`Runtime.save` 里的 `_head`。扣除后，`native_snapshot`、publish、证据和 fsync 约 18.5 s，不是方案写的"约 12 s"。§2 称"k=0 时两次哈希"，与它自己表格中 `Runtime.save` 的 k+1 项自相矛盾。

按"每区间含 3j+3 次单代哈希"拟合各 R run，得到修复后每步 = 无哈希常数 C + 1 次新代哈希：

| run | h (s/代) | C (s) | 修复后每步 (s) | 30 步训练段 (min) |
|---|---|---|---|---|
| smoke s401 R | 6.8 | 49.0 | 55.7 | 27.9 |
| smoke s409 R | 6.9 | 39.7 | 46.5 | 23.3 |
| **F4 s417 R** | 6.9 | 54.5 | 61.5 | **30.7** |
| **F1 s421 R r2** | 6.5 | 57.9 | 64.4 | **32.2** |

- 正式矩阵全是故障 run。故障 run 会启用 `install_load_observer`（`areal_ft1.py:96-99`、`:49-51`），两臂每次保存都额外做一次 `native_snapshot`：A 从约 7 s/步升到约 18.5 s/步，R 每步约多 6–11 s。方案 §5 用的是无故障 smoke，没有计入这部分。
- 固定开销（墙钟减去训练段）：R 的 FT1 run 为 4.9–10.4 min（F4 R 10.4，F1 R 7.2）。
- 修复后 R 的 30 步故障 run 预计约 38–43 min，再加上 M4 所述残余 O(k) 项，余量只有约 2–7 min。方案写的"35–40 min"是乐观下限。
- 建议：
  1. 把门控试跑改成故障 run，且至少包含一个会重启 trainer 的 cell（F2）。F4 R 只有一个 trainer 进程（`ft1-f4-s417-r-r1` observer 中只有 1 个 pid），测不到恢复路径的哈希与回滚。
  2. 在任何正式结果之前写明超限时的处理：两臂同时改上限并重新冻结，或者先做下面第 3 点再重测。
  3. 可选的纯 R 常数削减，不改语义：
     - 为 R 单独复制一份 `native_snapshot`，把逐张量哈希并行化，输出逐字节一致。observer 与 A 仍用原函数，以免改动两臂共用的观测成本。
     - 剖析 R 独有、每步 8–16 s 的 batch_taken→train_batch，方案完全没有提到这一段。

### S3. L3 与 FT-v1 §9 的证据保留规则冲突

- FT-v1 §9 要求："每run保留故障前最后可恢复状态、故障现场残留、首次恢复checkpoint及终点……不清理现有历史实验，不覆盖失败目录"。
- 在第 12 次更新附近注入故障后，运行中剪掉 k−2 及更早的代，会删除恢复时加载的那一代（故障前 head）和首次恢复后的提交。§8 第 2 问"验收后删除终点 checkpoint"也与 §9 的"终点"直接冲突。
- 建议：
  - L3 增加 pin 集合：每次 `select_recovery` 返回的 `generation`、每次恢复后的第一次提交、最终 head 一律不剪。pin 由 R 自己的 state 记录，不能由外部 harness 在运行中动 R 的目录。
  - pin 占用的空间单列为"证据保留成本"，不计入方法存储成本。
  - §8 第 2 问必须作为 FT-v1 的正式修订，两臂同样适用，并在正式结果之前冻结；不能只在本方案里决定。

### S4. L1/L3 对审计工具和既有测试的影响没有列全；§6"全部重跑通过"不成立

方案只提到了 `check_ft1_chain.py`。实际受影响的有：

| 位置 | 依赖 | 受谁影响 |
|---|---|---|
| `tests/ft/check_ft1_chain.py:87-94` | 链上每代 checkpoint 文件集合 = manifest，且全量哈希 | L3 |
| `tests/ft/check_ft1_chain.py:114` | 每代 `checkpoint/policy.json` 的 `global_step == step` | L3（manifest 里没有 step_info，marker 替代不了） |
| `tests/ft/check_ft1_chain.py:96,133,138` | 写死 10 代、320 样本 | 30 步 run 本来就要参数化 |
| `tests/ft/check_training_fault.py:92-96` | 每代全量哈希 | L3 |
| `tests/ft/check_training_fault.py:115` | abandoned 代的 midwrite 残留 `checkpoint/native` | L3 不能剪 abandoned 代 |
| `tests/ft/check_training_integration.py:54-58` | 每代全量哈希 | L3 |
| `tests/ft/test_p2_second_fault_probe.py:114-138` | 3 代链，逐代读 checkpoint 字节 | L3（第 0 代会被剪） |
| `tests/ft/p2_schedule_driver.py:1628-1644`（代表性用例 `F3.b01.i09` 等） | target 提交后，对**祖先** bootstrap 的 `model.bin` 做同大小 1 字节篡改，期望 `select_recovery` 拒绝（`state.select_recovery.corrupt`） | **L1 会让这些用例回归**（在线不再发现祖先的同大小篡改） |

- 这些都是已冻结的 P2/P3 合同。方案必须事先列出"预期改变的断言"及理由，并更新对应合同文档。旧证据上的历史判定保持不变。
- 强烈建议把 L3 改为**只剪 `native/*.distcp` 大分片**。保留 `policy.json`、`native-state.json`、`recover/*`、`native/.metadata`、`common.pt`、`metadata.json`（每代约 350 KB）。这样仍能省下 99.995% 的空间，而上表中 policy/step/小文件哈希类检查可以继续逐代执行；marker 列出被删文件及其 manifest 哈希，审计时核对"缺失文件集合 = marker 列表 ⊆ manifest 中的 distcp 分片"。
- 旧证据的处理：`ft1-smoke-s401-r-r1` 等目录的 checkpoint 已被人工删除（generations 下只剩 JSON），只有 `ft1-f1-s421-r-r2` 保留了全部 10 代。新版检查器必须把"无 marker 却缺文件"判为失败或"不可复验"，绝不能当作已剪枝。

---

## 二、中等问题

**M1. L3 marker 与"幂等续删"冲突。** `pruned.json` 含剪枝时间和 owner epoch，而 `_write` 默认 immutable（`state.py:91-96`）。崩溃后续删时如果重写 marker，会触发 `immutable record conflict`。marker 应只含确定性内容（manifest_sha256、待删文件列表及哈希）；续删时读取并复用已有 marker。

**M2. L3 必须按 token 派生的链定位 k−2，并排除非提交代。**
- `control['head']` 在"token 已写、control 未写"的崩溃窗口中会是陈旧值（`state.py:554-560`）。剪枝不能用它定位。
- 同源的既有活性问题：`Runtime.prepare` 用 `control['head']` 当 parent（`training_adapter.py:176-177`），而 `select_recovery` 不刷新该字段。在这个窗口崩溃后，R 会永久报 `parent mismatch`。这是修复前就有的问题，建议另立修复，不混入本批。
- 只剪有 token 且位于链上 k−2 及更早位置的代。abandoned 代、无 token 代、候选代一律不剪：midwrite 残留是 `check_training_fault.py:115` 的证据。
- 方案里"峰值 3 代约 21 GB"没有计入 abandoned 残留和 pin。按 1 次故障估算，约 5–6 代，35–41 GB。

**M3. L2 的 stat 身份比较需要写全，并如实写明边界。**
- 复核时要重新枚举整棵树：名称集合、`is_symlink`、`S_ISREG`，比较 `(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns, st_mode, st_nlink)`；不能只 stat 首次列出的文件。
- "任何写入都会改变 ctime"不准确。本机内核是 5.4，ext4 的时间戳按 jiffy 粗粒度更新，同一 tick 内的同大小覆写可能不改 mtime/ctime。现有逐文件 before/after 检查（`state.py:336-343`）有同样的弱点。
- 被削弱的仅是"首次哈希完成后、同一 tick 内的同大小覆写"。在 writer 已 join、finalize 收据、pending gate 都成立的前提下可以接受，但要在方法说明中披露。

**M4. "线性项消失"说得过头。**
- `control.json` 随 attempts/accepted 线性增长：10 步后 470 KB、448 个 attempts。本机只读测得解析 4.3 ms、编码 5.5 ms。
- 每步有数百次 `_locked` 读：`batch_identity.py:68,80,114`、`training_replay.py:155`、`validate_rows` 每行两次 `_current`，另有数十次带 fsync 的整文件重写（`authorize_attempt`/`accept_result`）。
- `_head` 仍按 O(k) 读取并哈希约 150 KB/代的 manifest 与 intent。
- observer 中 committed→下一 batch_taken 从 1.6 s 升到 4.5 s。残余斜率估计远小于 1 s/步，但 30 步累计约 1–2 min，在 45 分钟余量里不能忽略。合成基准必须用真实大小的 control.json、manifest，以及"单个大文件 + 若干小文件"的分布；不能用"30 代 × 200 MB"的均匀分布。

**M5. 恢复路径的计数与实现细节。**
- `select_recovery` 在 `_head` 之后还会调用 `_validate_token(head)`（`state.py:597`）。如果 `_head(verify_content=True)` 已经哈希过 head，必须合并，否则 head 会被哈希两次。
- 候选代提升路径（`state.py:585-588` → `commit_generation`）还有 `_head`、`_complete`、复核三次，方案 §2 没有计入。
- 修复前的恢复成本应按实际注入点（约第 12 次更新）和约 6.6 s/代估算：2k+1 ≈ 25 次，约 2.8 min，不是"13 分钟"。修复后这一成本确实计入 R 的 RTO，是改善。

**M6. 版本批次与 FT1 预算。**
- FT-v1 FT1 规定 GPU 预检最多 16 次。现有证据中已有 smoke 4 次，F1/F2/F4 相关 run 约 10 次（含技术重跑）。方案新增的 R smoke 1 次和 30 步 F4 1 对会超出上限，需要用户批准，并登记为非正式 pilot。30 步 run 本身也不在 FT1 的定义之内。
- A 臂也会 import `training_adapter`（`areal_ft1.py:74`），所以两臂的源码哈希 freeze 都会变。新批次需要重新跑 A；方案的 30 步对照里有 A，但 10 步 smoke 只有 R。

---

## 三、逐项核查结论

1. **根因**：成立。
   - 调用点：`state.py:387`（k）、`:527`（k）、`:498`（1）、`:550`（1）、`training_adapter.py:223-225`（k+1），合计 3k+3。
   - 恢复路径：`select_recovery`（`:566` 为 k，`:597` 为 1）、`RetainedLoader.__init__`（`training_replay.py:23` 为 k），候选代提升另计（M5）。
   - 被漏掉的 k 相关开销：control.json 与 manifest 的元数据读取（M4）。DrawLoader 快照（`training_replay.py:58-75`）对每条 draw 读一个小 blob，可以忽略；bridge/identity/artifacts 按行读 blob，与 k 无关。
   - 被漏掉的常数开销：R 独有的 batch_taken→train_batch 8–16 s，以及故障 run 中 observer 的 `native_snapshot`（S2）。
2. **L1 的正确性**：成立。
   - 恢复只加载 head：`training_adapter.py:113,280-289` 与 `check_ft1_load.py:43-47`。head 损坏时抛 `StateError`，没有回退（`make_loader` 未捕获）。
   - 消费集合来自 head 的 manifest：`training_replay.py:22-24`、`training_adapter.py:223-225`、`state.py:401-406,435-437`。
   - `writer_recovery.py`、`replay.py`、`rlvr_replay.py`、`oracle.py` 都不读取任何代的 checkpoint 内容。故障钩子只读 **head** 的 `policy.json`（`areal_training_fault.py:44`、`areal_midwrite_fault.py:120`）。
   - 依赖祖先 checkpoint 内容的只有离线审计和测试（见 S4 表）。FT-v1 §5 不含磁盘损坏，§5.1(5) 规定离线哈希不进入恢复决策，L1 与两者都相容。
3. **L2**：stat 身份比较能覆盖"finalize 后仍有 writer 在写"这一意图的主体，前提是按 M3 写全。并行 hashlib 确实会释放 GIL，结果也可以逐字节一致（dict 比较与 `sort_keys` 都与顺序无关），但在本工作负载下没有收益（S1）。
4. **L3**：
   - "先写 marker 再删"的顺序在进程崩溃下是一致的；与 `select_recovery` 的候选判定（要求有 token）和 pending writer 接管（只写当前代）都不冲突。
   - midwrite 和 F2 场景都不会产生可被剪的提交代。
   - 保留 head + parent 足够：当前代码中 parent 没有功能用途，只是余量。但存在 M1、M2、S3、S4 四个问题，审计工具的影响也没有列全。
5. **§4 不改部分与 §5**：
   - 保留 `wait_async_saves` 合理。
   - `native_snapshot` 约 18 s/步，是修复后最大的单项常数。它属于加载校验的仪器化，是否保留、是否只并行化 R 的副本，应交用户决定，不应默认归为"方法逻辑"。
   - §5 的估算见 S2，45 分钟风险高。
6. **公平性与合规**：
   - 改动只涉及 R 专用路径和审计工具，A 的执行路径与 observer 不变；不改保存频率（符合 §6）。L1 的在线边界与 L3 的保留策略需要披露；R 保留 2–3 代而 A 保留 1 份，属于 §6 规定"计费"的存储成本。
   - 需要补上：§9 证据保留（S3）、FT1 预算（M6）。另要写明动机是可行性（45 分钟上限、磁盘），不是 RTO，并且不根据 FT1 的 RTO 在不同变体之间挑选（§6"不得看过收益后调优"）。
   - 版本批次处理（结束旧批次、旧 run 不合并）正确。
7. **验证计划**：不足，见第四节。

---

## 四、对方案的具体修改建议

1. §2/§5：把单代哈希改为约 6.6 s，常数项拆为"约 18.5 s + 3 次哈希"；删去"与 rollout 重叠"的解释。按故障 run 重做估算（约 38–43 min），并写明余量。
2. L2：删除线程池，保留 stat 身份比较（按 M3 写全）。常数优化改用"首次哈希与 `native_snapshot` 并发"或 R 专用的并行张量哈希（S2 第 3 点）；两者都要求输出逐字节一致。
3. L3：
   - 只剪 `native/*.distcp`；marker 内容确定性、可复用（M1）。
   - 按 token 链定位，排除 abandoned、无 token 和候选代（M2）。
   - 增加 pin：恢复加载代、恢复后首个提交、终点（S3）。
   - 空间预算计入 abandoned 残留与 pin。
4. 在"需用户决定"中新增三项：
   - §8 第 2 问改为 FT-v1 的正式修订；
   - `native_snapshot` 的处置；
   - 超出 FT1 预算的试跑。
5. 修改 `select_recovery`，使 head 只被哈希一次（M5）。

## 五、验证计划需要补充的测试

- 列出所有预期会改变的既有断言（S4 表），给出更新后的期望：例如 P2 `*.i09` 用例改为篡改 head，另加"祖先同大小篡改在线不报、离线报"。其余测试必须原样通过。
- 在不同链长下计数全量哈希：每步恰好 1 次；恢复时恰好 1 次（head）；候选代提升路径为 O(1)。
- 在首次哈希与复核之间做增删文件、rename 替换（ino 变化）、截断后重写、新增 symlink，都应被拒绝。同 tick 同大小覆写记为已知边界。
- 剪枝的崩溃点：marker 之前、marker 之后删除之前、删除一半。续删时不重写 marker。control.head 陈旧时仍按 token 链定位。abandoned、候选代不被剪。pin 不被剪。
- 用 `ft1-f1-s421-r-r2`（10 代完整）只读跑新版 `check_ft1_chain`：结果应与原判定一致。对删掉 checkpoint 但没有 marker 的旧目录，应得到"不可复验"。
- 合成基准使用"单个大文件 + 11 个小文件"的分布，以及真实大小的 control.json（每步增加 32 个 attempts）和 manifest，测 prepare、commit、save 各段随 k 的斜率。
- GPU：用 30 步 F2（会重启 trainer）做门控，最好再加 F4；记录每步间隔斜率、`native_snapshot` 段耗时、磁盘峰值（含 abandoned 与 pin）。先写好通过/不通过的判据再跑。
