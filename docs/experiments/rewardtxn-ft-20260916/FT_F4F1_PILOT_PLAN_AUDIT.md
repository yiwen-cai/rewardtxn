# F4/F1 试点方案与统一 RTO 口径 独立审计（2026-09-26）

**审计对象**：[FT_F4F1_PILOT_PLAN_20260926.md](FT_F4F1_PILOT_PLAN_20260926.md)，下称"方案"。

**对照基准**：分支 HEAD `2241492`（阶段 2 已实现），以及 `third_party/areal` 的工作树。

**审计方式**：只读，未改代码或已有文件，未使用 GPU。

## 结论：有条件通过

试点本身值得做，规模对"能否测量"这个目的也够用。

但方案中 **A 端 RTO 的定义对 A 偏宽**：AReaL 的 recover checkpoint 在原地覆盖写入，所谓"持久"只存在于一个瞬间（见 A-1）。另外，**F4 几乎必然被原生重试吸收**，训练器不会重启，所以 F4 测到的是正常路径的持久化延迟，而不是恢复能力（见 F-1）。

这两点必须写进口径和判据，否则试点结果会被误读为正式矩阵的依据。

---

## 1. A 端 RTO 端点（问题 1）

### A-1 AReaL recover checkpoint 的真实持久完成点

**写入顺序**（同一次 `dump`，主线程执行）：

1. `_save_checkpoint` → `engine.save`，进入 `save_checkpoint`（`recover.py:292-299, 416-441`）。其中又依次执行：
   - 先对**上一次**异步写盘做非阻塞回收 `_reap_finished_async_saves()`（`checkpointer.py:607-609`）。回收时才执行 finalize，并写出上一代的 `.metadata`；
   - 再调用 `generate_state_dict` 并调度本次写盘（`checkpointer.py:612-640`），fork 出一个 writer 进程。
2. **随后立即**写 `RecoverInfo.dump`（`recover.py:301-316`，内容为 `step_info.json` 等，普通 `open/write`，没有 fsync）。此时本次 DCP 还在写。

**路径固定，每步覆盖。**
- 检查点路径由 `Saver.get_recover_checkpoint_path` 给出，固定为 `.../recover_checkpoint`（`saver.py:79-90`）；
- `recover_info_path` 也是固定路径。
- 旧证据也能印证：A 的 `checkpoint_save_returned` 对 `meta.path` 做哈希时，能看到上一代完整的 6.9 GB 文件（阶段 1 审计 §4）。

**结论：A 在第 u 次更新后的"完整可恢复状态"只在一个很短的时刻成立。**
- 这个时刻从 finalize(u) 返回开始（发生在 save(u+1) 内部的回收，或者在收尾时的 `wait_async_saves`）。
- 同一个调用里，紧接着就会调度 u+1 的写盘，覆盖同一批文件，并把 `step_info` 改写为 u+1。
- 从此到 finalize(u+1) 之间，磁盘上的 `step_info` 指向 u+1，DCP 却是部分写入的 u+1 或 u 与 u+1 的混合。这个窗口内发生崩溃，A 无法恢复。FT1 报告已经记录了同样的情况（`FT1_RUN_REPORT.md:47`），F2 钩子因此要在注入前先排空写盘（`ft1_fault_hooks.py:119-121`）。

**"finalize(u) 与 recover_info 取较晚者"可以测到，但语义对 A 有利：**
- recover_info(u) 早于 finalize(u) 写入，所以"较晚者"总是 finalize(u)；
- 但 A 的持久化并不单调，这个状态会在同一次调用内被破坏。

**必须：**
- 在报告中把 A 的端点标注为"首次瞬时完整"（下界）；
- 同时记录 A 的"可恢复窗口"，即 finalize(u) 到下一次调度写盘之间的时长，以及故障注入时 A 是否正处于覆盖写窗口内（有待决写盘，且 `step_info` 大于已 finalize 的 step）；
- 故障恰好落在覆盖写窗口内的 A 样本，应单独分类，不能与"正常可恢复"混在一起算 RTO。

另外：R 的 token、manifest、control 都做了 fsync；A 的 recover_info 和 `.metadata` 不保证 fsync。在当前的故障模型（进程被杀、机器不断电）下两者等价，但口径中应写明"持久＝对进程崩溃持久"。

### A-2 类级包装能否在两臂一致地观测

**能一致观测，前提是安装顺序正确。**
- R 在 `attach` 时把**当时**解析到的绑定方法保存下来，再做实例包装（`training_adapter.py` 中 `attach` 的 `schedule, finalize = queue.schedule_async_request, ...`）；
- 所以类级包装必须在 `PPOTrainer` 构造之前安装，也就是在 `attach` 之前。`install_hooks()` 在 `areal_ft1.py:289` 调用，早于构造 trainer，满足这一点。
- 实现时应加断言：`attach` 时取到的方法确实是类包装。
- `areal_midwrite_fault.py:78-91` 的额外实例包装同样会穿过类包装。

**fork 出的 writer 不会调用这两个方法，没有另一条路径。**
- finalize 的函数（包括写 `.metadata`）在主进程的 `maybe_finalize_async_calls` 里执行；
- writer 子进程只写 `.distcp`。

**需要注意：**
- `AsyncCallsQueue.close()` 内部也会调用 `maybe_finalize_async_calls`（async_utils 0.18 版 `:710-718`），会产生一次 `dcp_finalized`，这是预期行为。
- `maybe_finalize` 返回空列表的调用也会被包装，应只在列表非空时发事件。
- **类包装修改的是两臂共用的观测器**，会改变 A 路径的代码。按已有不变量，这需要新冻结，并把它列为试点改动。

### A-3 用顺序把 `checkpoint_save_start` 配对到 `dcp_scheduled` 不够可靠

- 在单主线程下，顺序配对通常是对的：`save` 调用 → 回收（可能产生 `dcp_finalized`）→ 调度。
- 但 R 的 B2 会在 `original_dump` 之前调用 `wait_async_saves`；F2 钩子、midwrite 探针也会插入额外的 finalize。事件交错时，顺序配对容易出错。
- **建议显式绑定**：`MegatronPPOActor.save` 的包装先把 `meta.path` 和一个 save_id 放进线程局部变量；类包装的 `schedule` 读取它，写进 `dcp_scheduled`；再维护 call_id → save_id 的对应关系，用于配对 `dcp_finalized`。

### A-4 目标行到更新的映射

- F4/F1 的目标是冻结的 5518 组，不需要从被杀的更新反推目标，只需要找出"故障后首个消费了 5518 组的更新"。
- 沿用"批次指纹 → 生成指纹 → source row"的方法是可行的：
  - 每次生成（包括重新生成）都有 `generation_complete` 事件，里面带 `source_row_id`（`areal_ft1.py:204-210`，由各进程写到自己的 pid 文件）；
  - 重新生成后指纹会变，但新指纹对应新事件，仍能映射回 5518；
  - R 在恢复时复用旧 receipt、不重新生成，此时指纹对应旧进程的事件，所以**必须读取所有 pid 的观测文件**。
- `finalize_ft1_fault.py:26-29` 断言每个指纹只对应一个 source row，这一点保留；另外应断言"5518 组的 8 条都能映射到"。
- 对 R 可以直接用 manifest 里 group 的 `prompt.source_row_id`（`finalize_ft1_fault.py:39-40`），更直接。对 A 必须走指纹。

---

## 2. R 端端点在 lag-1 下的含义（问题 2）

- 正常路径下，第 u 代的 `committed` 事件在 **save(u+1) 开头的 B2** 发出。A 的 finalize(u) 也发生在 save(u+1) 内部的非阻塞回收里，两臂的端点在时序上大体对称。
  - 差别在于：R 的 B2 会阻塞等待；A 的回收是非阻塞的，写盘没完成就推迟到 save(u+2)。所以 R 的端点可能早于 A。报告要分开列出两者的等待成分。
- `via=recovery`：由恢复进程在选择恢复点时提升第 u 代，并补发 `committed` 事件（`training_adapter.py:271-272`）。它的时刻包括重启、选择恢复点和全量内容哈希的时间，属于如实的"首次纳入恢复权威"。
- 在 F4 被吸收的情况下，R 的端点就是正常路径上的提交，与恢复无关（见 F-1）。
- `finalize_ft1_fault.py:37-42` 按 `committed` 事件累计 manifest，已经把 `via=recovery` 算在内。应改为按 token 链逐代累计，事件只用来取时间戳，避免依赖事件的完整性。

---

## 3. F4、F1 与接入清单（问题 3）

### F-1 F4 会被原生重试完全吸收（高概率，而且已有实证）

- 切点是一次性地杀掉第 4 个评分子进程。严格评分每条样本最多尝试 2 次（`reward_api.py:86-87`：strict 时 `max_retries=1`），所以重试会对**同一输入**成功。
- FT1 的 F4 A 实跑已经证明了这一点：重试对相同 input SHA 成功，训练器没有重启（`FT1_RUN_REPORT.md:37`）。
- 后果：
  - 两臂都只多做 1 次评分，损失范围相同；
  - R 的事务层（receipt 复用、token）完全没有被触发；
  - RTO 等于"5518 组所在批次的正常持久化延迟"。lag-1 下 R 本就晚提交一步，这个指标反映的是正常路径设计，**不是恢复能力**。
- **必须：**
  - 试点判据增加"训练器是否重启"（新 pid 或新 owner epoch）以及"重试次数"；
  - 预先写定：如果吸收成立，F4 在正式矩阵中只作为"无回归"检查（两臂分类一致、R 不劣化），不进入 RTO 的比较。
  - 如果确实需要一个能触发恢复的评分故障，应另设切点（例如让重试也失败，或者杀掉训练器），并单独审计。

### F-2 F1 尚未在 GPU 上执行，恢复路径未知

- F1 的公共钩子只做过 CPU 合同测试，真实的生成 worker SIGKILL 从未执行过（`FT1_RUN_REPORT.md:59`）。
- 杀掉 SGLang 之后，AReaL 可能出现三种情况：
  1. 请求层重试或重连（被吸收）；
  2. rollout 抛出异常，训练器退出，由 launcher 执行 `retries: 1`，整组重启（训练器加推理服务）；
  3. 挂死。
- 试点要能分辨这三种情况：需要记录 launcher 的重启事件、推理服务的 pid 变化，并设置挂死超时。
- **900 s 窗口能否恢复**：以旧数据估计，重启到首批约 225 s（启动约 75 s，加上 trainer 到首批约 150 s），再训练到 5518 组被消费并持久化。5518 在数据顺序中的位置取决于种子，如果排在第 10 步之后，就永远不会被消费。
- **必须：**
  - 冻结时按种子预先计算 5518 所在的步序，只选在故障后、第 10 步之前会被消费的种子；
  - 或者把"未消费"单独归为一类。
- 情况 2 下，A 的恢复取决于故障时 A 是否处在覆盖写窗口（A-1）。F1 不像 F2 那样先排空写盘，所以 A 有相当的概率加载到不一致的状态。这必须单独记录，是真实的原生行为，不能算技术无效。
- 对 R：lag-1 下 `writer.json` 更常处于待决状态，重启会走 `verify_cleanup`，需要 job-lifecycle 清理回执。要确认最小流程的 launcher 链路（`job_lifecycle`）在 F1 重启时会产生这个回执。

### F-3 接入改动清单：不完整

方案列出的文件都需要修改，此外还缺以下几处：

- `tests/ft/run_ft_minimal.py`：
  - 目前硬编码只允许 F2/no_fault（`:19-24, 60-61, 121`）；
  - 记录里写死了 `f2_ordinal`（`:86`）；
  - 只有 F2 的 R 臂检查"全部 32 条复用"（`:99`）。
  - 需要为 F4/F1 定义各自的 R 臂期望：F4 没有复用可言，F1 视恢复路径而定。
- `run_training_fault.py`：F1 已经复制 `f1-target.json`（`:67-68, 115`）。但 schedule 中目标为 `generator`/`reward` 的 waiter 注册要确认在最小流程中可用：需要 `FT_CONTROL_SOCKET`、`FT_RUN_NONCE` 环境变量，F1 钩子依赖它们（`ft1_f1_trainer.py:15-19`）。
- `areal_pilot_hooks.py`：类级包装、save_id 绑定（A-2、A-3），属于共用观测器的改动。
- `check_ft1_chain.py`、`check_training_fault.py`、`check_training_integration.py`：阶段 2 已针对提升的代做过修改，但要确认在 F4/F1 下（训练器可能不重启，或者整组重启）判定仍然成立。
- `finalize_ft1_fault.py`：
  - A 分支的端点（`:43-44`）；
  - `persistence_time_kind` 与 `formal_rto_eligible`（`:52-54`）；
  - 新增"训练器是否重启 / 重试次数 / A 是否处于覆盖写窗口"等字段；
  - 用 token 链替代事件累计（见 §2）。
- `check_ft1_fault.py`：现有 F1/F4 的判定（`:40-92`）要与最小流程的输出目录结构对齐。
- `run_ft_formal.py`：`pilot` 类型；冻结文件需记录种子对应的 5518 步序。
- 在 CPU 上补充以下测试：
  - 类包装与 R 实例包装叠加时，两臂事件一致；
  - save_id 配对；
  - A 端点计算，用构造的"覆盖写窗口"样例；
  - F4 吸收的判别。

---

## 4. 试点判据与规模（问题 4）

- 4 对（F4、F1 各 2 对）用于回答"能否测量、是否被吸收、能否在窗口内恢复"是够的，不应用于决定效应大小。方案已声明不做统计检验，这一点正确。
- **判据需要补充：**
  - F4、F1 各自的"是否被吸收"（训练器是否重启）；
  - 5518 是否在故障后 10 步内被消费；
  - A 是否处于覆盖写窗口；
  - F1 挂死的超时处理；
  - 切点未命中时补做一次的规则（方案已有）。
- **预先写定决策规则，示例：**
  - F4 被吸收 → 正式矩阵中 F4 只做无回归检查，对数可以减少；
  - F1 两对都在窗口内恢复且可测 → 进入正式矩阵；
  - F1 挂死或无法测量 → 先修接入，不进入正式矩阵。
- 预算约 8 GPU·h：以往每对约 1.5–2 GPU·h，与之一致。如果 F1 发生挂死或需要补做，可能超支，应设置上限。

---

## 必改项（简短）

1. **A 端点**：标注为"首次瞬时完整（下界）"，并记录 A 的可恢复窗口，以及故障时 A 是否处于覆盖写窗口；覆盖写窗口内的样本单独分类。口径中写明"持久＝对进程崩溃持久"（A-1）。
2. **类级包装**：必须在 `attach` 之前安装并加断言；只在返回列表非空时发事件；作为共用观测器的改动登记新冻结（A-2）。
3. **save 与写盘调度的配对**改为用线程局部的 save_id 或 `meta.path` 显式绑定，不按顺序配对（A-3）。
4. **目标映射**：读取所有 pid 的观测文件，并断言 5518 组的 8 条都能映射到；R 端改为按 token 链累计（A-4、§2）。
5. **F4**：判据增加"训练器是否重启 / 重试次数"；预先写定"被吸收 → 只做无回归检查，不进入 RTO 比较"（F-1）。
6. **F1**：判据能区分"被吸收 / 整组重启 / 挂死"，设挂死超时；冻结时只选 5518 会在故障后 10 步内被消费的种子；确认 F1 重启时会产生 `verify_cleanup` 所需的 job-lifecycle 回执（F-2）。
7. **补全接入清单**：`run_ft_minimal` 的场景白名单与各场景的 R 臂期望、`areal_pilot_hooks`、`check_ft1_fault` 与最小流程目录对齐、三个 check 工具在 F4/F1 下的回归、`finalize_ft1_fault` 的新字段；并补 CPU 测试（F-3）。
8. **预先写定试点的决策规则和 GPU 预算上限**（§4）。
