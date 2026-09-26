# F4/F1 试点方案 v1.2 增量审计（F4'：评分进行中杀训练进程，2026-09-26）

- **对象**：[FT_F4F1_PILOT_PLAN_20260926.md](FT_F4F1_PILOT_PLAN_20260926.md) 的 §6（v1.1，落实上轮审计的必改项）和 §7（v1.2，F4' 新方案）。
- **基线**：[上轮审计](FT_F4F1_PILOT_PLAN_AUDIT.md)。
- **代码**：分支 HEAD，阶段 2（lag-1）已实现。
- **方式**：只读审计，未改代码或已有文件，未使用 GPU。

## 结论：有条件通过

F4' 的切点可以实现，而且两臂确实停在同一个逻辑点。但方案对两臂恢复行为的预期有两处和代码不符，必须改。

1. **A 很可能根本不会重生成 5518 组，而是直接丢掉它（`safe_discard`），因此 A 端点可能无定义（A-2）。**
   AReaL 的 recover_info 保存了数据加载器的位置。rollout 会提前取数，5518 组通常在最近一次 dump 之前就已被取走，恢复后 A 会跳过这一组。这与冻结 F2 中 A 为 `safe_discard` 0/32 的机制相同。
2. **R 只能采纳"已 accept 的完整 receipt"，而且 lag-1 下它们大概率因版本领先于恢复点而被拒（R-2）。**
   - 只写了 response 或 reward 的中间产物永远不会被复用。
   - 在 lag-1 下，新生成回答的策略版本往往比恢复后的版本还高，`0 <= current - v` 这一检查不成立，所以即使有 receipt 也会被拒。
   - 结果是"R 采纳 ≥1 条"这一判据可能系统性不满足。这是 lag-1 的如实代价，应预先登记，而不是事后解释。

---

## 1. 切点可实现性与两臂一致性（问题 1）

- **包装链。** 共用观测器在类级别替换了 `RLVRWorkflow._compute_rewards`（`areal_ft1.py:70-79`，包装进入时发 `generation_complete`）。两臂的子类都通过 `super()` 调用到这个包装：
  - A：`ObservedRLVRWorkflow._compute_rewards`（`areal_pilot_hooks.py:152-165`）→ `super()`；
  - R：`CallReturnRLVR._compute_rewards`（`rlvr_replay.py:290-301`）→ `super()`（`:298`）。

  因此，在 ft1_fault_hooks 中对同一个类属性再包一层、在"返回处"计数，两臂走的是同一段代码。
  - 前提 1：新包装必须在 `install_observer()` 之后安装。`areal_ft1.py:289-294` 目前的顺序满足这一点。
  - 前提 2：只统计**成功返回**。严格评分如果内部重试后仍然失败，不计数。
- **R 在返回之后才落盘第 4 条 reward。** R 在包装返回后还要做 `_check`、`_validate_return` 和 `_put(reward)`（`rlvr_replay.py:299-301`）。钩子在包装内部、返回前阻塞，所以第 4 条 reward 一定不会落盘。前 3 条是否已 accept，取决于单线程 I/O 池的进度（见 R-1）。
- **"8 条都已生成"的条件必须保留。** 评分是串行的，生成是并发的；第 4 条评分返回时不能保证 8 条都已生成。要沿用原 F4 的"生成掩码 == 255"判定（`ft1_fault_hooks.py:47-60`），不满足时记为 `fault_cut_missed`。
- **复用 F2 的 trainer `Client`。**
  - `Client` 在 `install` 时由主线程创建（`ft1_fault_hooks.py:88`）。之后在 rollout 事件循环线程里调用 `ready`/`wait_release` 是可以的：这两个方法只校验 `os.getpid()`（`descendants.py:213-226`），同一个 socket 也不会被并发使用（F4' 时没有别的线程在用）。
  - 新增契约 `ft1-f4t-trainer` 要在控制器的场景白名单和 schedule 中注册。
  - 重启后的新进程注册时会拿到 `already_fired`，计数钩子必须据此跳过，保证只触发一次。
- **在事件循环线程里阻塞：可以接受，但要写明影响。**
  - 阻塞期间所有 rollout 协程都停住，这正是想要的"冻结 rollout 现场"。
  - 但**主线程和 I/O 线程还在跑**：训练、B1/B2 提交、fork 写盘进程、已排队的 `accept_result`、reward 的 put 都可能在 ready 到 SIGKILL 之间继续推进。
  - 因此"切点"只冻结了 rollout，训练所处的阶段是任意的。要求记录 ready 与 kill 的时间差，并在 kill 时（由控制器侧或观测器）记录主线程所处的阶段。

## 2. 恢复行为（问题 2）

- **R-1 R 能采纳什么。**
  - 重启后，`TrainingBridge.authorize` 只在同时满足以下条件时才复用：存在 receipt，且 `receipt['attempt'] == old`（`training_replay.py:161-170`）。receipt 只在整条样本完成 tensor 阶段并 `accept_result` 之后才存在（`rlvr_replay.py:258`）。
  - 不能复用时会授权新的 attempt，artifact 目录按 attempt 区分（`rlvr_replay.py:236`），所以**只有 response 或 reward 的中间产物永远不会被复用**。
  - `restore_artifacts` 的采纳也只针对 `origins`，而 `origins` 仅在可复用时才设置（`training_replay.py:106-143`）。
  - **§7"R 可能已持久化前 3 条 reward 与若干 response …… 优势正来自这种差异"需要改写**：R 能复用的只有切点前已经 accept 的样本（0–3 条），其余全部重新生成和评分。
- **R-2 版本检查（关键）。**
  - 复用还要求 `0 <= current_version - v <= max_lag`（`training_replay.py:168-169`）。
  - `current_version` 是恢复后的 `last_step_info.global_step + 1`（`rl_trainer.py:481`）；`v` 是回答生成时的权重版本，每次 `update_weights` 后 `set_version(step+1)`（`rl_trainer.py:888-895`）。
  - lag-1 下，已应用但未进入恢复链的更新通常有 1–2 次：第 k 代要到 save(k+1) 开头才提交；它的 finalize 证据也是在那时（或 AReaL 回收时）才写入，未 finalize 的代在恢复时会被放弃。
  - 因此切点附近新生成的 5518 回答，版本 `v` 常常**大于**恢复后的版本，差值为负，会被拒绝。拒绝本身是正确的语义：该策略版本已经不存在。
  - 只有用较旧权重生成的回答（off-policy 窗口内）才可能复用。F2 能 32/32 复用，是因为那批样本的版本 ≤ 恢复点。
  - **要求：**
    1. 在每条 5518 样本上记录 `v`、恢复后的版本、是否被提升、是否可复用；
    2. 用阶段 2 门控无故障 run 的事件，离线模拟"在每一步各个时刻切断时 5518 的可复用比例"，并把预期采纳率预先登记；
    3. 如果预期采纳率接近 0，事先说明 F4' 在 lag-1 下主要测的是 RPO+1 的代价，而不是 R 的复用优势。
- **A-2 A 不一定会重生成（关键）。**
  - A 恢复时从 recover_info 加载 `dataloader_info`（`recover.py:301-316`，在每次 dump 时写入）。
  - rollout 会提前取数（`max_head_offpolicyness: 2`，`prepare_batch` 会先提交后续批次），所以切点时 5518 组很可能已经被取走，并记录在最近一次 dump 的加载器状态里。恢复后的加载器会跳过它，A 永远不会再训练这一组，分类为 `safe_discard`，A 的 RTO 端点**无定义**。
  - 这与冻结 F2 中 A 为 `safe_discard` 0/32 的机制一致（`FORMAL_RESULTS_20260925.md`）。
  - **要求：**
    - 判据"两臂都能测出端点"改为"R 可测；A 按分类处理"。`safe_discard` 时记为未恢复，RTO 为无穷或删失，不算技术无效；
    - 统一口径中写明对删失的处理；
    - 同时记录 5518 在 A 的加载器状态中是否"已被取走"。
- **A 的覆盖写窗口。** 切点处 A 可能正在后台写 recover_checkpoint。SIGKILL 只杀训练进程（`descendants.py:164-166`），fork 出的写盘子进程由 job_lifecycle 的子进程收割器负责清理（`job_lifecycle.py:30-60`，先 SIGTERM，再 SIGKILL）。于是 A 的检查点被部分覆盖，重启时可能加载失败或加载到不一致的状态。这属于原生行为，按 v1.1 §6.1 单独分类即可；另外还要登记"A 重启失败导致重试耗尽"的结局。

## 3. lag-1 与 writer_gate（问题 3）

- **两个待决候选。** 切点处可能同时存在两代未决 generation：第 k 代（已保存，待 B2 提交）和第 k+1 代（已 prepare 或已保存）。阶段 2 的 `select_recovery` 两候选链规则可以处理这种情况：
  - c1 完整（有 finalize 证据、`native-state.json` 已发布）则提升，否则 c1 和 c2 都放弃；
  - 两代的消费义务都回滚，由 `RetainedLoader` 的待生成队列重新处理。
- **writer.json 可能处于待决状态。** 如果切点落在 save(k) 与 finalize 之间，重启时会走 `verify_cleanup`（`training_adapter.py` 的 `make_loader`；`writer_recovery.py:39-67`）。它需要满足两点：
  - 训练进程是 job_lifecycle 的根进程；
  - 存在唯一一张已完成、`cleanup.empty` 的回执。

  F4' 是第一次在任意阶段杀训练进程的场景，**要求**在最小流程中用 CPU 或 midwrite 工程路径确认 launcher 重启训练器时这张回执能生成，并在试点证据中记录是否出现了 `pending_writer_abandoned`。
- **R 的 hasher 等线程被杀时，所有写入都是原子的（tmp 文件加 link/replace），不会产生半写文件。**

## 4. 5518 的位置与种子筛选（问题 4）

- 两臂的数据顺序相同：R 的 DrawLoader 包装的是同一个基础加载器，种子也相同。因此可以离线重放加载器（种子加 `StatefulDataLoader` 的 sampler），计算 5518 所在的训练步 s。
- 由于提前取数，5518 的生成和评分发生在第 s−2 到 s 步之间，切点也在这段时间内。
- **筛选条件：**
  1. s ≥ 2：切点之前至少有一次 commit 或 finalize，两臂都有可恢复点。否则两臂都从头开始，信息量低，不过 R 仍可能复用 receipt，可以单列一类；
  2. 重启后要在第 10 步内重新训练到这一组。假设重启约 225 s，重训练 s' 步，要求 s' 加上剩余步数 ≤ 10，即 s ≤ 8；
  3. 只选这样的种子，并把 s 写入冻结。
- 5518 只作为 F4'/F1 的目标，可以与 F1 共用同一套筛选逻辑。

## 5. §6 必改项落实情况（问题 5）

在方案层面，8 条都已落实。其中第 5 条已被 F4' 取代，第 6 条的种子筛选需要按本文 §4 加上 s 的上下界。

以下尚待实现时核验：
- 类包装在 `attach` 之前安装的断言；
- `meta.path` 或 save_id 的显式配对；
- 读取所有 pid 的观测文件；
- 沿 token 链累计端点；
- 回执生成；
- 新字段和 CPU 测试。

建议在实现文档中逐条给出"代码位置加测试名"。

---

## 必改项（简短）

1. **F4' 钩子**：安装在 `install_observer` 之后，包在同一个类属性上；只统计成功返回；保留"8 条已生成"的掩码条件；已触发过的进程跳过；登记新契约。记录 ready 到 kill 的时间差，以及 kill 时主线程所处的阶段（§1）。
2. **改写 §7 对 R 优势的描述**：只有已 accept 的 receipt 可以复用，中间产物不复用。记录每条样本的 `v`、恢复后的版本、是否被提升、是否可复用；用门控无故障 run 离线预估并登记预期采纳率（R-1、R-2）。
3. **A 恢复口径**：预期 A 为 `safe_discard`，按删失（未恢复）处理 RTO；判据改为"R 可测，A 按分类处理"；记录 5518 在 A 的加载器状态中是否已被取走，以及 A 是否处于覆盖写窗口或重启失败（A-2）。
4. **确认** job_lifecycle 在 F4' 重启时生成回执，并记录 `pending_writer_abandoned`（§3）。
5. **种子筛选**：离线计算 5518 所在的步 s，要求 2 ≤ s ≤ 8，把 s 写入冻结（§4）。
6. **决策规则补充**：如果 R 的采纳为 0 是由版本领先导致的（预期之内），F4' 转为度量"lag-1 的 RPO 代价"，不作为复用优势的依据；这一点要在跑之前写定。
