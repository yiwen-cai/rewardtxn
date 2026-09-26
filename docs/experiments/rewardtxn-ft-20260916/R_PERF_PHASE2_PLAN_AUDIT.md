# R 性能阶段 2 方案独立审计（2026-09-26）

对象：[R_PERF_PHASE2_PLAN_20260926.md](R_PERF_PHASE2_PLAN_20260926.md)（下称"方案"）。对照的代码为当前 HEAD：`14fff2c`，其中阶段 1 实现在 `240b8d3`、`9825fcd`，包括 `scripts/ft/state.py`、`training_adapter.py` 以及 `tests/ft` 下的验收工具。

只读审查：没有修改代码或已有文件，没有启动 GPU。

## 结论：有条件通过

state 协议扩展的方向是对的：token 链与 `_chain` 不变，consumed 仍然只由 token 授权。

但有 **3 个阻断项**，必须在实现前改掉：

1. **B1 放在 optimizer 前太晚。** 快照捕获的是活引用，其中 RNG tracker 状态可能在 `train_batch` 的前向中被改动；另外 B1 只 join 快照计算，**不保证 `native-state.json` 已持久化**。F2 在被杀时第 0 代可能缺少这个文件，于是被放弃，R 的 F2 结果会改变（A-1、A-2）。
2. **prune 移到锁外删除后，与 `_head` → `_check_files` → `_identities` 的枚举存在竞态**，会在 prepare 或 commit 中抛出 `FileNotFoundError`（A-9）。
3. **现有验收工具不认"由恢复提升"的代和 pending 形式的 parent。** `check_training_fault.py`、`check_training_integration.py` 断言 `manifest['parent'] == token['parent']`，并且要求每一代都有同一 pid 的 `committed` 事件；`check_ft1_chain` 要求 `committed` 事件与链一一对应（A-7）。

---

## 1. state 协议（问题 1）

### 不变量核对

- **"被消费的样本只来自已提交链"：成立。**
  - 第 k+1 代的 token 要求提交时的 head 就是 P（CAS），并且 P 的 token 中的 `intent_sha256` 等于 k+1 代 pending parent 里记录的值。
  - `_chain`（`state.py:591`）仍然只看 token 链，所以一个 consumed 能生效，意味着它的所有前驱都已提交。
  - 训练进程在 P 未提交时就用 k+1 的样本更新参数，这属于"已应用但未提交"的状态，RPO 上界是 2 步。与方案声明的 +1 一致：相对同步版多回退 1 步。
- **恢复时不会提升不完整的代：基本成立。** 提升前必须通过 `_complete`：
  - optimizer 和 finalize 两类 evidence 齐全；
  - 11 个组件映射完整，`native-state.json` 与 `policy.json` 存在；
  - `referenced == files`；
  - 对全部文件做内容哈希（`state.py:564-609` 的等价代码）。
  - c2 只在 c1 已提升后才考虑。
  - 仍存在的风险见 A-2 和 A-4。
- **回滚消费义务：成立。**
  - c1 被放弃时，c2 也写 abandoned，两者都进入 `rollback_intents`（`state.py:838-839`）。
  - 新进程的 `RetainedLoader` 以 token 链 head 的 consumed 为准（`training_replay.py:22-24`），被放弃两代的样本会重新成为待生成或可复用。

### 需要写明或补充的地方

- **A-5 prepare 对 P 的附加约束。** 除方案列出的条件外，还要求：
  - `P.intent.execution_epoch == owner.epoch`，且 `execution_owner` 是当前 owner（防御上一任 owner 遗留的未决代；正常情况下 `select_recovery` 已经处理掉了）；
  - `P.consumed ∩ samples = ∅`，并且 `data_snapshot.drawn` 以 `P.drawn` 为前缀；
  - pending parent 的 `intent_sha256` 必须对 P 的 `intent.json` 原始字节计算，与 token 的定义一致（`state.py:646` 的等价行）；
  - 有两个未决代，或者 P 已被 abandoned 时，一律拒绝。
- **pending parent 对以下几处的影响：**
  - `commit_generation` 的 `intent["parent"] != head`（`state.py:773`）；
  - `select_recovery` 的候选 parent 检查（`state.py:826`）；
  - `prepare_generation` 的 parent/previous/`prior_updates` 检查（`state.py:623-675` 一带）。
  - 这三处都要改成"committed 形式或 pending 形式"两种分支，并加一个限制：**pending 形式只允许指向当前唯一的未决代，或者（恢复时）指向刚刚提升的 c1**。
- **`select_recovery` 的顺序：**
  - 先在锁内判定 c1 和 c2 的完整性；
  - 目前 promote 在锁外调用 `commit_generation`（`state.py:833-834`），c2 的 CAS 需要在 c1 提交之后重新读 head；
  - c1 提升失败（commit 抛异常）时，c2 必须写 abandoned，不能停留在未决状态。
- **提升只校验 DCP 文件内容的哈希，不校验"`native-state.json` 与 DCP 语义一致"。** 后者到加载时才比较（`training_adapter.py:476-480`），不一致直接报错，不会退回 parent。同步版也是这样；但阶段 2 新增了一个不一致的来源（A-1），见下。

---

## 2. 提前落证据（问题 2）

- **A-1 快照可能与 DCP 内容不一致（阻断）。**
  - `generate_state_dict` 返回的是活引用（`training_adapter.py:361`）。方案把 hasher 的生存期延长到第 k+1 步的前向和反向（B1 放在 optimizer 前）。
  - 参数张量和优化器矩在 optimizer 之前确实不会变。
  - 但 RNG 部分不同：Megatron 的 `get_cuda_rng_tracker().get_states()` 依实现可能返回可变对象的引用（启用 cudagraph 安全 RNG 时，tracker 保存的是 Generator，前向中的 dropout 会原地推进它）。另外 lr_scheduler 和优化器 `param_groups` 中也可能有可变容器。
  - 阶段 1 的屏障在 `begin`，早于任何训练前向，所以不暴露这个问题；阶段 2 会暴露。
  - **要求**二选一：
    1. 在 `save` 的主线程中，把非参数、非矩的全部组件（`rng_state`、`lr_scheduler`、`param_groups` 与 step 等标量）**深拷贝或直接规范化为值**，hasher 只处理参数和矩张量；
    2. 把 B1 提前到 `prepare`/`begin`，这会损失一部分重叠收益。
  - 无论选哪种，都要加一条 CPU 测试：在 B1 之前推进 RNG 或 tracker，快照结果不得改变。
- **A-2 `native-state.json` 的发布时点（阻断，影响 F2）。**
  - `_complete` 要求组件文件存在（`native-state.json` 在 `components` 列表中，缺失会报 `component file missing`）。
  - 方案中 B1 只 join 快照计算。如果 hasher 算完但还没 `_publish` 时进程被杀，这一代就不完整，会被放弃。F2 在第 2 次 optimizer 之后被杀，第 0 代因此可能被放弃，R 的 F2 结果从"提升第 0 代"退化为回到空链。
  - **要求：** B1 必须等 `native-state.json` 完成持久发布（`_publish` 返回）。hasher 快照失败时，由主线程在 B1 同步计算并发布。
- **A-3 finalize evidence 的正确性：可以成立，但需写明以下条件。**
  - finalize 包装只能在 `finalize(...)` **正常返回**、且返回的 call_id 属于某个已登记的待决代时，才写 `finalize` evidence 和 `writer_gate(False)`。call_id 到代的映射必须来自 `save` 时登记的表，不能用 `_io_generation()` 这类"当前"状态（`training_adapter.py:292-293`）。
  - `maybe_finalize_async_calls` 失败会抛异常，不会返回 id，所以"`.metadata` 缺失却写了 evidence"的情况不会发生。
  - 包装内部顺序：先 evidence，后 gate，再发 `async_finalized` 事件。
  - `record_evidence` 要取 `mutation.lock`，而 finalize 包装可能从 AReaL 的 `_reap_finished_async_saves`（`checkpointer.py:609`，在 `save_checkpoint` 中）或 F2 钩子里被调用，这些都在主线程上，没有问题。
- **A-4 optimizer evidence 与 `policy.json` 提前到 `original_dump` 之前：可以。** optimizer 和 scheduler 已经成功（`training_adapter.py:338`），内容与同步版相同。只是崩溃后这一代会留下 evidence，而 DCP 未完成，`_complete` 会因缺少 finalize evidence 而放弃它，这是安全的。
- **writer 已退出但未 finalize：** 不会写 finalize evidence，这一代不会被提升，安全。

---

## 3. 屏障位置与 AReaL 的非阻塞 reap（问题 3）

- **B2 在 `save(k+1)` 开头：位置正确。** 它在 `RecoverHandler.dump` 包装内、`original_dump` 之前执行，因此：
  1. `wait_async_saves()` 会把第 k 代的 call 在主线程上 finalize 掉（如果还没完成）；
  2. 之后 AReaL `save_checkpoint` 里的 `_reap_finished_async_saves()`（`checkpointer.py:607-609`）面对的是空队列，不会与包装重复写 evidence；
  3. 同一时刻最多一个 writer，`writer.json` 单槽成立；
  4. 当前 HF saver 关闭（`training.yaml` 中 `saver.freq_steps: null`），`save` 前没有其他 DCP 调用。
- **A-6 B2 与 loader.consumed 的先后顺序。** 方案在 `save(k)` 中把 `loader.consumed` 设为第 k 代 intent 的 consumed。B2 提交第 k 代时，现有代码会用链上的值覆盖它（`training_adapter.py:410-412`）。如果 B2 发生在第 k+1 代的 consumed 已经设置之后，就会**回退**成 k 的 consumed。
  - 要求：B2 只能单调前进（不覆盖更新的视图），或者在 B2 之后再设置第 k+1 代的视图，并补一条测试。
  - `RetainedLoader.__iter__`（`training_replay.py:42`）依赖这个视图来跳过已占用的组，所以回退只会导致重复授权被拒，不会造成错误消费；但会让 prepare 的 consumed 计算出错，进而让 `prepare_generation` 报错。
- **A-1、A-2 说明 B1 的位置需要调整或加条件。** 如果采用 A-1 的方案 1（主线程把非张量组件规范化为值），B1 可以留在 optimizer 前，但必须包括发布完成。
- **结束路径：**
  - `train` 包装的正常和异常出口依次执行 B1、B2 可以；
  - 异常出口中 B2 失败时，保留现有的 `abandon_pending`（`training_adapter.py:505-511`）；
  - `close` 只 join、不提交是正确的（第 k 代可以在恢复时被提升）。

---

## 4. F2 与验收工具（问题 4）

- **F2 在 R 侧的状态，要满足以下条件才与冻结版一致：**
  - 钩子的顺序：`ft1_fault_hooks.py` 的包装在外层，Runtime 的 optimizer 包装在内层，B1 在 `original_optimizer` 之前执行，先于钩子的 `wait_async_saves`（`ft1_fault_hooks.py:121`）；
  - 钩子的 drain 经过 finalize 包装，写入第 0 代的 finalize evidence 与 gate(False)；
  - 满足 A-2 后，第 0 代完整，恢复时被提升；第 1 代只有 intent，被放弃；
  - 恢复后的 head 等于冻结版，目标样本通过 adoption 复用，分类预期不变。
  - 前提是 A-1、A-2 都修好。
- **A-7 验收工具的兼容性（阻断）。**
  - `check_training_fault.py:92` 和 `check_training_integration.py:54` 断言 `manifest['parent'] == token['parent']`。pending 形式的 parent 必然不相等。
  - `check_training_fault.py:132-137` 要求每一代都有 `update_prepared…committed` 六个事件，并且来自同一 pid。被提升的第 0 代既没有 `committed` 事件，提交者也是新进程。
  - `check_ft1_chain.py:98-99` 要求 `committed` 事件列表等于整条链。
  - `finalize_ft1_fault.py:38` 用 `committed` 事件累计目标并取端点；被提升的代没有这个事件，但目标行在后续提交中，端点仍可取到。需要复核端点语义是否被改变。
  - **要求：**
    1. 提升时由恢复进程写一个 `committed` 事件，并带 `via='recovery'`；或者工具改为以 token 链为准，事件只作为辅助；
    2. parent 的比较改成"token.parent 是 committed 形式，manifest.parent 是 committed 形式或指向 token.parent 所指代的 pending 形式"；
    3. 单 pid 断言对提升的代放宽为"执行 pid 与提交 pid 可以不同，但 `commit_epoch` 必须等于 `execution_epoch + 1`"；
    4. 在 CPU 上用构造好的证据目录回归这三个工具。
  - 同时搜索并修改 `p2_schedule_driver.py`、`check_ft_minimal_source.py`、`test_p2_second_fault_probe.py` 中读取 parent 的地方（它们读的是 token.parent，预计不受影响，但要确认）。

---

## 5. prune 后台化（问题 5）

- **A-9 锁外删除与枚举的竞态（阻断）。**
  - 当前实现仍在锁内删除（`state.py:900-905`；阶段 1 审计要求的第 12 条尚未落地）。
  - 如果改为锁外删除，其他线程在锁内执行 `_head` 时，会对**每一代**执行 `_check_files` → `_identities`（`state.py:394-406`）。`rglob` 列出文件后再 `lstat`，此时文件可能已被删除，抛出 `FileNotFoundError`，而这个异常不是 `StateError`。结果是 prepare、commit 或 pin 随机失败。
  - **要求**二选一：
    1. `_identities` 对"已在 `pruned.json` 中列出的名字"容忍消失（`lstat` 捕获 `FileNotFoundError` 后跳过，并由 `_check_files` 确认该名字在 removed 中）；
    2. 删除仍在锁内进行，但放到后台线程，并一次只删一个文件、每删一个就释放锁。
  - 另外：
    - 同一时刻最多一个 prune；
    - `present` 列表要在删除时重新判定，避免重复 unlink；
    - prune 线程必须在 `close` 之前 join，否则 `_locked` 会报 inactive owner；
    - 进程被 SIGKILL 时依靠标记恢复，现有语义支持。
- **保留集合：** keep=2 只针对已提交链，未决代不在链上，永远不会被 prune；恢复加载的代有 pin 保护。
  - 在 lag=1 下，恢复时需要读取内容的只有 head（`verify_head_content`）和候选代，都不会被删。
  - 需要补一条测试：c1 被提升时，它的 parent（旧 head）的 `.distcp` 已被 prune 不影响提升。`_complete` 不读 parent 的内容，实际上不会受影响，但要测一下。

---

## 6. 性能预期（问题 6）

- **能去掉的：** 屏障中的 finalize 4.7 s、join 1.8 s、prune 0.8 s 都可以和下一步重叠。只剩 B2 的 commit 约 0.2 s，再加上 B1 可能的等待。
- **登记值 1.00–1.15 偏乐观。** 门控显示 R 每步"等下一批"已达 5.9 s（A 为 0.85 s），说明 R 的 rollout 已接近成为瓶颈：I/O 线程 CPU 33.8 s，flock 等待 15 s。去掉屏障后，更多时间会变成等 rollout，周期会趋向 R 的 rollout 时长，而不会降到 A 的水平。
- **B1 的风险：** 快照的 D2H 走默认流，会排在训练核函数后面（阶段 1 已观察到 join 1.8 s）。它与第 k+1 步的前向和反向并行，会拉长训练段；采用 A-1 方案 1 时主线程还要多花值拷贝的时间。
- 建议登记为 1.05–1.25。并在方案中写明：如果不达标，主要原因预计是 rollout 侧的 I/O 线程开销（§1.4.2），届时报告原因即可，不再自动推进下一阶段（方案已写）。
- **n=1 的问题：** A 本次循环只有 158 s，而旧版约 211 s，跨 run 波动很大。门控应至少使用同冻结的 2 对，或者报告每步中位数，不要只报总和。

---

## 必改项（简短）

1. **快照一致性：** 在 `save` 主线程中把 RNG、scheduler、`param_groups`、step 等非张量组件按值规范化，或把 B1 提前到 `begin`；加测试：B1 前推进 RNG 或 tracker，快照不变（A-1）。
2. **B1 等到 `native-state.json` 持久发布完成；** hasher 失败时由主线程补算并发布（A-2）。
3. **finalize 包装只在 finalize 正常返回后，按 `save` 时登记的 call_id→代 映射写 evidence 和 gate**（A-3）。
4. **写明 prepare 对 P 的附加约束；** commit、`select_recovery`、prepare 中的 parent 检查分成两种形式；c1 提交失败时 c2 必须写 abandoned（A-5、§1）。
5. **B2 与 `loader.consumed` 只能单调前进，** 并补测试（A-6）。
6. **验收工具：** 提升的代要有带 `via='recovery'` 的 `committed` 事件，或工具改以 token 链为准；修改 parent 相等断言和单 pid 断言；在 CPU 上回归 `check_training_fault`、`check_training_integration`、`check_ft1_chain`、`finalize_ft1_fault`（A-7）。
7. **prune：** `_identities` 容忍已在标记中的文件消失，或者分片在锁内删除；最多一个 prune；`close` 之前 join（A-9）。
8. **性能登记改为 1.05–1.25；** 门控至少 2 对，或报告每步中位数（§6）。
9. **CPU 测试补充：**
   - F2 式序列：B1 → optimizer → 钩子 drain → SIGKILL，结果为提升第 0 代、放弃第 1 代；
   - hasher 发布前被杀，这一代被放弃；
   - finalize 抛异常时不写 evidence；
   - prune 与 prepare 并发时不出错。
