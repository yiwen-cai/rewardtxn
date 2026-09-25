# R 正常路径性能重设计方案 v1 独立审计（2026-09-25）

审计对象：[R_PERF_REDESIGN_PLAN_20260925.md](R_PERF_REDESIGN_PLAN_20260925.md)（下称"方案"），依据 [R_OVERHEAD_PROFILE_20260925.md](R_OVERHEAD_PROFILE_20260925.md)（下称"剖析"）。

审计方式：只读当前分支代码（`scripts/ft/*.py`，不含 `*.bak-*`；`third_party/areal` 工作树，含本项目补丁），重算三对正式无故障 run 的事件时间线（`minimal_evidence/formal-nofault-p1{1,2,3}-*`），在宿主机上做了两项 CPU 微测（仅读取证据文件）。没有修改任何代码或已有文件，也没有启动 GPU。

Megatron 说明：`third_party/areal/uv.lock:3307-3308` 锁定 megatron-core 0.17.0，但本机没有这个版本的源码。下文关于 `AsyncCallsQueue` 的引用取自 0.18.0 源码（`/home/lijun/repos/Megatron-LM/megatron/core/dist_checkpointing/strategies/async_utils.py`）。AReaL 自己的注释（`checkpointer.py:668-713`）也写明这些调用是集合通信，两者结论一致。

---

## 结论

**有条件通过**：
- D1 按保守版实施，并补上本文列出的漏项；
- D2a 按本文 R-1 至 R-6 修改设计后才能实施；
- D3 按现有写法**不通过**，需要改写；
- 阶段 1 的性能目标（≤1.15 倍 A）在现有设计下大概率达不到，需要预先登记预期值，不要预设能达标。

最重要的三个问题：

1. **H1 归因错误。** R 的 `prepare` 不会等待任何评分或 receipt。批次到手 → `update_prepared` 这段多出的 12–23 s，是**训练主线程被同进程内并发的下一批 rollout 拖慢**（争用），再叠加训练侧重复校验。按"每条 receipt ≤20 ms"验收 D1，无法证明热点已经消除。
2. **D2a 把 `wait_async_saves` 移到后台线程，会在非主线程发起集合通信，并且与主线程并发操作非线程安全的 `AsyncCallsQueue`。** 方案只要求 `close()` join，时点太晚：`areal_ft1` 会在 `runtime.close()` 之前读取 head、调用 `wait_async_saves`、销毁引擎。
3. **D3 的"分块并行 SHA-256 且输出逐字节不变"做不到。** 检查点正文只有一个 6.9 GB 文件，按文件并行没有收益；对单个文件分块并行，就必须改变摘要定义。

---

## 1. 方案 §0 事实核对

| # | 方案陈述 | 代码或证据 | 判定 |
|---|---|---|---|
| F0-1 | `_io()` 每次调用内含 4 次 `_check` | `rlvr_replay.py:81-90`：执行器外前后各 1 次，`guarded()` 内前后各 1 次 | 正确 |
| F0-2 | 每次 `_check` = `flock` + 完整解析 464 KB 的 `control.json` | `rlvr_replay.py:72-79` → `state.py:152-162`（`_read` 使用 Python 层 `object_pairs_hook`，见 `state.py:42-48`）。p11 最终 control 为 464 KB；本审计宿主机测得解析 3.8 ms | 正确 |
| F0-3 | 每条样本约 6 次 `_io` | 首次生成路径实为 **8 次**：`rlvr_replay.py:243, 271, 282, 290, 300, 251, 255, 258` | **少计 2 次** |
| F0-4 | 合计约 43 次加锁完整解析、1 次整文件重写 | 每条样本在 rollout 侧：`arun_episode` 自身 42 次（`:229` 1 次、`:238` 1 次、8×`_io` 32 次、3×`_put` 内部 `_check` 3 次（`:106`）、独立 `_check` 4 次（`:250, 275, 285, 297`）、`accept_result` 1 次（`state.py:277`））；`IdentityBridge.arun_episode`/`_attach` 另有 4 次（`batch_identity.py:114, 90, 109, 131`）；`TrainingBridge.authorize` 另有 2 次（`training_replay.py:155`、`state.py:240`）。合计 **48 次解析、2 次 control 重写**（`authorize_attempt` `state.py:251` 和 `accept_result` `state.py:291`）。训练侧每条样本还有 6 次（见 F0-7），每步另有 5–6 次 `_head` | **少计，且漏掉 authorize 写入与训练侧** |
| F0-5 | 部分 `_check` 在事件循环线程上同步执行 | 每条样本约 23 次加锁解析在事件循环线程上：`_io` 外层 16 次，加上 `:229, 238, 250, 275, 285, 297` 和 `batch_identity.py:131` | 正确（数量被低估） |
| F0-6 | artifact IO 走单线程 `_io_pool` | `rlvr_replay.py:63`（`max_workers=1`）；identity 也是单线程（`batch_identity.py:52`）。严格评分同样是单 worker（`rlvr_replay.py:54`；`reward_api.py:88-89`），但 **A 臂也一样**，不是 R 特有 | 正确（需注明 A 也串行评分） |
| F0-7 | （剖析 §2）`update_prepared ← 等待评分／receipt 完成（H1，约 14 s）` | 见下方说明 | **归因错误** |
| F0-8 | 能解释每条 0.38–0.49 s 间隔的 1/3–1/2 | R 的评分间隔中位数 **0.47–0.51 s**，A 为 **0.66–0.73 s**（三对 run 的 `score_execution_started` 事件）。R 的 rollout 每条样本并不比 A 慢，拿"间隔"来解释 R 的开销，前提就不成立 | **前提不成立** |
| F0-9 | 30 步时 control 约 1.4 MB | control 大小随 attempts 线性增长（10 步 448 个 attempts，464 KB），外推合理 | 合理 |

**F0-7 说明：**

- **当前批的 receipt 在 `batch_taken` 之前就已全部持久化。** `prepare_batch` 只返回已完成的轨迹（`rl_trainer.py:699`；`workflow_executor.py:1437-1458`）。每条轨迹在返回前都已执行 `accept_result`（`rlvr_replay.py:258` → `batch_identity.py:127-133`）。`Runtime.prepare`（`training_adapter.py:235-261`）不等待任何 receipt。
- **观测器的 `train_batch` 事件在 `runtime.prepare` 之前发出。** `areal_ft1.py:288-289` 先安装 `training_adapter.install`，后安装 `install_hooks`，所以 `areal_pilot_hooks.py:247-253` 是外层包装。因此 `train_batch → update_prepared` 这段就是 `runtime.prepare` 本身的耗时。
- **三对 run 共 27 步的时间线（本审计重算）：**
  - `batch_taken → update_prepared` 为 8–26 s，A 的对应段（`batch_taken → optimizer_start`）约 2 s；
  - `update_prepared` 始终落在**并发的下一批最后一次评分**之后约 0.4–3 s；
  - 当 `train_batch` 本身很晚（16–21 s）时，`prepare` 只用 0.4–0.9 s；
  - 结论：训练主线程上 R 的 CPU 工作本身只有约 1–2 s，被拉长到覆盖整个 rollout 的时长。这是争用的特征（GIL、同一把 `mutation.lock`、fsync），不是"等待 receipt"。

**训练侧漏项（方案完全没有覆盖）：**
- `begin_update` 调用 `validate_rows`（`batch_identity.py:195`、`141-165`）；
- `prepare_train_batch` 再调用一次（`batch_identity.py:226`）；
- `Runtime.prepare` 第三次调用 `_resolve`（`training_adapter.py:240`）；
- 每次 `_resolve` 有 2 次加锁解析（`batch_identity.py:78-84, 90, 109`），还要读 blob、做 SHA、执行 `_load_tensors`/`_validate_tensors`（`rlvr_replay.py:188-206`）；
- 每步合计约 192 次加锁解析、96 次 `_resolve`，全部在训练关键路径上，并且与 rollout 争用。

**解析量总计：** 每步约 32×48 + 192 ≈ 1,730 次，按 3.8 ms 计约 6.6 s 持有 GIL 的 JSON 解析，分布在 4 个线程上。

---

## 2. 阶段 1 各项的正确性评估

### D1.1 control 进程内缓存

**D1.1-a 先改后写会让未持久化的 receipt 可见（必须修改）。**
- 现有写路径都是先原地修改 `control` 对象，再调用 `_write`：`state.py:249-251`、`290-291`、`650-651`。
- 如果缓存交出的是同一个对象，而 `_write` 在 fsync 或 rename 时失败（`test_state.py:299-310` 正好在 head 写入处注入了 `OSError`），缓存中就会留下未持久化的 receipt 或 head。
- 后果：同进程中的 `prepare_generation`（`state.py:510`，比较 `control["accepted"]`）可能消费一个没有落盘的 receipt，违反"receipt 先持久化后使用"。
- 要求：写方在副本上修改（顶层和被改子字典做浅复制即可）；`_write` 成功返回后才替换缓存；锁内出现任何异常都清空缓存。

**D1.1-b 用 stat 身份判定缓存有效不够严格（建议改）。**
- `_write` 每次都创建新 inode，再 `os.replace`（`state.py:84-100`）。
- 旧 inode 释放后可能被复用；Linux 文件时间戳是粗粒度时钟，同一刻度内的多次写入 mtime 可能相同。
- head 更新是等长改写（`state.py:650`），同尺寸改写确实存在。
- 建议：改为"读原始字节并与缓存字节比较"。读 464 KB 远快于 3.8 ms 的解析，而且判定是精确的，方案中"已知限制"一条可以删掉。

**D1.1-c 缓存键和旧 owner 检查。**
- 缓存必须按 root（`state.py:155` 的路径）存放，不能挂在 `Owner` 上。
- epoch/nonce 比较必须针对"当前文件内容"进行（`state.py:160-161`）。同进程中可能有多个 Owner，测试里就有 `acquire_owner` 重入的情况（`test_state.py:95-99`）。
- 缓存只能在持有 flock 时读写。不同 fd 上的 flock 在同一进程的不同线程间同样互斥，这一点成立。

**D1.1-d fork 继承。**
- 评分子进程会继承缓存，但子进程不写 control（`state.py:226` 的 `after_in_child=owner.close`）。无害。

### D1.2 减少检查次数

- 可行。现有合同测试要求在 await 期间 attempt 被替换后，**不能发布** response/reward（`test_rlvr_replay.py:184-201, 305-318`）。执行器内的"操作前检查"保留下来即可满足。
- 注意 `_put` 内部还有一次 `_check`（`rlvr_replay.py:106`），在 `guarded` 中重复执行，应一并去掉。
- 事件循环线程上还有两处加锁：`rlvr_replay.py:229`（读 run 字段，可以放进缓存或在 `_io` 中执行）和 `batch_identity.py:131`（await 之后的 `_current`）。只移除 `_check` 不能让事件循环彻底不再阻塞。
- 语义理由应补充：fencing 的真正依据是 `owner.lock` 的独占（`state.py:190-192`）加上 `accept_result` 在锁内做 CAS（`state.py:278-281`）；artifact 目录按 attempt 隔离（`rlvr_replay.py:236`）。所以 `_check` 只是提前失败的手段，不是安全边界。

### D1.3 accept_result 组提交

- 收益接近零：严格评分是单 worker 串行（`rlvr_replay.py:54`），两次 accept 之间约 0.5 s，一轮里几乎凑不出可合并的写。
- 代价：需要跨协程排队，而且失败时要整批回滚缓存。
- **建议删除，采用保守版。** 真正可以合并的是 `TrainingBridge.authorize`：主线程上单一生产者，每组 k=8 条样本各做 2 次加锁、1 次写（`training_replay.py:151-175`），改成每组 1 次读、1 次写即可，不改变语义。

### D1.4 artifact 顺序发布

- 保留顺序发布可以接受。但每条样本约 13 次 `_publish`（`replay.py:82-101`、`109-114`：response 2 次、reward 2 次、tensor 7 个原始 blob 加 2 次、identity 1 次），每次 2 个 fsync，约 26 次 fsync。
- 这项开销与 control 解析同量级，方案 §0 没有计入。计时探针应单独记录。

### D2a 后台提交 + 屏障

**R-1 集合通信离开主线程（阻断项）。**
- `wait_async_saves` → `maybe_finalize_async_calls(blocking=True)`（`checkpointer.py:685-713`）。AReaL 注释明确写着"Must be called collectively … runs an all_reduce internally"（`checkpointer.py:668-676, 690-707`）。
- Megatron 的 finalize 在 `no_dist=True` 时也会无条件执行一次 `all_reduce(call_idx)`（0.18 版 `async_utils.py:668-700`）。
- 在保存返回到屏障这段时间里，主线程自己也在做集合通信：`_export_and_commit_stats` 中的 `dist.barrier(group=cpu_group)`（`rl_trainer.py:1462`）、stats 的 `all_reduce`（`stats_tracker.py:395`），还有 `current_platform.synchronize()`（`rl_trainer.py:1463`）。
- 同一个进程组被两个线程并发发起集合通信，不在 PyTorch/NCCL 的支持范围内。当前 world size=1（`training_adapter.py:198-202`）时可能碰巧无害，但不能作为设计依据。
- **要求：** finalize（`wait_async_saves`）保留在主线程，放在屏障处执行。后台线程只能做两件事：一是不调用集合通信、只轮询 writer 进程是否退出；二是 writer 退出后对 `.distcp` 做 prehash，并对内存状态做快照哈希。`.metadata` 由主线程 finalize 写出，它和其他小文件留到 commit 时哈希。commit 时现有的 stat 身份复核（`state.py:641-642`）保证预先算好的摘要可以复用。

**R-2 `AsyncCallsQueue` 不是线程安全的（阻断项）。**
- 这个队列内部是 deque 加 `popleft`，此外还有 AReaL 补丁的 `scheduled`/`finalized` 包装（`training_adapter.py:214-227`）。
- 主线程上还有以下调用方：
  - `areal_ft1.py:310` 的收尾 `wait_async_saves`；
  - `PPOTrainer.__exit__` → `close`（`rl_trainer.py:1615-1618`）→ 引擎 `destroy` → `checkpointer.close()`（`megatron_engine.py:854-860`）；
  - F2 钩子（`ft1_fault_hooks.py:121`）。
- `Runtime.close()` 在 `areal_ft1.py:324-325` 的 `finally` 中，晚于以上全部调用。**方案中"close() 必须 join"时点太晚。**
- **要求：** 在 `install()` 中包装 `PPOTrainer.train`，正常返回和异常路径都要先 join committer，再让 `train()` 返回或抛出。F2 钩子在屏障之后执行，队列已空，没有问题。

**R-3 最后一步没有屏障（阻断项）。**
- 最后一步保存之后不会再有 `begin`。`areal_ft1.py:322` 从 `control.json` 读出的 `retained_head` 会是上一代。
- `check_ft1_chain.py:98-99` 要求每一步都有 `committed` 事件，并与 token 链一致。
- R-2 的 `train` 包装可以同时解决这个问题。

**R-4 按值捕获。**
- committer 当前从 `self.update`、`self.generation`、`self.optimizer_stats`、`step_info`、`self.actor.get_version()` 读取数据（`training_adapter.py:291-297, 312`）。`begin` 会覆盖其中几项（`training_adapter.py:232-233`），并以 `self.generation is not None` 为由拒绝开始新一步（`training_adapter.py:230-231`）。
- 要求：`save` 在主线程捕获全部值（包括 policy.json 的内容、`snapshot_id`、`scheduled` 调用 ID、`pending_gate`）；committer 不访问可变的 `self.*`；屏障作为 `begin` 的第一条语句。

**R-5 GPU、RNG 与内存。**
- `generate_state_dict` 必须在主线程上调用，紧跟 `original_dump`。它可能触发优化器分片逻辑乃至集合通信，理由同 R-1。
- RNG 字段本身就是新拷贝（`torch.get_rng_state`、`random.getstate` 等），在保存时刻捕获，比现在"finalize 之后才捕获"更贴近 DCP 保存的值，这一点对正确性有利。
- 模型参数和优化器张量捕获的是**活引用**。在 d1p1t1、不 offload、非 AWEX 共置的前提下，屏障之前没有任何写操作：`update_weights`（`rl_trainer.py:890`）早于保存；`compute_logp` 只读（`:793`）；`ppo_update` → `train_batch` → `prepare` 早于 optimizer。但 `_should_offload_actor` 会在保存之后执行 `_offload_model`（`rl_trainer.py:908`），这会与后台读取冲突。**要求在 `attach` 中断言不 offload、非 colocate**（与 `training_adapter.py:198-202` 同处）。
- 后台线程的 `.cpu()` 走默认流，会与 `compute_logp` 串行执行，`rl_trainer.py:1463` 的同步也会等它。不影响正确性，但会吃掉部分重叠收益。
- 方案提到的"固定内存 D2H"备选需要约 7 GB 主机内存。必须实测宿主 RSS 峰值和 GPU 峰值（仓库中已有 OOM 分析记录）。

**R-6 异常与线程生命周期。**
- committer 若是 daemon 线程，解释器退出时会被中途杀掉；若是非 daemon 线程，退出时会等待它，而此时进程组可能已经销毁。
- 要求：committer 的每个出口都经过 R-2 的 join；异常在屏障、`train` 包装和 `close` 三处都能抛出且只抛一次。
- `Runtime.close()` 必须先 join，再 `owner.close()`，否则 committer 的 `_locked` 会报"inactive owner"（`state.py:153-154`，`training_adapter.py:326-327`）。

**R-7 故障钩子与观测语义。**
- F2 切点在第 2 次成功 optimizer 时触发（`ft1_fault_hooks.py:113-128`），晚于第 1 步的屏障。此时第 0 代已提交、第 1 代只有 intent，与现在相同，F2 在 R 侧的状态**不变**。
- 工程故障脚本在 `Runtime.event` 中挂钩：`areal_training_fault.py:32-52` 用 `optimizer_applied`，在主线程上，没有影响；`areal_midwrite_fault.py:93-127` 用 `async_scheduled`，仍在 `save` 的主线程部分，没有影响。但 `async_finalized`/`committed` 事件改由后台线程发出，子类的 `event` 覆写会在后台线程中运行。
- 观测器的 `recover_handler_dump_returned` / `checkpoint_save_returned`（`areal_pilot_hooks.py:271-283, 297-305`）对 R 不再代表"已提交"。现有检查器用的是 `committed`（`check_ft1_chain.py:98`、`check_training_integration.py:72-81`），不受影响。

**R-8 writer_gate 待决窗口变长。**
- `writer_gate(True)`（`training_adapter.py:276`）到 `writer_gate(False)`（`:300`）这段窗口，原来只在 `save` 内部，现在延长到 committer 完成。
- 这段时间里如果发生非 F2 的崩溃（F1/F4 或真实故障），重启时要走 `verify_cleanup`（`training_adapter.py:166-171`；`writer_recovery.py:39-67`），必须有唯一的 job-lifecycle 清理回执，否则无法接管。
- 需要在方案中说明，并把这一路径纳入崩溃窗口测试。

**R-9 RPO 的说法需要修正。**
- 按步计的 RPO 上界不变：任何时刻最多有 1 次已应用但未提交的更新，恢复时 `select_recovery` 也可能提升已完整的候选代（`state.py:667-682`）。
- 但"最新更新未提交"的状态会在墙钟时间上与 rollout、eval、logp 重叠。对于按事件时点注入的故障（F1/F4），切点时的 head 可能从 k 变成 k−1，冻结版的分类预期需要重新推导。
- 要求把方案的说法改为"RPO 上界不变；暴露窗口变化"。

**TrainingStop（方案 D2a-3）：** 同意，先 join 再抛出（`training_adapter.py:313-314`）。

### D3 摘要计算

- **检查点正文只有一个文件。** 本次 run 某代的 manifest 中，`native/__0_0.distcp` 为 6,917,735,465 字节，其余文件都小于 250 KB。按文件并行没有收益。
- **对单个文件分块并行后，得不到原来的整文件 SHA-256。** 这与方案"输出格式与逐字节结果不变"自相矛盾，还会改变 `manifest["files"]`、恢复时的内容校验（`state.py:431-434`）以及 `check_ft1_chain` 的口径。
- 可以保持格式不变的替代方案（每一项都需单独审计）：
  1. 在 R 专用、由 fork 出的 DCP writer 中流式计算摘要。writer 是 fork 出来的，可以只在 R 进程中打补丁；
  2. writer 退出后立即在后台 prehash（见 R-1），把哈希移出关键路径；
  3. `native_snapshot_r` 的 `normalized` 做了一次整块 `.numpy().tobytes()` 拷贝（`training_adapter.py:32-34`）。可以改为对 memoryview 直接计算 SHA，摘要完全相同。但 `normalized` 由 A 路径观测器共用（`areal_ft1.py:239-242`），必须复制一份 R 专用版本，不能修改共用函数（方案不变量 5）。
- "拆分 snapshot 与 prehash 各自耗时"这一项同意，应列为 D3 的第一步。

---

## 3. 屏障位置

- **必须阻塞的调用：`Runtime.begin` 的第一条语句**，也就是 `MegatronPPOActor.ppo_update` 包装处（`training_adapter.py:381-383`，由 `rl_trainer.py:826` 调用）。在屏障处依次执行：
  1. join prehash 和快照线程；
  2. 主线程执行 `wait_async_saves()`（R-1）；
  3. 完成 commit，或 join 负责 commit 的线程。
- 不能把屏障放在 `prepare`：`begin` 会因 `self.generation` 未清空而报错（`training_adapter.py:230-231`），并覆盖 `self.update`。另外 `ppo_update` 在 `train_batch` 之前不会修改参数（`actor.py:366-465`），所以放在 `begin` 既安全又最早。
- **不需要更早的屏障（例如 rollout 的 authorize 之前）。** 保存返回到 `begin` 之间，读取 committer 相关状态的地方逐一核对如下：
  - `RetainedLoader.__iter__` 读 `loader.consumed`（`training_replay.py:42`）。它运行在主线程的 `prepare_batch` 中（`workflow_executor.py:1437-1456`），而 group_id 按出现序号唯一（`replay.py:219-223`），待决代的组不会被再次产出，所以这次读取是良性的。整体替换集合的引用在 GIL 下是原子的。
  - `TrainingBridge.authorize` 和 rollout 的 `_check`/`accept_result` 都在同一把 `mutation.lock` 内读写 control。commit 在锁内先读后改再写 head（`state.py:613-651`），不会丢失更新；待决代的 attempt 不会被替换，因为 authorize 在本 epoch 内跳过已有的 attempt（`training_replay.py:158-160`）。
  - pin 和 prune 只由 committer 访问。`snapshot()` 和 `_head` 只在屏障之后的 `prepare` 中调用（`training_adapter.py:248-259`）。
  - 例外：R-2 和 R-3 列出的 `train()` 之后的读取与销毁路径，必须另设 join。

---

## 4. 性能估计

**方案数字与剖析不一致。**
- 方案 D2a 写"上一步的 finalize 与哈希约 12–14 s"，但剖析实测 finalize 等待约 10.5 s，快照、哈希和提交约 12.7 s，合计约 23 s，另有 prune 约 1 s。
- 方案假设可以与"A 的等待下一批（约 10 s）"重叠。然而 R 的下一批最后一次评分在批次到手后 12–23 s 就已结束，而保存返回在 21–32 s（见 F0-7 的时间线）。**R 没有等待下一批的空闲时间可以用来重叠。**

**周期模型。** 设：C 为保存返回到提交完成的时间，x 为保存返回到下一次 `begin` 之间的非 R 工作，y 为 `begin` 到 optimizer 的时间，d 为保存同步部分的时间。
- 现行设计：周期 ≈ x + C + y + d。
- D2a：周期 ≈ max(C, x) + y + d。

D2a 只能省下 x，大约 4–6 s（eval、stats、resume、logp、advantage 以及 `pruned → batch_taken` 那一段）。

**阶段 1 估计。** 假设 D1 加训练侧去重能把 y 降到 2–3 s，d ≈ 4.6 s，C ≈ 23–24 s：周期约 30 s，约为 A（21–22 s）的 1.3–1.45 倍。只有把 C 压到约 18 s 以下（即哈希基本移出关键路径，见 D3 的替代方案 1 和 2），才可能接近 1.15 倍。方案给出的 24–28 s 偏乐观，并且依赖一个错误的重叠前提。

**遗漏的热点：**
1. 训练侧重复校验（F0-7）。这是 H1 的真正载体。
2. 争用本身，包括 GIL、flock、fsync。应记录各线程的 `time.thread_time()` 和 flock 等待时间。
3. `prune_generations` 在锁内删除 6.9 GB 文件并做目录 fsync（`state.py:728-751`，事件中约 1 s）。D2a 之后它会阻塞 rollout 事件循环中的 flock。标记已先持久化，删除可以移到锁外。
4. 每步 5–6 次 `_head`（`training_adapter.py:251, 303`；`state.py:474, 614, 702, 729`）。每次读取全部代的 token 和约 150 KB 的 manifest。10 代时实测约 18 ms，开销随代数线性增长。
5. `authorize_attempt` 每条样本各写一次 control（F0-4）。
6. `Runtime.event` 每个事件都做一次 fsync（`training_adapter.py:155-159`），量小，可以忽略。

另外，共用观测器的 `checkpoint_save_returned` 会对 `meta.path` 做全量哈希（`areal_pilot_hooks.py:278`）。在 A 臂中这里要哈希上一代完整的 6.9 GB，这很可能是 A 保存段约 9 s（R 约 4.5 s）的主要来源（推断，未计时）。它由两臂共用，不能改，但做比率解读时要注明。

---

## 5. 结论与必改项

**结论：有条件通过**（D3 按现有写法不通过）。

必改项：
1. **修正 §0 和 H1 归因**：订正计数（8 次 `_io`；每条样本 48 次解析、2 次写；训练侧每步约 192 次解析）。删除"prepare 等待 receipt"的说法；D1 的验收指标改为 `batch_taken → update_prepared`，并同时统计各线程 CPU 时间。
2. **D1 补充训练侧去重**：在同一次更新内，每个 identity 只执行一次 `_resolve`/`validate_rows`，缓存结果供 `begin`、`prepare_train_batch` 和 `prepare` 复用（`batch_identity.py:195, 226`；`training_adapter.py:240`）。authorize 改为按组合并（`training_replay.py:151-175`）。
3. **D1.1 缓存**：写时复制，`_write` 成功后才替换缓存，出现异常就清空；有效性判定改为字节比较；缓存按 root 存放，只在持有 flock 时访问。
4. **D1.2**：同时去掉 `_put` 内部重复的检查，并处理 `rlvr_replay.py:229` 和 `batch_identity.py:131` 两处事件循环上的加锁。**D1.3 删除。**
5. **D2a 的 finalize 留在主线程，放在 `begin` 屏障处执行。** 后台线程只做不调用集合通信的 writer 退出轮询、prehash 和快照哈希；`generate_state_dict` 在主线程调用。
6. **在 `install()` 中包装 `PPOTrainer.train`**：正常返回和异常路径都先 join；`Runtime.close()` 也要 join，并且先于 `owner.close()`。这样覆盖 `areal_ft1.py:310-325` 和 `rl_trainer.py:1615-1618`。
7. **committer 只使用 `save` 在主线程按值捕获的数据**；屏障作为 `begin` 的第一条语句。
8. **`attach` 增加断言**：不 offload actor、非 AWEX colocate。
9. **修改方案表述**：RPO 改为"上界不变、暴露窗口变化"；说明 writer_gate 待决窗口延长对 F1/F4 及真实崩溃的影响。
10. **D3 改写**：删除"大文件分块且输出不变"；先拆分 snapshot 与 prehash 的耗时，再从 D3 的三个格式不变替代方案中选择。
11. **重新登记阶段 1 的性能预期**（约 1.3–1.45 倍，除非 C 下降），并预先写明：达不到 1.15 倍时是否直接进入阶段 2。
12. **prune 的删除移到锁外**（标记先持久化，现有恢复语义允许）。

建议测试（CPU 为主）：
- T1 缓存：在 `accept_result`/`commit` 的 `_write` 中注入失败，之后同进程的 `prepare_generation` 不得看到未持久化的 receipt；同尺寸的外部改写必须被发现。
- T2 现有监督测试全部通过：`test_rlvr_replay.py:184-201, 305-318` 以及 `test_state*.py`。
- T3 线程纪律：用假 `AsyncCallsQueue` 和假 `torch.distributed` 断言 `maybe_finalize_async_calls` 与所有集合通信只在主线程发生。
- T4 收尾：`train()` 返回时，`retained_head` 等于最后一个 `committed`，`check_ft1_chain` 能通过。
- T5 异常路径：保存返回后主线程抛异常、committer 仍阻塞时，join 必须早于引擎 `destroy`，且没有死锁。
- T6 崩溃窗口：方案 §4.2.3 列出的各点，加上"committer 提交期间主线程处于 `prepare_batch`/authorize"和"writer 待决期间被杀，走 `verify_cleanup` 路径"。
- T7 GPU 门控：除时间比外，还要记录 GPU 与宿主内存峰值、`native_state_loaded.exact_match` 以及 F2 分类。

---

*审计者说明：本文中的时间线数字由三对正式无故障 run 的 observer-pilot、observer-ft1 和 rewardtxn 事件重算得到；微测数字（control 解析 3.8 ms，`_head` 等效读取约 18 ms）在宿主机上读取 p11 证据文件测得。F2 在 R 侧的行为未在 GPU 上重跑。*
