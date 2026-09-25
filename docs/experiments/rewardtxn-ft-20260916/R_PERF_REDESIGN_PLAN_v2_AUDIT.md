# R 性能重设计方案 v2 增量审计（2026-09-25）

对象：[R_PERF_REDESIGN_PLAN_v2_20260925.md](R_PERF_REDESIGN_PLAN_v2_20260925.md)（下称 v2）。基线：[R_PERF_REDESIGN_PLAN_AUDIT.md](R_PERF_REDESIGN_PLAN_AUDIT.md)（下称 v1 审计）的 12 条必改项。只读审查，未改代码或已有文件。

Megatron 引用说明：本机只有 megatron-core 0.18 的源码（`/home/lijun/repos/Megatron-LM/.../async_utils.py`），锁文件是 0.17.0，行为需在容器内复核。

## 结论：有条件通过

- v1 的 12 条必改项已基本落实，其中第 10 条按用户决定改成了格式变更路线（B），可以接受。
- 新发现两类问题，实施前必须修改：
  1. **屏障内的执行顺序可能死锁，且"writer 已退出"不能证明写入成功**（下文 N-1、N-2）；
  2. **分块摘要的格式定义有歧义，兼容规则不完整**（下文 N-3 至 N-6）。

## 1. v1 十二条必改项的落实情况

| # | 要求 | v2 位置 | 判定 |
|---|---|---|---|
| 1 | 订正计数与 H1 归因，修改验收指标 | §0 | 已落实 |
| 2 | 训练侧去重；authorize 按组合并 | §1.3、§1.4 | 已落实。补充：`authorize_attempts` 必须对整组逐条做 CAS，要么全部成功要么全部失败，并补一个"组内有一条已是本 epoch 的 attempt"的用例 |
| 3 | control 缓存：写时复制、字节比较、按 root、持锁访问 | §1.1 | 已落实 |
| 4 | 去掉 `_put` 重复检查；事件循环上不再加锁；删除 D1.3 | §1.2、§1.5 | 已落实 |
| 5 | finalize 留在主线程；后台线程不调集合通信；`generate_state_dict` 在主线程 | §2 | 已落实，但屏障内的顺序有问题，见 N-1 |
| 6 | 包装 `train`，正常与异常路径都 join；`close` 先 join 再关 owner | §2 收尾 join | 已落实 |
| 7 | 按值捕获；屏障作为 `begin` 第一条语句 | §2 | 已落实 |
| 8 | 断言不 offload、非 colocate | §2 | 已落实 |
| 9 | 修正 RPO 与 writer_gate 窗口表述 | §2 语义说明 | 已落实 |
| 10 | 重写 D3 | §3 | 改走格式变更路线（用户决定），见 N-3 至 N-6 |
| 11 | 重新登记性能预期，并写明门控规则 | §4、开头的用户决定 | 已落实 |
| 12 | prune 删除移到锁外 | §2 屏障第 4 步 | 已落实 |

## 2. hasher 线程只轮询 writer 退出：能否可靠判定 `.distcp` 已写完？

结论：**可以作为"开始预先哈希"的触发条件，不能作为"写入成功"的依据。** 具体如下。

**前提条件：必须是非持久化 writer。**
- AReaL 构造的是 `AsyncCallsQueue()`，默认 `persistent=False`（async_utils 0.18 版 `:621-627`）。此时每次保存都 fork 一个 `TemporalAsyncCaller.process`（`:237-280`），写完后进程退出。
- 如果将来改用 `PersistentAsyncCaller`（`:349` 起），worker 进程长期存活、永不退出，hasher 会一直等下去。
- `areal_midwrite_fault.py:84-86` 已经把 `queue.persistent` 当作一个可能出现的配置。
- **要求：** 在 `attach` 中断言 `not queue.persistent`。

**判定方法：** 只能读取 `process.exitcode`（`is_alive()` 内部会调用 `waitpid(WNOHANG)`）。
- 这些操作会修改 `multiprocessing.Process` 的状态。在屏障之前，主线程不会碰这个对象，所以这样用可以接受。
- 但要写清楚：这是对 Megatron 私有字段的依赖。

**退出码为 0 不等于写入成功。**
- writer 在子进程里捕获的异常和写入结果，要等主线程 finalize 时才被汇总；`.metadata` 也是在 finalize 时写出的。
- 所以 hasher 算出的摘要只能算**推测值**。提交的权威仍然是：屏障处 `wait_async_saves()` 成功 → commit 时对 stat 身份做复核（`state.py:641-642`）→ 小文件在 commit 时哈希。v2 §2 已经隐含了这个顺序，**但应明文写出**：退出码不为 0，或 finalize 抛出异常时，推测摘要全部作废。

## 3. 新发现的问题

**N-1 屏障内先 join hasher、后 finalize，可能死锁（阻断）。**
- v2 §2 的屏障顺序是：先 join hasher，再 `wait_async_saves()`。
- 如果 writer 子进程要等父进程取走结果（例如经 mp 管道或队列回传）才能退出，那么父进程在 finalize 之前一直不取，writer 就不退出，hasher 永远等不到，屏障就挂住。0.18 源码中结果的回传方式需要到 0.17 容器里核实。
- `areal_midwrite_fault.py:54-55` 会让 writer 无限睡眠，也会触发同样的挂死（F2 类的 SIGKILL 最终能收场，但 CPU 测试会卡住）。
- **修改：** 屏障先调用 `wait_async_saves()`（主线程等 writer 完成并做 finalize）；随后设置一个事件通知 hasher "writer 已完成"；最后再 join hasher。hasher 的等待条件改为"writer 退出 **或** 收到主线程的事件"，并设置超时。

**N-2 writer 与 finalize 失败时的推测摘要。** 见 §2 的"退出码为 0 不等于写入成功"。另外补一个测试：writer 以非 0 码退出，或 finalize 抛出异常时，不产生 token，推测摘要被丢弃。

**N-3 摘要格式的歧义（必须修改）。**
- v2 在分块条目里继续使用 `"sha256"` 这个键名，但它的含义已经变成"块摘要拼接后的摘要"。旧工具读到它会误以为是整文件 SHA-256。
- 修改：改用新键名，例如 `"digest": {"alg": "sha256-chunked-v1", "chunk_bytes": 67108864, "chunks": [...], "root": ...}`，或者在 manifest 顶层写 `digest_schema: 2`，并禁止分块条目出现 `sha256` 键。
- 拼接规则要写死：按原始 32 字节拼接还是按 hex 拼接，是否把 `size` 和 `chunk_bytes` 一起纳入 root。建议纳入，防止相同块序列被解释成不同大小的文件。
- 校验时要求块数等于 ceil(size/chunk_bytes)，除最后一块外每块都是完整大小，`chunk_bytes` 必须等于冻结的常量。
- 需要写明 256 MiB 阈值的比较是 `>` 还是 `>=`，以及 0 字节文件如何处理。

**N-4 版本号放在哪里，兼容范围多大。**
- v2 写"格式带 schema 版本号"，但没说放在哪。当前 manifest 和 token 都是 `schema: 1`（`state.py:528, 645`）。
- 建议：manifest 顶层增加 `digest_schema`，并纳入 `config_sha256` 或 frozen 版本，保证同一个 run 只使用一种格式。同一条链中出现新旧两种格式时，必须拒绝。
- 需要同步修改的地方：
  - `_inventory` 和 `prehash` 缓存的元组结构（`state.py:343-388`）；
  - `_validate_token(content=True)`（`state.py:431-434`）改为按 `digest_schema` 分派；
  - `_prune_marker` 的相等比较（`state.py:401`）和 `_check_files`（仍只检查大小，不受影响）；
  - `select_recovery` 提升候选代时使用的 `prehashed`（`state.py:669-671`）；
  - 验收工具：`check_ft1_chain.py`（整文件 SHA 比对）、`finalize_ft1_fault.py`、`check_training_integration.py`、`oracle.py`、`profile` 等所有读取 `manifest["files"]` 的地方都要搜一遍。
- 旧格式兼容只用于读取和校验历史证据。新 run 不得写出旧格式。

**N-5 块摘要与 `native-state.json`。** 快照哈希（`native-state.json` 的内容）不在分块格式的范围内，格式应保持不变，否则恢复时的 `exact_match` 比较会失去可比性。v2 §2 写了"memoryview 直接计算 SHA、摘要相同"，这一点需要在测试中逐字节对比新旧输出。

**N-6 安全性。** 分块 Merkle 式摘要在抗篡改上与整文件 SHA-256 等价（前提是按 N-3 把大小绑定进 root）。T7 已经要求翻转任一块中 1 字节必须被拒，应再加两种篡改：截断文件、交换两个块的顺序。

## 4. 性能预估（§4）

- 模型 max(C, x) + y + d 与 v1 审计一致。C 取 11–13 s 的前提是：writer 写盘约 10.3 s（由 `async_scheduled` 到 `async_finalized` 测得）加上 8 线程分块哈希 6.9 GB 约 1–2 s（热页缓存），这个前提合理。
- 需要注意的风险：
  1. **y 取 2–3 s 偏乐观。** rollout 恢复得更早，与训练线程的争用可能更多；D1 效果未经实测。
  2. **快照要从 GPU 拷出约 7 GB 并哈希，与 `compute_logp` 以及同步点（`rl_trainer.py:1463`）争用。** 它也在 C 之内，未必能完全藏住。
  3. **屏障内 finalize、commit 和 prune 标记的开销不止 1–2 s。** commit 要遍历并 fsync 所有目录，还要做 `_head`，需实测。
- 登记区间 1.05–1.25 可以接受。低端的前提是 R 变成受 rollout 限制（R 的 rollout 每批约 15–20 s，比 A 约 21 s 快）。门控规则已经预先写定，没有问题。

## 5. 必改项（简短）

1. 屏障顺序改为：先 `wait_async_saves()` → 再通知 hasher → 最后 join；hasher 以"writer 退出 **或** 主线程事件"为等待条件，并设超时（N-1）。
2. `attach` 断言 `not queue.persistent`；写明只读取 `process.exitcode`；推测摘要以 finalize 成功加 stat 复核为准，失败即作废（§2、N-2）。
3. 分块条目改用新键名或 `digest_schema`；写死拼接规则、大小绑定、块数与块长校验、阈值比较方式和空文件处理（N-3）。
4. `digest_schema` 纳入冻结配置；同一条链中混用格式时拒绝；列出并修改所有读取 `manifest.files` 的状态代码和验收工具；旧格式只读（N-4）。
5. `native-state.json` 格式不变，新增 memoryview 与旧 `normalized` 逐字节一致的测试（N-5）。
6. 测试补充：writer 非 0 退出或 finalize 失败时不产生 token；writer 挂起时屏障不死锁（midwrite 场景）；截断和块换序篡改被拒；`authorize_attempts` 整组全成或全败（N-2、N-6、#2）。
7. 在 §4 注明 y、快照拷出和屏障内提交三项的开销待实测。
