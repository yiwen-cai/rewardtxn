# R 正常路径性能重设计方案 v2（2026-09-25）

取代 [v1](R_PERF_REDESIGN_PLAN_20260925.md)，逐条落实 [独立审计](R_PERF_REDESIGN_PLAN_AUDIT.md) 的 12 条必改项（下文以【审计#n】标注）。

**用户决定（2026-09-25）：**
- 先做"修正后的阶段 1 + B（分块摘要格式）"；
- GPU 门控实测 R/A 循环时间比 > 1.15 时，**直接进入阶段 2**（lag=1 流水提交，RPO+1 步，用户已接受），不再另行讨论【审计#11】。

## 0. 已订正的事实与归因【审计#1】

**开销计数**
- rollout 侧：每条样本 8 次 `_io`，约 48 次加锁完整解析 control，2 次 control 重写（`authorize_attempt` 与 `accept_result`）。
- 训练侧：每步约 192 次加锁解析、96 次 `_resolve`（同一 identity 分别在 `begin_update`、`prepare_train_batch`、`Runtime.prepare` 中各解析一次）。
- 每步合计约 1,730 次解析，约 6.6 s 持 GIL。
- 每条样本另有约 13 次 `_publish`，约 26 次 fsync。

**H1 真正原因**
- 是训练主线程上的 R 校验与同进程并发 rollout 争用 GIL、`mutation.lock` 和 fsync，把 `batch_taken → update_prepared` 从本应的 1–2 s 拉长到 8–26 s。
- 不是 prepare 等 receipt：当前批的 receipt 在 `batch_taken` 前已全部持久化。
- 严格评分单 worker 串行是 A、R 共有的，不是 R 特有的开销。

**D1 验收指标**：`batch_taken → update_prepared` 的中位数，加上各线程 `thread_time` 与 flock 等待。

**比率解读注意**：共用观测器在 `checkpoint_save_returned` 时对 A 的上一代做全量哈希，这很可能是 A 保存段约 9 s 的主要来源（推断）。两臂共用这个观测器，不做修改，但解读 R/A 比率时注明。

## 1. D1 receipt／校验路径

1. **control 缓存**【审计#3】
   - 缓存按 state root 存放，只在持有 flock 时访问。
   - 有效性用"原始字节与缓存字节比较"判定（读 464 KB 远快于解析）。
   - epoch/nonce 每次都和当前内容比较。
   - 写方在副本上修改，`_write` 成功后才替换缓存；锁内任何异常都清空缓存。
2. **去冗余检查**【审计#4】
   - `_io` 只保留执行器内"操作前"与"操作后"两次检查；
   - 去掉 `_put` 内部的重复检查；
   - `rlvr_replay.py:229`（读 run 字段）与 `batch_identity.py:131`（`_current`）挪进 `_io` 执行，事件循环线程上不再同步加锁。
   - 语义依据：fencing 靠 `owner.lock` 独占加上 `accept_result` 锁内 CAS；`_check` 只用于提前失败。
3. **训练侧去重**【审计#2】
   - 同一次更新内，每个 identity 只做一次 `_resolve`/`validate_rows`，结果按 (identity, attempt) 缓存，供 `begin_update`、`prepare_train_batch`、`Runtime.prepare` 复用；
   - 下一次 `begin` 时清空缓存；
   - `prepare_train_batch` 发布 intent 前仍在锁内核对 attempt 未变。
4. **authorize 按组合并**【审计#2】：`TrainingBridge.authorize` 每组 k 条样本改为 1 次读、1 次写（新增 `state.authorize_attempts` 批量版），逐条语义与原来相同。
5. 删除 v1 的 D1.3（accept 组提交）。artifact 仍顺序发布。

## 2. D2a 后台提交加 `begin` 屏障（state 协议不变）

**主线程在 `save` 中同步完成**【审计#5、#7】
- `writer_gate(True)`；
- `original_dump`（发起异步 DCP）；
- `generate_state_dict`，得到张量活引用与 RNG 值拷贝；
- 按值捕获 committer 需要的全部数据：generation、snapshot_id、optimizer_stats、policy.json 内容、scheduled 调用 ID、step_info；
- 清空 `self.generation`/`self.update`。

**后台线程（hasher）只做三件事，不调用任何集合通信、不碰 `AsyncCallsQueue`**
1. 轮询 fork 出的 writer 进程退出（不调用 finalize）；
2. writer 退出后，对 `native/*.distcp` 做分块摘要（见 §3）；
3. 对捕获的状态做 R 专用快照哈希：memoryview 直接计算 SHA，不再 `.tobytes()` 拷贝；不修改共用的 `normalized`。

**`begin` 屏障**：作为 `Runtime.begin` 的第一条语句，由主线程执行：
1. join hasher，有异常则抛出；
2. `wait_async_saves()`（finalize，集合通信留在主线程），核对 scheduled ⊆ finalized；
3. 发布 native-state/policy，record_evidence，`writer_gate(False)`，commit（对 `.metadata` 等小文件在 commit 时哈希，已有的 stat 身份复核保证预算摘要可复用），更新 `loader.consumed`，pin；
4. prune：标记在锁内持久化，大文件删除移到锁外【审计#12】。

**收尾 join**【审计#6】
- `install()` 包装 `PPOTrainer.train`：正常返回与异常路径都先执行屏障（最后一代也完成提交），再返回或抛出。这样覆盖 `areal_ft1.py:310-325` 的 `wait_async_saves`、`retained_head` 读取，以及 `rl_trainer.py:1615` 的 destroy。
- `Runtime.close()` 先 join 再 `owner.close()`。
- `TrainingStop` 在屏障完成后才抛出。
- 异常只抛一次。

**前提断言**【审计#8】：`attach` 断言不 offload actor、非 colocate（与 d1p1t1 断言放在一起）。

**语义说明**【审计#9】
- 按步计的 RPO 上界不变：任何时刻最多 1 次已应用未提交的更新。
- 暴露窗口在墙钟上变长（与 eval、logp、rollout 重叠），writer_gate 待决窗口也随之变长。F1/F4 与真实崩溃会更常走 `verify_cleanup` 路径，扩展到 F1/F4 前要重新推导冻结分类预期。
- F2 在 R 侧状态不变：切点在第 2 次 optimizer，晚于第 1 步的屏障。
- `async_finalized`/`committed` 事件仍由主线程在屏障内发出。

## 3. D3：分块摘要（B，格式变更）【审计#10】

**第一步：计时。** 先拆分 `native_snapshot_r` 与 `prehash` 各自的耗时。

**新摘要格式**
- 对大于 256 MiB 的文件，manifest 记录 `{"size", "chunk_bytes": 64 MiB, "chunks": [sha256...], "sha256": sha256(拼接的块摘要)}`；小文件仍记整文件 SHA-256。
- 用 8 线程并行计算。
- 格式带 schema 版本号，旧 token 仍按旧格式校验（兼容历史证据）。

**需要同步修改的地方**
- `state` 的 `_complete`/prehash/缓存复用；
- 恢复时对 head 的内容校验（`state.py:431-434`）；
- `check_ft1_chain` 等验收工具的口径；
- CPU 测试加篡改用例：任一块内翻转 1 字节必须被拒。

摘要在 hasher 线程里算，不在关键路径上。

## 4. 性能预期（重新登记）

**周期模型**：周期 ≈ max(C, x) + y + d。
- C：保存返回到可提交的时间，包括 DCP 写盘约 10 s；
- x：保存返回到下一次 begin 之间的非 R 工作，约 4–6 s；
- y：begin 到 optimizer；
- d：保存的同步部分。

**阶段 1 加 B 的预估**
- D1 后，y 约 2–3 s；
- 摘要与快照并行后，C 主要由 DCP 写盘决定，约 11–13 s；
- 预计周期约 18–22 s，另加屏障处 finalize/commit 约 1–2 s。
- 登记值：R/A 约 1.05–1.25（A 约 21 s），不确定性主要来自 DCP 写盘与争用。

**门控规则（预先写定）**：R/A 循环时间比 ≤ 1.15 → 停在本阶段；> 1.15 → 进入阶段 2。

## 5. 不变量（同 v1 §3）

另外两条：
- 未持久化的 control 修改对同进程不可见；
- 集合通信与 `AsyncCallsQueue` 只在主线程上调用。

## 6. 测试

- **T1** 缓存写失败注入：`accept_result`/`commit` 的 `_write` 失败后，同进程 `prepare_generation` 看不到该 receipt；同尺寸外部改写必须被发现。
- **T2** 现有 `test_state*`、`test_rlvr_replay`（attempt 替换后禁止发布）、P2 i09 全部与 HEAD 逐项一致。
- **T3** 线程纪律：用假 `AsyncCallsQueue` 与假 `torch.distributed` 断言 finalize 与集合通信只在主线程发生。
- **T4** 收尾：`train()` 返回时 `retained_head` 等于最后一个 committed，`check_ft1_chain` 通过。
- **T5** 异常路径：保存后主线程抛异常且 hasher 阻塞时，join 早于 destroy，无死锁。
- **T6** 崩溃窗口：hasher 运行中、屏障 finalize 前后、commit token 写入前后、prune 标记后删除前、writer 待决期间被杀（走 `verify_cleanup`），均恢复到正确 head，无 `invalid_commit`。
- **T7** 分块摘要：篡改任一块被拒；旧格式 token 仍可校验。
- **T8** GPU 门控（需批预算）：新冻结，1 对无故障加 1 对 F2；记录循环时间比、GPU 与宿主内存峰值、`exact_match`、F2 分类、`batch_taken → update_prepared`、各线程 CPU 时间。

## 7. 实施顺序

1. 计时探针（默认关闭）；
2. D1；
3. D3 分块摘要；
4. D2a；

每步后跑 CPU 回归。修改前按惯例做 `.bak` 备份。完成后请用户批准 GPU 门控预算。

## 8. v2.1 修订：落实增量审计的 7 条必改项（[v2 审计](R_PERF_REDESIGN_PLAN_v2_AUDIT.md)）

1. **屏障顺序**：改为 `wait_async_saves()`（finalize）→ 通知 hasher → join。hasher 等待"writer 退出或主线程通知"，并设超时；writer 挂起时屏障不能死锁。
2. **断言与作废规则**：`attach` 断言 `not queue.persistent`；hasher 只读 `process.exitcode`。writer 退出码非 0、或 finalize 失败时，推测摘要全部作废，不产生 token。推测摘要最终以 finalize 成功加 commit 时的 stat 复核为准。
3. **分块格式**：
   - 新条目只用新键名：`{"digest_schema": "sha256-chunked-v1", "size", "chunk_bytes": 67108864, "chunk_sha256": [...], "chunked_digest"}`；
   - `chunked_digest = sha256(b"sha256-chunked-v1\0" + size(8 字节大端) + chunk_bytes(8 字节大端) + 各块 32 字节原始摘要依次拼接)`；
   - 校验块数 = ceil(size/chunk_bytes)，末块长度按 size 推出；
   - 阈值为 `size >= 256 MiB` 才分块，其余包括空文件仍用整文件 `sha256`。
4. **版本号纳入冻结配置**：同一条链上混用新旧格式一律拒绝。`_inventory`/prehash 缓存、`_validate_token`、`_prune_marker`、`select_recovery` 的 prehashed、`check_ft1_chain`、`finalize_ft1_fault`、`check_training_integration`、`oracle` 同步修改；旧格式只读。
5. **`native-state.json` 格式不变**：新增测试，验证 R 专用的 memoryview 写法与共用 `normalized` 的输出逐字节一致。
6. **补测试**：writer 非 0 退出或 finalize 失败时无 token；writer 挂起时屏障超时报错而不死锁；截断文件和块换序两种篡改被拒；`authorize_attempts` 整组要么全部成功、要么全部失败（组内任一失败时 control 不变）。
7. **§4 待实测项**：begin→optimizer 的 2–3 s、约 7 GB 快照 D2H 与 logp 的争用、屏障内 finalize 加 commit 的实际耗时，都在 T8 中单独记录；登记区间 1.05–1.25 保持不变。
