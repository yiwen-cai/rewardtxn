# R 正常路径性能重设计：阶段 1 + B 实现与 CPU 验证（2026-09-26）

依据 [方案 v2/v2.1](R_PERF_REDESIGN_PLAN_v2_20260925.md)（已通过两轮[审计](R_PERF_REDESIGN_PLAN_v2_AUDIT.md)，用户 2026-09-25 批准实现）。修改前备份为 `*.bak-20260926-perf`。本轮只用 CPU，未启动 GPU。**代码改动结束了 2026-09-24 的冻结版本**，旧正式结果不与新版本合并。

## 1. 改动

| 项 | 文件 | 内容 |
|---|---|---|
| 探针 | `scripts/ft/perf_probe.py`（新） | `RTX_PERF_PROBE=<路径>` 时累计各段墙钟与线程 CPU，退出时写 `<路径>.<pid>.json`。默认关闭，不影响状态或判定 |
| D1.1 control 缓存 | `state._locked` | 按 state root 缓存，按原始字节判定是否有效。只读段共享已解析对象；写段（`authorize_attempt(s)`、`accept_result`、`commit_generation`）拿私有解析，退出时清缓存；锁内任何异常都清缓存 |
| D1.2 去冗余检查 | `rlvr_replay` | `_io` 只保留 I/O 线程上的前后两次检查；删除 `_put` 内部和事件循环上的独立 `_check`；`run` 字段读取移到 I/O 线程；await 后重新检查重复样本 |
| D1.2 | `batch_identity.arun_episode` | await 之后的 `_current` 移到 identity I/O 线程 |
| D1.3 训练侧去重 | `batch_identity.resolve_verified`、`training_adapter.prepare` | 同一更新内每个 identity 只完整解析一次，之后每次使用仍调用 `_current` 复核授权；`begin_update` 时清空 |
| D1.4 authorize 合并 | `state.authorize_attempts`（新）、`TrainingBridge.authorize` | 每组 k 条改为 1 次读、1 次全成或全败的写 |
| D3 分块摘要 | `state`（`_hash_chunks`、`_entry_chunked`、`_manifest_schema`、`_inventory(chunked=…, only=…)`） | 按 v2.1 §8.3 实现：`sha256-chunked-v1`、64 MiB 块、≥256 MiB 才分块、8 线程；旧格式只读校验；同链混用新旧格式一律拒绝 |
| D3 | `tests/ft/offline_generation.py` | 独立实现的双格式校验（`file_matches`），`check_ft1_chain`／`check_training_*` 经 `check_files` 使用 |
| D2a 后台提交 | `training_adapter` | 见第 2 节 |
| 快照 | `training_adapter._normalized_leaf_r` | R 专用：直接对缓冲区做 SHA，不再 `tobytes()` 拷贝；共用的 `normalized` 不变 |
| 测试 | `test_state_efficiency.py` | 计数替身透传新关键字参数（行为不变） |

## 2. D2a 后台提交

**`save`（主线程）**
- 步骤：`writer_gate(True)` → `original_dump` → 记录 writer pid → `generate_state_dict` → 按值捕获提交所需的全部数据 → 启动 `_Hasher` → 清空 `generation`/`update` 后返回。
- 若遇到 `stop_after`，先执行屏障再抛 `TrainingStop`。

**`_Hasher` 线程**
- 对已捕获的状态计算快照签名。
- 等 writer 退出后，对 `native/*.distcp` 做分块 prehash。判断退出只读 `/proc/<pid>/stat`，从不调用 waitpid。
- 它不做集合通信，不访问 `AsyncCallsQueue`。任何失败都只让屏障改为同步重算。

**`settle` 屏障（主线程）**
- 触发点：`begin` 的第一条语句、`PPOTrainer.train` 包装的正常返回与异常路径、`stop_after`。
- 顺序：`wait_async_saves()` → 核对 scheduled ⊆ finalized → 通知 hasher 并 join（900 s 上限）→ 发布 native-state/policy → 两份 evidence → `writer_gate(False)` → commit（推测摘要只在 stat 身份不变时复用）→ 更新 `loader.consumed` → `committed` 事件 → pin → prune。

**其他**
- `close()` 先 `abandon_pending()`（只 join hasher，不提交；这一代留给恢复逻辑处理），再关 owner。
- 断言：`attach` 拒绝持久化 writer 队列；`train` 包装拒绝 actor offload 与 AWEX 共置。
- 状态协议不变：prepare 时无未决代，parent 等于已提交 head。RPO 上界不变，暴露窗口变长（见方案 §2）。

**未做**：v2 §1 的"prune 删除移到锁外"（审计#12）。prune 仍在锁内，但现在由屏障调用，在关键路径上约 1 s；GPU 门控时用探针实测后再决定。

## 3. CPU 验证

| 检查 | 结果 |
|---|---|
| 宿主 `test_state*`（含新增 `test_state_perf` 14 项） | 53/53 通过 |
| 容器全部 `tests/ft/test_*.py`，与 HEAD 基线（`git worktree` 快照、同镜像、同环境变量）逐测试对照 | 38 个原有模块结果逐项一致，差异只有日志时间戳和新增测试。基线里已存在的环境性失败（容器内 `test_state` 2F/4E、`test_p2_schedule_audit` 11E、`test_p2_schedules` 1E）新旧相同，与本改动无关 |
| 容器 `test_deferred_commit`（新，8 项）+ `test_native_snapshot_r` + `test_state_perf` | 24/24 通过 |
| 分块哈希实测（p06 保留的 6.4 GB 真实分片，热缓存） | 8 线程 1.17 s（原整文件单线程 7.3 s）；16 线程 0.6 s |

**新测试覆盖的审计项**
- T1：缓存写失败不可见、同尺寸外部改写能被发现、stale owner 被拒、读段异常清缓存；
- `authorize_attempts` 全成或全败；
- T3：`generate_state_dict` 与 finalize 只在主线程执行；
- 推测摘要复用，文件被改后重新哈希；
- finalize 失败不产生 token；
- writer 不退出时走通知路径，不死锁；
- `begin` 先执行屏障；
- 放弃未决代后恢复到上一代；
- 分块摘要：翻转一字节、截断、块换序均被拒；旧格式可校验；混链被拒；离线独立校验一致；
- 快照 memoryview 写法与共用 `normalized` 输出一致（含非连续数组）。

**覆盖缺口（需在 GPU 门控中确认）**
- T6 中"屏障内被 SIGKILL"的各窗口，落盘状态与同步版相同（已有 `test_state` 的 SIGKILL 前后 token 测试覆盖）。
- "writer 待决期间被杀 → `verify_cleanup`"沿用 `test_writer_recovery`，未新增端到端测试。
- 真实 Megatron 队列上的 writer pid 取值路径（`async_calls[-1].async_caller.process.pid`，0.17.0 源码中为 `TemporalAsyncCaller.process`）尚未在 GPU 上验证；取不到时退化为屏障通知路径，正确性不受影响。

## 4. 下一步

GPU 门控（需用户批准预算）：
- 新冻结，1 对无故障 + 1 对 F2，开启探针；
- 判据：R/A 循环时间比 ≤1.15，F2 分类与原冻结版本一致，`native_state_loaded.exact_match`，内存与磁盘峰值；
- 比值 >1.15 时按用户预先决定直接进入阶段 2。
