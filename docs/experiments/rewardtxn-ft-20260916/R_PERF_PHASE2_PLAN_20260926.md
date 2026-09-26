# R 性能阶段 2 方案：lag=1 流水提交 + 争用削减（2026-09-26，待审计）

触发依据：[门控结果](PERF_GATE_RESULT_20260926.md)，训练循环 R/A = 1.33 > 1.15。用户已于 2026-09-25 预先批准进入阶段 2，并接受 RPO 至多多回退 1 步。本方案**力求在常见故障下不实际增加回退**（见 §2）。

## 0. 门控中剩余开销的归属（每步，R 约比 A 多 5 s）

`batch_taken→update_prepared` 约 9.6 s，其中包含 `begin` 屏障（settle）约 7.5 s：
- finalize 等写盘 4.7 s；
- join hasher 1.8 s（快照在默认流上排在主线程计算之后）；
- prune 0.8 s（锁内删除 6.9 GB）；
- commit 0.2 s。

去掉屏障后剩余约 2 s，与 A 的约 2.4 s 相当。

此外 R 每步"等下一批"为 5.9 s，A 为 0.85 s：
- R 的 rollout I/O 线程 10 步累计 CPU 33.8 s，flock 等待累计 15 s；
- A 的保存段约 8.9 s（R 约 4.0 s），其中很可能包含共用观测器对上一代的全量哈希（推断）。两臂的比率要带着这一点解读。

## 1. 设计：屏障从 `begin(k+1)` 移到 `save(k+1)` 开头

第 k 代的提交最晚在第 k+1 步的 `save` 开头完成（主线程）。此时第 k 代的 DCP 写盘已与第 k+1 步的 rollout、logp、训练整段重叠，finalize 与 hasher 预计已完成，屏障只剩 token 写入等小开销。

### 1.1 状态协议（state.py）

1. **prepare_generation**
   - 允许恰好一个未决前驱 P：P 有 intent，无 token，无 abandoned，且 `P.intent.parent == 已提交 head`。
   - 新 intent 的 parent 写成 `{"generation": P, "intent_sha256": sha(P 的 intent)}`（pending 形式）。
   - 数据谱系、consumed 和"样本不重复"都相对 P 的 intent data 校验：`consumed == P.consumed ∪ samples`，drawn 以 P 的 drawn 为前缀。
   - 已退役逻辑更新（`prior_updates`）同样纳入 P。
   - 有两个未决代时拒绝。
2. **commit_generation**
   - parent 为 pending 形式时，要求 `head.generation == P`，且 P 的 token 的 `intent_sha256` 等于记录值；
   - token 的 `parent` 仍取提交时的 head（`{generation, token_sha256}`），所以 `_chain`/`_head` 这些只看 token 的校验逻辑不变。
3. **select_recovery**
   - 允许至多两个未决代，且二者必须构成链 `c1.parent == head`、`c2.parent → c1`（pending 形式），否则按原规则报错；
   - 按顺序处理：c1 完整则提升；c2 仅在 c1 已提升且自身完整时提升；
   - 任何一环不完整，则该代及其后代都写 abandoned。
4. **不变**：token 链、manifest 格式、prune 标记、pin 语义、`writer.json` 单槽（见 1.3）。

### 1.2 运行时（training_adapter.py）

**提前落证据**：让崩溃时第 k 代可以被恢复提升，而不是被放弃。
- `save(k)` 主线程：
  - 在 `original_dump` 前写 `optimizer` evidence 与 `policy.json`（optimizer 和 scheduler 已成功，内容与现在相同，只是时点提前）；
  - `generate_state_dict` 捕获 → 启动 hasher；
  - 本地 consumed 视图更新为第 k 代 intent 的 consumed（供 `RetainedLoader` 跳过已被未决代占用的组）。
- **finalize 包装**（主线程；不管是 AReaL 的非阻塞 reap、我们的屏障，还是故障钩子调用的 `wait_async_saves`，都经过它）：对应第 k 代的 call 完成 finalize 后，立即写 `finalize` evidence 与 `writer_gate(False)`。
- **hasher**：算出快照后立即发布 `native-state.json`（纯文件 IO，不涉及集合通信），然后做分块 prehash。

**两处屏障**
- **B1，optimizer(k+1) 之前**：只 join hasher 的快照部分，保证第 k 代状态签名在参数被修改前算完。预计通常已完成（窗口为 rollout 加 logp）。若等待明显，再评估改为对 DCP 已暂存的 CPU 副本做哈希。
- **B2，save(k+1) 开头**：`wait_async_saves()`（通常已无待办）→ join hasher → commit(k)（复用 prehash）→ 事件 `committed` → pin；prune 交给后台（§1.4）。

**结束路径**
- `train` 包装的正常与异常出口、`stop_after`：依次执行 B1 与 B2；
- `close`：join hasher，但不提交。

`writer_gate` 仍是单槽：B2 在调度第 k+1 代写盘前完成第 k 代，同一时刻至多一个待决 writer。

### 1.3 故障语义（重点审计）

**F2**（第 2 次 optimizer 之后被杀；钩子在杀之前主线程调用了 `wait_async_saves`）
- 第 0 代此时已有：intent、完整 DCP（含 `.metadata`）、`optimizer`/`finalize` evidence（finalize 包装在钩子的 drain 中写入）、`policy.json`、`native-state.json`（B1 已保证）。只是没有 token。
- 第 1 代只有 intent。
- 恢复：提升第 0 代，放弃第 1 代，得到的 head 与原冻结版相同（原版第 0 代已在 `begin(1)` 提交）。
- 预期 R 仍为 `correct_recovered`，需 GPU 验证。

**一般崩溃（F1/F4/真实）**
- 若第 k 代写盘已 finalize、快照已发布，则恢复时提升，与同步版的 RPO 相同；
- 否则放弃第 k 与第 k+1 代，多回退 1 步（用户已接受的上界）；
- `writer.json` 待决时仍走 `verify_cleanup`。

**检查工具**：`check_ft1_chain` 要求每一步都有 `committed` 事件。提升路径的提交由恢复进程完成（已有机制），需确认工具对"由恢复提升"的代计数正确。

### 1.4 争用削减（与阶段 2 一起实施）

1. **prune 后台化**：B2 之后交给独立线程。标记在锁内持久化，删除在锁外进行（审计 v1#12）；同时至多一个 prune。
2. **rollout I/O 线程 CPU**：先用探针细分 `rlvr.io`（按操作类型），再决定是否把每条样本 8 次 `_io` 合并为较少的批次。本方案只承诺先细分探针，不预设改法。
3. 共用观测器对 A 的全量哈希不改（两臂共用），报告中注明。

## 2. 预期与门控

- 预期 R 每步去掉约 7.5 s 屏障，加上 prune 后台化，训练循环接近 A（登记 1.00–1.15，n=1 波动大）。
- 门控：新冻结，1 对无故障 + 1 对 F2（新种子），判据同前（循环比 ≤1.15；F2 分类不变；`exact_match`）。
- 另加一项：F2 恢复后的 head 必须等于被提升的第 0 代。
- 不达标时报告原因，不再自动进入下一阶段。

## 3. 测试（CPU）

1. **state**：
   - pending 形式 parent 的 prepare/commit；
   - 两个未决代时 prepare 被拒；
   - P 被放弃后 k+1 不可提交；
   - pending parent 的 intent_sha 被篡改时拒绝；
   - 数据谱系相对 P 校验；
   - 恢复时两候选链的四种组合（c1 完整/不完整 × c2 完整/不完整）；
   - 非链状的两个未决代报错；
   - 旧有 SIGKILL 前后 token 测试回归。
2. **运行时（假 actor）**：
   - 提前证据的时点；
   - 由 finalize 包装写 evidence（含"故障钩子式 drain"）；
   - B1 在 optimizer 前 join；B2 在 schedule 前提交；
   - 崩溃窗口模拟：save 后、B1 前、finalize 后 B2 前，均恢复到预期 head；
   - prune 后台线程与恢复并发的安全性。
3. 容器全量回归，与 `240b8d3`/`9825fcd` 基线逐项对照。

## 4. 流程

本方案 → 独立审计 → 用户批准 → 实现（`.bak` 备份）→ CPU 验证 → GPU 门控（约 3–4 GPU·h，另行批准）。
