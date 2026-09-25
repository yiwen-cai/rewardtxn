# R 正常路径性能重设计方案 v1（2026-09-25，待独立审计）

依据：[R_OVERHEAD_PROFILE_20260925.md](R_OVERHEAD_PROFILE_20260925.md)。用户决定（2026-09-25）：可以接受"训练检查点提交比训练晚一步"（RPO 增加至多一步），以换取正常路径接近 A。本方案先尝试**不改变 RPO**的阶段 1，只有阶段 1 不达标时才启用用户已同意的阶段 2。

目标：同版本无故障 run 中，R 训练循环时间不超过 A 的 1.15 倍（阶段 1），不超过 1.05–1.10 倍（阶段 2）；F2 仍 `correct_recovered` 32/32；全部现有 CPU 合同不回退。

## 0. H1 根因（CPU 核实，部分确认）

每条样本在 `TrainingRLVR.arun_episode` 路径上的 control 访问次数（静态计数）：
- `_io()` 每次调用内含 4 次 `_check`；
- 每次 `_check` = `flock` + 完整解析 464 KB 的 `control.json`；
- 每条样本经过约 6 次 `_io`，另有 `_put`、`bind`、`accept_result`；
- 合计约 43 次加锁完整解析、1 次整文件重写。

实测（p11 最终 control.json）：加锁读 3.6 ms，写加 fsync 4.8 ms。模型估计每条样本约 160 ms，32 条约 5.1 s。它能解释观测到的每条 0.38–0.49 s 间隔的约 1/3–1/2。

其余部分的来源待确认，候选有：
- 张量载荷编码／哈希；
- 所有 artifact IO 走单线程 `_io_pool`；
- 部分 `_check` 在 asyncio 事件循环线程上同步执行，同时挡住生成与评分协程；
- 与训练主线程争 GIL。

所以阶段 1 的第一步是加细粒度计时（见 §4）。

另外，control 随 attempts 线性增长，30 步时约 1.4 MB，每次解析的成本会同比放大。

## 1. 阶段 1：不改提交协议、不改 RPO

### D1 receipt 路径（H1，目标每条 ≤20 ms）
1. **control 进程内缓存**：`_locked` 在持有 flock 后先 `fstat` control.json，(ino, mtime_ns, size, ctime_ns) 与缓存一致时复用已解析对象（深拷贝仅在调用方修改时进行），不一致才重新解析。
   - 单 owner 语义不变：epoch/owner_nonce 仍逐次比较，任何外部改写都会改变 stat 身份并触发重读。
   - 已知限制与 L2 相同：同一时钟刻度内同大小的改写无法识别，需审计确认可接受。
2. **去冗余检查**：`_io` 由 4 次 `_check` 改为 2 次（执行器内操作前与操作后）；移除事件循环线程上的同步 `_check`。
   - 语义论证：fencing 需要的是"操作前后 owner/attempt 未变"，而线程外的两次检查与线程内的两次检查读取的是同一个 control。
3. **accept_result 合并写**：同一轮次多个 receipt 在 IO 线程内批量进入一次 control 写（组提交），每条 receipt 仍在 `accept_result` 返回前持久化。
   - 保守版：只做 1+2，不做批量。
4. artifact 的 blob/指针发布仍用单线程顺序执行（不增加并发写，降低审计面）。

### D2a 后台提交＋下一次 prepare 屏障（H2，RPO 不变）
- `Runtime.save` 在同步完成以下事项后返回：
  - `original_dump`（发起异步 DCP）；
  - 在主线程捕获 `generate_state_dict` 的张量引用与 RNG **值拷贝**。
- 由单个 committer 线程完成：`wait_async_saves` → finalize 身份核对 → 快照哈希 → prehash → 发布 native-state/policy → record_evidence → `writer_gate(False)` → `commit_generation` → 更新 `loader.consumed` → pin/prune → `committed` 事件。
- **屏障**：下一次 `Runtime.begin`／`prepare` 必须先 join 上一个 committer，并在其失败时抛出。
  - 因此 `prepare_generation` 看到的仍是"无未决代、parent 等于已提交 head"，state 协议与恢复逻辑**不变**，RPO 不变。
  - 张量在下一次 optimizer 之前不会被修改（optimizer 在 prepare 之后），因此后台哈希读到的是第 k 代状态。
- 需要审计的点：
  1. 后台线程读取 GPU 张量时，主线程同时在做下一批 logp 前向（CUDA 流与同步问题；必要时先在主线程发起 D2H 到固定内存，后台只哈希 CPU 副本）；
  2. writer_gate、subreaper 回执、故障钩子里的"保存返回"时点语义；
  3. `TrainingStop`（stop_after）必须在 join 后才抛出；
  4. `close()` 必须 join。
- 可重叠窗口：约为 A 的"等待下一批"（约 10 s）加上 logp 等计算。上一步的 finalize 与哈希约 12–14 s，预计残余阻塞 1–4 s/步。

### D3 摘要计算（H3）
- `prehash` 按文件（及大文件分块）并行 SHA-256，8 线程（hashlib 释放 GIL）。输出格式与逐字节结果不变，只改执行方式。
- 拆分并记录 `native_snapshot_r` 与 `prehash` 各自耗时，决定是否进一步把快照改为对异步 DCP 暂存的 CPU 副本做哈希。

**阶段 1 预估**：每步从约 53 s 降到约 24–28 s（A 约 21 s），即循环约为 A 的 1.15–1.3 倍。这是估算，以实测为准。

## 2. 阶段 2（仅在阶段 1 未达 1.15 倍时启用；用户已接受 RPO＋1 步）

**D2b 有界流水提交（lag=1）**：允许第 k+1 代在第 k 代未提交时 prepare，其 intent 的 parent 指向待决的第 k 代。第 k+1 代的提交要求第 k 代已提交（链 CAS）。恢复时，未提交的第 k 代连同其后代一起被放弃，回退到第 k−1 代。

这是 state 协议层的修改，需要改：
- `prepare_generation` 的"unresolved generation"与 parent 检查；
- 数据快照谱系（第 k+1 代的 consumed 以第 k 代的待决快照为前缀）；
- writer_gate 改为按代记录；
- `_head` 与恢复选择的孤儿代处理；
- 故障钩子与验收的 RPO 口径。

阶段 2 需另写方案并单独审计。

## 3. 不变量（两阶段都必须保持）

1. 被消费的样本只来自已提交链上的 receipt；未提交代的消费义务在恢复后重新生成或复用。
2. 已提交 token 对应的检查点内容摘要与落盘文件一致；恢复时只对要加载的 head 做内容哈希。
3. 单 owner fencing：stale owner 的任何写入被拒。
4. 已发布的 artifact 不可变；故障切点处不会出现半提交。
5. 两臂共用的观测器与 A 路径代码不改（只改 R 专用代码与 state 内部实现）。

## 4. 验证计划

1. **计时探针（CPU）**：给 `_check`、`_io`、`_put`、`accept_result`、`prehash`、`native_snapshot_r` 加可开关的累计计时（默认关闭，不进正式冻结的判定输出），先在 CPU 离线重放中复现 32 条样本的路径。
2. **CPU 合同**：
   - 宿主 state 测试全过；容器内 `tests/ft` 与当前 HEAD 逐项对照；
   - P2 i09 用例前后一致；
   - 新增后台提交的崩溃窗口测试：finalize 前、哈希中、commit 写 token 前后、下一次 prepare 屏障处被杀，均须恢复到正确 head，且无 `invalid_commit`。
3. **GPU（需预算批准）**：
   - 新冻结，1 对无故障 + 1 对 F2 门控，每对约 1.5–2 GPU·h；
   - 判据：R/A 循环时间比 ≤1.15，F2 两臂分类与本轮一致，磁盘峰值不高于本轮。
   - 达标后再决定是否以新版本扩展 F4/F1。旧冻结的结果不与新版本合并。

## 5. 流程

本方案 → 独立审计 → 用户批准 → 实现（修改前做 `.bak` 备份）→ CPU 验证 → GPU 门控（单独批预算）。
