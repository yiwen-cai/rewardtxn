# 训练内存 / 磁盘 I/O 优化

## 结论

当前训练路径的主要压力来源不是模型参数本身，而是：

1. `sglang_server_concurrency` 默认 512；3 个 rollout engine 最多约 1536 个并发请求。
2. fully-async 的 Python/Ray backlog 会保留大量未消费的 rollout 样本。
3. 旧版 CAS 每写一条 reward 都从头扫描 `rewards.jsonl`，形成 O(N²) 磁盘读取。
4. checkpoint 位于 `/public`，1.5B 单个 checkpoint 约 21G；历史 `runs/` 会持续累积。

## 已落地的默认保护

### 1. 限制 rollout 并发

`scripts/day2_slime_train.sh` 默认改为：

```text
--sglang-server-concurrency 64
```

可通过 `RTX_SGLANG_CONCURRENCY=128` 调高。该参数只限制请求并发，不改变
`global-batch-size`、`rollout-batch-size` 或 `n-samples-per-prompt`。

### 2. 限制 Ray object store 和 spill

默认配置：

```text
object store: 16 GiB
plasma:       /dev/shm
spill/temp:   /rtx-scratch/<exp_id>/ray
```

宿主入口将 `/tmp/rewardtxn`（根盘/本地 NVMe）挂载为 `/rtx-scratch`，避免 Ray
spill 和 CAS 索引落到 `/public` 数据盘。可通过以下变量覆盖：

- `RTX_LOCAL_SCRATCH`
- `RTX_RAY_OBJECT_STORE_MEMORY`（字节）
- `RTX_RAY_TMP_DIR`
- `RTX_RAY_SPILL_DIR`
- `RTX_MAX_TOKENS_PER_GPU`（仅在 GPU 显存仍紧张时调低）
- `RTX_SGLANG_MEM_FRACTION_STATIC`（仅影响 GPU KV cache）

如果 `queue_warm` 持续增长，可设置 `RTX_FULLY_ASYNC=0`，退回 slime 的普通
rollout 编排，避免跨 rollout 的后台结果队列；代价是长尾请求可能降低吞吐。

### 3. CAS 从全量扫描改为索引

`scripts/phase2_seal_rm.py` 使用 SQLite `PRIMARY KEY(logical_id)`：

- 旧 `rewards.jsonl` 只在索引初始化/外部变更时扫描一次；
- 正常去重为索引查询，不再逐条扫描全文；
- group-RM 的一组样本一次事务批量写入；
- duplicate audit 只保留 key、位置、原因和时间，不重复保存 response/label。

索引默认位于 `RTX_CAS_INDEX_DIR`，宿主入口将其放到本地 scratch。JSONL 仍是
可读的审计/恢复证据。

### 4. 滚动保留 checkpoint，限制累计占用

训练脚本默认启动 `scripts/checkpoint_retention.py`，以
`latest_checkpointed_iteration.txt` 作为完成标记，只保留最近 **2 个已完成的
完整 checkpoint**。较新的、尚未成为 latest 的目录绝不会被删除；因此 Phase 3B
仍可从 latest 恢复，并保留一个前序回退点。历史 `manifests/` sidecar 不删除，审计
链不受影响。

可调参数：

- `RTX_CKPT_KEEP=2`：保留数量；设为 `0` 关闭自动清理；
- `RTX_CKPT_RETENTION_INTERVAL=15`：清理轮询秒数；
- `RTX_CKPT_RETENTION_MIN_AGE=30`：删除前最小稳定秒数。

这会把累计 checkpoint 占用从 `N × 单个 checkpoint` 限制到约
`2 × 单个 checkpoint`，但不会减少每次保存本身的写入量。按当前单份约
21--41 GiB 估算，保留上限约 42--82 GiB（保存瞬间可能再有一份临时目录），
而不是随着长程步数线性增长。

### 5. 长程实验的可选 checkpoint 策略

不需要从中途恢复的长程 Clean/Seal 对照可以使用：

```text
RTX_SAVE_INTERVAL=100
RTX_CKPT_KEEP=1
RTX_NO_SAVE_OPTIM=1
```

`--no-save-optim` 会显著减小 checkpoint，但会禁用 optimizer state 恢复。Phase
3B 自动恢复必须保持 `RTX_NO_SAVE_OPTIM=0`；自动恢复入口也会拒绝不安全配置。

### 6. 新实验隔离

`day1_run.sh`、`day2_run.sh`、`phase2_run.sh` 的默认实验 ID 含秒级时间戳；如果
目标 `runs/<exp_id>` 已有内容，宿主入口会拒绝启动，避免旧的 rewards、manifest 或
checkpoint 状态串入新实验。只有明确恢复旧实验时才设置 `RTX_ALLOW_REUSE=1`。

### 7. 实验配置预设 (profile) 与冒烟入口

宿主入口支持 `RTX_PROFILE` 预设，一行切换实验类型（`scripts/experiment_profiles.sh`）：

```text
RTX_PROFILE=smoke      短冒烟: 4 步, save-interval 2, 并发 16, 保留 2 份 ckpt
RTX_PROFILE=phase3b    可恢复: 保留 optimizer state, 保留 2 份 ckpt, save 10
RTX_PROFILE=long       长程不可恢复: --no-save-optim, 保留 1 份 ckpt, save 100
```

冒烟入口（新实验前必跑）：

```bash
bash scripts/smoke_test.sh
```

它会用 `RTX_PROFILE=smoke` 启动 4 步对照实验，验证 Ray/CAS/checkpoint/retention/
日志链路。另可用 `RTX_META_ONLY=1` 只生成 meta.json、不启动容器，用于本地预检配置：

```bash
RTX_PROFILE=phase3b RTX_META_ONLY=1 bash scripts/day2_run.sh none -1 -1 20
```

训练脚本会拒绝 `RTX_NO_SAVE_OPTIM=1` 与 `--load` 同时出现（optimizer state 缺失
会破坏恢复语义）。meta.json 的 `commit_sha` 改为运行时读取当前 git HEAD，不再硬编码。

## 不建议的做法

- 内存紧张时不要开启 `--async-save`：它可能保留额外的 checkpoint buffer。
- 不要为了降低 host I/O 直接把 Ray spill 放进 `/dev/shm`，这会转化为更高的主机内存占用。
- `--sglang-mem-fraction-static` 主要影响 GPU KV cache，不是 host 内存 / 数据盘
  高负载的首要开关。
- 不要降低 global batch 作为第一步，否则会改变训练对照语义。

## 当前验证（未启动训练实验）

已完成：

- Python 源码编译检查；
- 相关宿主入口和自动恢复入口 `bash -n`；
- checkpoint retention 的安全删除/保留测试；
- 单进程 CAS、批量 CAS、4 进程并发 CAS、旧 JSONL 导入测试；
- 并发测试确认同一 logical ID 恰好写入一条 reward 记录。

本次修改没有启动训练实验。历史 checkpoint payload 已按确认删除，因此现有完整
`phase3_regress.sh` 的离线结果为 8/9：CAS 相关检查全部通过；唯一失败是
`manifest_audit` 找不到已删除的 checkpoint 目录（保留的 sidecar/报告仍在）。如需
重新通过该门禁，需要用新实验重新生成 checkpoint 证据。
