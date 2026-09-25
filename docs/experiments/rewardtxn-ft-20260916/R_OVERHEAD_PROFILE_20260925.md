# R 正常路径开销剖析（2026-09-25，仅 CPU／既有证据）

数据：3 对正式无故障 run（`formal-nofault-p1{1,2,3}`）。工具：`uv run python tests/ft/profile_r_overhead.py`（启动器日志为 UTC；训练器事件取 observer-pilot，R 事务事件取 `rewardtxn/events.jsonl`，评分事件取 observer-ft1）。未启动 GPU。

## 1. 阶段拆分（中位数，秒）

| 阶段 | A | R | R−A |
|---|---:|---:|---:|
| 启动→trainer | 71.6 | 74.5 | +3 |
| trainer→首批 | 144.1 | 153.0 | +9 |
| **训练循环（10 步）** | **211.3** | **507.3** | **+296** |
| 循环后收尾 | 64.9 | 64.6 | 0 |
| 启动器外（容器、验收） | 39.8 | 40.7 | +1 |
| 总墙钟 | 529.5 | 846.9 | +317 |

结论：R 的额外开销约 93% 在训练循环里，每步多约 30 s（A 每步约 21 s，R 约 53 s）。启动、收尾、验收的差异可以忽略。

## 2. R 单步时间线（p11 第 5–6 步，相对 batch_taken）

```
 0.00 batch_taken
      下一批生成 0.3–4.2 s 完成；32 次评分串行执行，每次约 0.11 s，间隔中位 0.38 s，持续到 17.0 s
12.48 train_batch
18.35 update_prepared        ← 等待评分／receipt 完成（热点 H1，约 14 s）
19.58 optimizer_end
19.71 dump_start → 24.13 dump_returned（发起异步 DCP）
34.67 async_finalized        ← wait_async_saves() 同步等待（热点 H2，约 10.5 s；A 这段与下一批 rollout 重叠）
47.40 committed              ← native_snapshot_r ∥ prehash + commit（热点 H3，约 12.7 s）
48.32 pruned → 51.94 下一 batch_taken（约 4.5 s）
```

## 3. CPU 微测（p06 R 全量保留数据）

- `state.prehash` 单代 6.5 GB：冷缓存 8.7 s，热缓存 7.3 s（单线程 SHA-256，约 0.9 GB/s）。它和 `native_snapshot_r` 并发执行，是 H3 的主体之一。
- `control.json`（463 KB，448 attempts）：读约 4 ms，写加 fsync 约 5 ms。10 步规模下不是热点，旧假设"control 线性增长"不能解释本轮开销，30 步时仍可能放大。

## 4. 热点与候选方向（尚未实施，需按流程先审计再批准）

| # | 热点 | 每步 | 候选方向 | 语义风险 |
|---|---|---:|---|---|
| H1 | 评分串行，prepare 等全部 receipt | ~14 s | 先 CPU 复现：确认 0.38 s 间隔来自 fork 型严格评分、逐样本 `_locked`／fsync 发布，还是与 trainer 争用；然后并行评分，或合并逐样本的持久化写 | 低到中：receipt 仍须在 prepare 前持久化 |
| H2 | 同步等待异步 DCP 落盘 | ~10.5 s | 提交延后到后台线程，与下一批 rollout 重叠；下一次 `prepare` 前必须拿到 commit token，消费权限不提前 | 中：改变 commit 时点，需重做故障切点与验收合同测试 |
| H3 | 快照与全量内容哈希 | ~12.7 s | 分块并行哈希，或写入时流式计算摘要；拆分 snapshot 与 prehash 各自耗时 | 低：摘要内容不变 |

理论上限：三项全部消除后 R 每步约 16 s 可压缩，循环约 +296 s 可减至 +100 s 以内。实际值要实施后测量，不预设。

## 5. 下一步

1. CPU 细剖 H1（在进程内给评分和持久化路径加计时，离线重放一批 32 条），并拆分 H3 两个子项的耗时。
2. 写修复方案 → 独立审计 → 用户批准 → 实现 → CPU 合同回归。
3. 代码改动会结束当前冻结版本。优化版需要新冻结，并跑少量无故障与 F2 门控（需 GPU 预算批准），旧结果不与新版本合并。
