# Day 3 实验汇总 — Queue 消费与 Learner 崩溃窗口探针 (Q0/Q1, B3/B4 对照)

日期: 2026-08-25 | 栈: TransferQueue release/v0.1.10 (8497a52a) + ray 2.58 | 实验: p1-tq-Q0Q1-s42-20260825

## 源码级事实 (controller.py)

1. `get_meta` (fetch 模式) 在返回元数据时**立即调用 `mark_consumed(task, idx)`** 标记消费
2. `mark_consumed` 置 consumption_status[idx]=1, **不可逆**
3. **无任何 timeout / reclaim / ACK 重放机制** (仅 ZMQ 请求超时, 非消费恢复)

=> "get_meta 标记已消费 -> Learner 崩溃 -> 数据永久丢失" 从源码逻辑上必然成立

## 实测证据 (tq_crash_probe.json)

### 场景 A (Q0: get_meta 后崩溃) —— 核心结果
```
producer 写入 32 样本 (indexes 0-31)
consumer1: get_meta -> 拿到 32 条 -> SIGKILL (未 get_data / 未 clear)
consumer2 (重启): get_meta -> RuntimeError: Timeout waiting for sufficient data.
                Required: 32, Available: 0
=> 32 条样本永久丢失: 队列状态 = 已消费, 实际从未被训练
=> Q0 静默丢失窗口确认
```

### 场景 B (正常消费对照)
get_meta -> get_data -> clear_samples 完整消费后, 重取为空 (预期, 非丢失)

### 场景 C (B3 幂等去重边界)
- 同 task 重复请求: 不重复分发已标记样本 (按 task 消费状态去重 ✓)
- 但: **幂等去重无法覆盖 "已标记消费未训练" 窗口** (场景 A)
- 重复投递产生新索引, 去重不识别投递源

## 与文档 Day 3 判定指标的对应

| 文档要求 | 结果 |
|---|---|
| 是否存在 "队列显示已消费, 但 Optimizer 从未执行且未持久化" 的静默丢失 | ✅ **存在且已实证** (场景 A) |
| 重启后的 Consumer 能否重新获取该数据 | ❌ 不能 (Available: 0) |
| 是否导致该 Batch 永久丢失 | ✅ 是 (无重放/恢复机制) |

## B3/B4 机制对比结论 (Day 5 矩阵输入)

- **B4 (Reserve/Occupy/Consume)**: get_meta 即 Occupy/Consume 语义, 无超时回滚 ->
  Q0 窗口存在, 单独无法达到 exactly-once
- **B3 (幂等去重)**: 仅防同 task 重复消费, 对消费-训练崩溃窗口无感知

## L0 探针 (Learner 崩溃窗口, p1-slime-L0-K8-s42-20260825)

注入: 训练至 step 7-8 时 kill -9 MegatronTrainRayActor (trainer, 容器内 PID)

| 观察项 | 结果 |
|---|---|
| ray 自动重启 actor | 否 (ActorDiedError) |
| 训练结果 | Job failed (exit 1), 容器 Exited(1) |
| checkpoint 恢复点 | 无 (10 步训练 save-interval 10, 未到保存点; checkpoints/ 为空) |
| 恢复行为 | 无自动恢复, 8 步训练成果(权重更新)全部丢失, 需人工重启 |

文档 5.3 L0 期望 (丢弃当前梯度, 从 Cs 重做) vs 实际 (无自动重启/无恢复点/训练失败) ->
**Learner 崩溃窗口实证: 队列已产出数据 + 训练未提交 + 崩溃后成果丢失**

## 产物

```
runs/p1-tq-Q0Q1-s42-20260825/     # Q0/Q1/B3/B4 队列侧 (meta.json + tq_crash_probe.json)
runs/p1-slime-L0-K8-s42-20260825/ # L0 训练侧 (meta.json + l0_result.json + logs)
```

## 下一步 (Day 4)

L2/L3/C1/C2: slime 训练侧 Optimizer<->Checkpoint<->ACK 窗口 (杀 trainer rank /
checkpoint 落盘 vs ACK 丢失), 补 B5 (per-step checkpoint 全量重算开销实测)
