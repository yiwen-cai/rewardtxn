# Day 4 实验汇总 — Optimizer ↔ Checkpoint ↔ ACK 窗口 (L2/L3/C1/C2 + B5)

日期: 2026-08-26 | 栈: slime v0.3.1 | 配置: Qwen2.5-1.5B + GSM8K, 4 卡 (0,1,2,5), save-interval 4

## 方案修正说明 (用户授权)

- 文档 L2 定义为 "Optimizer 执行中单卡崩溃", 但我们为单 actor (TP=1) 架构,
  无多 rank DP -> L2 修正为 "Optimizer 阶段 kill trainer actor (MegatronTrainRayActor)"
- 文档 Day 4 "Backward 期间杀 Rank 1" 与 L2 定义错位 (Backward 属 L1), 一并修正

## 实验 A: L2_rank_kill (p1-slime-L2-K8-s42-20260825)

训练至 step 8 时 kill -9 trainer actor (Optimizer 阶段), 观察与恢复验证:

| 观察项 | 结果 |
|---|---|
| 自动重启 | 否 (ActorDiedError, Job failed exit 1) |
| 默认重启 (无 --load) | **不加载 checkpoint, 从 step 0 重新训练** (10 步成果丢失) |
| 显式 --load (num_rollout=16) | AssertionError: scheduler num_steps 512 vs checkpoint 256 不匹配, 无法启动 |
| 显式 --load (num_rollout=8) | 成功加载 iter 7, 但 rollout 轮次已耗尽, **0 个新 step 直接结束** |

**核心发现: 恢复路径双重死锁**
1. megatron OptimizerParamScheduler 要求恢复配置的 num_steps 与 checkpoint **精确相等**
   (num_steps = train_iters × global_batch = 保存时已消耗样本数)
   => 恢复必须用 num_rollout == 保存时已跑步数, 否则 AssertionError
2. 即使配置匹配加载成功, fully-async 的 rollout 轮次 (num_rollout) 无法续接,
   恢复后**无数据可训**, 训练直接结束

=> 文档验收要求 "全 Trainer 回滚至上一个 Checkpoint, 严禁局部继续" 无法满足:
   slime 既不能自动回滚, 也不能在崩溃后有效恢复 (恢复=死锁或空转)

## 实验 B: C1_ack_lost 修正版 (checkpoint 落盘成功 vs 恢复)

- checkpoint 落盘成功 (iter 7, 耗时 20s, latest_checkpointed_iteration.txt=7) 后 kill
- 重启: **无 StepToken / ACK / 幂等识别概念** (slime 同步写盘, 无提交协议)
- 验收要求 "通过 Checkpoint 内嵌的 StepToken 幂等识别已提交状态, 绝不重复 apply 梯度":
  **不满足** — 无 StepToken 机制; 且恢复死锁 (见实验 A) 使已提交/未提交边界无法利用

## B5 开销实测 (Day 5 矩阵输入)

- checkpoint 保存耗时: iter 3 = **31.0s**, iter 7 = **20.0s** (save-interval 4)
- step_time 均值 30.8s => **保存开销 = step 时间的 65%~100%** (per-step checkpoint 代价量级)
- 崩溃重算成本: 丢失 2 步 × 30.8s + rollout 全量重生成 (若从头恢复 = 10 步 × 30.8s ≈ 308s)
- Selective Replay 对比: 仅重算失败 Reward (文档主张节省 ≥30%) 的基准参照

## 产物

```
runs/p1-slime-L2-K8-s42-20260825/{meta.json, l2_result.json, logs, checkpoints(iter3,7), manifests, rewards.jsonl}
runs/p1-slime-L2-reload8b-K8-s42-20260825/ (恢复验证日志: 加载成功但 0 step)
runs/DAY4_SUMMARY.md
```

## 结论

- L2: 崩溃 -> 无回滚、无自动恢复、恢复双重死锁 (scheduler 匹配 + rollout 续接)
- C1/C2: 无 StepToken/ACK 幂等机制, "已提交状态识别" 不存在
- B5: per-step checkpoint 开销 65-100% step 时间, 全量重算成本明确
- 支撑 Go 条件 g2 (真实崩溃窗口) / g5 (Selective Replay 节省对比) / g6 (协议开销对照)
