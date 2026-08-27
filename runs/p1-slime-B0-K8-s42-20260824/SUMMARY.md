# Day 1 实验摘要 — p1-slime-B0-K8-s42-20260824

日期: 2026-08-25 15:00-15:16 UTC+8 | 栈: THUDM/slime v0.3.1 (a6272da0) | 基线: B0 (无故障)

## 结果总览

| 验收项 (文档 Day 1) | 结果 | 证据 |
|---|---|---|
| 无故障连续 20 Step | ✅ | step 0-19, Job raysubmit_vYh86LU56YWQXH27 succeeded, 容器 Exited (0) |
| throughput_base | ✅ | 20,368 tok/s (稳态 step≥2), 全周期 19,365 tok/s |
| Step Latency 基准 | ✅ | median 11.9s / mean 34.1s (稳态); step 0 预热 199.5s 不计入 |
| checkpoint 落盘 | ✅ | iter_0000009, iter_0000019 (save-interval 10), 各 6.5GB |
| TransferQueue 最小实例 | ✅ | 64 样本 / 8 组流转, 121.6 samples/s, TrajectoryID 校验一致 |
| 最小 Lineage | ✅ | manifests/steps.jsonl (20), groups.jsonl (80 组 × K=8) |

## 配置 (与文档 10.2 meta.json 一致)

- K=8 (n-samples-per-prompt 8), U=4 组/步 (rollout-batch-size 4), global batch 32, 4 卡 0,1,2,5
- lr 1e-4, kl-loss-coef 0.01, seed 42, deepscaler rule-based RM, max-len 1024, temp 1.0
- 1 Actor + 3 Rollout (sglang mem-fraction 0.55)

## 指标 (metrics.json)

- throughput_base: 20,368 tok/s (稳态), actor_train_tflops ~60
- step_latency: median 11.9s, max(稳态) 波动来自 rollout 等待 (wait_time_ratio 0.62)
- rollout: 31.3s/步 (2 引擎), 稳定生成
- 无错误、无重试、无静默丢失 (基线对照)

## 产物

```
runs/p1-slime-B0-K8-s42-20260824/
├── meta.json          # 10.2 规范全字段
├── logs/train.log     # 完整训练日志
├── manifests/steps.jsonl   # StepToken = step 序号 (Day1 占位)
├── manifests/groups.jsonl  # group_id, trajectory_ids, K=8 (provenance=derived)
├── checkpoints/       # iter_0000009, iter_0000019
├── metrics.json       # 含 Day5 judge_gates 输入字段
└── tq_probe.json      # TransferQueue 流转结果

脚本: scripts/day1_run.sh, day1_slime_train.sh, day1_lineage.py, day1_tq_probe.py
```

## 备注

- StepToken 目前以 step 序号 + checkpoint iteration 占位; Day 2-4 将注入故障并扩展为 manifest 语义
- slime 日志仅聚合指标, 单组 Lineage 为静态推导 (provenance=derived), Day 2 起结合注入日志增强
- 基线供 Day 2-5 故障注入对照使用
