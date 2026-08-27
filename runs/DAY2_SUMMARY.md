# Day 2 实验汇总 — Reward/Verifier 层故障注入 (2026-08-25)

模型: Qwen2.5-1.5B-Instruct | 数据: GSM8K (7473 条, dapo 格式) | K=8, U=4, 20 步, 3 卡 (4,5,7)
注入机制: --custom-rm-path (day2_custom_rm.py), 窗口按 group_index (step = group_index//4)
修复记录: 见 DAY2_FIXES.md (role 字段 / 1.5B 模型 / deepscaler 分隔符 bug / v2 构造)

## 结果对比矩阵

| 注入 | 切点 | 窗口 | 对照组判对 | 窗口内 v1 | 注入子组 | 组统计量偏移 | 训练行为 | 静默性 |
|---|---|---|---|---|---|---|---|---|
| **R3_skew** (跨版本混算) | R3 | g20-39 (step5-9) | 6.7% | 51.2% | **0.0%** (v2 strict) | **-0.256** | 20 步正常完成 | ✅ 完全静默 |
| **R1_crm_crash** (RM 崩溃) | R1 | g20-23 (step5) | 15.4% | — | 32 样本丢失 | 组被丢弃 | 20 步完成, step5 loss 异常 (0.0027 vs 邻步 0.057/0.0006) | ⚠️ 半静默 (warning 日志, 训练无感) |
| **R2_dup** (陈旧重试混入) | R2 | g20-39 (step5-9) | 14.7% | 55.0% | **0.0%** (retry-stale) | **-0.275** | 20 步正常完成 | ✅ 完全静默 |

## 三类静默错误证据 (文档 Day 2 预期: B0 静默错组/挂起)

1. **R3 Revision Skew**: 同组 8 条轨迹中 4 条用 v1、4 条用 v2 计算, 组内均值被系统性拉低
   (-0.256), Advantage 归一化在污染统计量上进行, 训练全程无报错。**显式 Group ID 校验无法拦截**
   (组 ID 合法, 版本混入不可见) —— 对应文档 B1 边界。
2. **R1 Worker 崩溃**: RM 抛异常 -> slime 将整组静默丢弃 (无重试, 无告警升级),
   32 样本永久丢失, 该 step 在残缺 batch 上更新模型 (loss 异常), Job 仍 succeeded。
3. **R2 重复交付不一致**: 陈旧重试结果混入组内, 注入子组全 0 分, 组均值偏移 -0.275, 训练无感知。

## 与 Day 1 基线对比

- throughput_base: 15.5-16.2k tok/s (1.5B 模型, 3 卡) vs Day1 20.4k (0.5B, 4 卡) —— 模型变大吞吐下降, 属预期
- step latency: ~32-33s vs Day1 11.9s (1.5B rollout 更慢)
- 注入对吞吐无显著影响 (skew/dup 无重算, crm_crash 丢弃后反而少算) —— 开销侧证据

## 产物

```
runs/p1-slime-{skew|crmcrash|dup}-K8-s42-20260825/
├── meta.json          # fault_injection 记录 crash_point/窗口/机制
├── logs/train.log     # 含 injected crash 日志 (R1) 与完整训练
├── rewards.jsonl      # 逐样本 (group_index/index/reward/verifier/injected)
├── manifests/         # steps.jsonl + groups.jsonl
├── metrics.json       # 含 day2_observation + silent_errors_count
├── analysis.json      # 注入效果分析 (判对率/组偏移)
└── checkpoints/
```

## 结论

- 在锁定开源栈 (slime v0.3.1) 上稳定复现 **3 类静默/半静默错误**, 其中 R3/R2 完全静默且穿透组 ID 校验
- 组归一化污染可测 (组均值偏移 -0.25~-0.28, 注入子组判对率归零)
- 为 Day 5 Go 门禁 g1 (≥2 类静默错误) / g3 (梯度污染可测) 提供直接证据
- 修复后的方案 (custom-rm 注入 + 逐样本 instrumentation) 全程可用, 可直接扩展至 Day 3/4 注入
