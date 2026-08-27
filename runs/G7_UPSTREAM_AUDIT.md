# g7 上游核查: 无现成等价保证 (2026-08-26)

| 项目 | pin 版本 | 上游 main 最新 | 等价机制 |
|---|---|---|---|
| THUDM/slime | v0.3.1 (a6272da0) | a3f500977f (08-26) | 无: 仅 routing replay 样本级检查 / trajectory loss_mask 去重, 无 Step 级 Durable Commit |
| areal-project/AReaL | b83d1f40 | 94ce16558b (08-26) | 无: 匹配为误报 (countdown 规则 / perf_tracer contextvar) |
| Ascend/TransferQueue | 8497a52a | 5cb184ef (08-25) | 无: 匹配为误报 (sampler 消费语义) |

结论: 所 Pin 版本与上游 main 均未提供等价的 Durable Step Commit / StepManifest /
StepToken 机制 -> Go 条件 7 满足
