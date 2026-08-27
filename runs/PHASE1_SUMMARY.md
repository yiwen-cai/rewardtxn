# Phase 1 实验总结报告 — RewardTxn 故障探针（5 天 + 补充）

**日期**: 2026-08-24 ~ 2026-08-26
**目标**: 实证"现有开源 RLVR/GRPO 栈在崩溃/重试/ACK 丢失下存在静默错误"，为 Phase 2 协议设计提供证据与基线
**最终裁决**: **GO（7/7 Go 条件满足）**，批准进入 Phase 2

---

## 1. 实验配置总览

| 项 | Day 1（旧基线） | Day 2-5 + 补充（最终配置） |
|---|---|---|
| 模型 | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
| 数据 | DAPO-Math-17k | GSM8K（7,473 条，DAPO 格式） |
| 组语义 | K=8, U=4, 32 样本/步 | 同左 |
| 训练 | 20 步, lr 1e-4, KL 0.01, seed 42 | 同左 |
| 卡数 | 4 卡 (0,1,2,5) | 3~4 卡（按可用性） |
| RM | deepscaler（修复版 v1/v2） | 同左（custom-rm 注入框架） |

**关键修复记录**（详见 DAY2_FIXES.md）：GSM8K role 字段缺失 / 0.5B 能力不足换 1.5B /
slime deepscaler `</think>` 分隔符 bug / 1.5B rotary-base 上游错误 / `--ipc=host` OOM。

---

## 2. 各日实验结果

### Day 1：基线（旧配置 0.5B）
- 20 步无故障完成，ray Job succeeded，2 checkpoint（iter 9/19）
- throughput_base 19,365 tok/s（稳态 20,368），step latency median 11.9s
- 建立最小 Lineage（steps.jsonl / groups.jsonl）与 TransferQueue 最小实例

### Day 2：Reward/Verifier 层故障注入（R1/R2/R3）
| 注入 | 机制 | 对照组判对 | 注入子组 | 组统计量偏移 | 静默性 |
|---|---|---|---|---|---|
| **R3 跨版本混算** | 组内 4 条 v1 + 4 条 v2 | 6.7% | **0.0%** (v2) | **-0.2562** | ✅ 完全静默 |
| **R1 RM 崩溃** | 窗口内抛 RuntimeError | 15.4% | 32 样本丢失 | 组被丢弃 | ⚠️ 半静默 |
| **R2 陈旧结果混入** | 注入子组用陈旧值 | 14.7% | **0.0%** | **-0.2750** | ✅ 完全静默 |

**R1 细节**: 崩溃组被 slime 整组静默丢弃（无重试），step 5 在残缺 batch 上训练
（loss 0.0027 异常于邻步 0.057/0.0006），Job 仍 succeeded。

### Day 3：Queue 消费与 Learner 崩溃窗口（Q0/Q1/L0）
- **Q0**（TransferQueue）：consumer get_meta（立即 mark_consumed）后 SIGKILL →
  重启重取失败（`Required: 32, Available: 0`）→ **32 样本永久丢失**。
  源码级证实：mark_consumed 不可逆、无 timeout/reclaim/ACK 重放机制。
- **L0**（slime trainer）：kill MegatronTrainRayActor → ray 不自动重启
  （ActorDiedError）→ Job failed，无 checkpoint 恢复点，8 步成果丢失。
- B3/B4 机制边界：幂等去重只防重复消费、Occupy/Consume 无超时回滚 → 窗口均存在。

### Day 4：Optimizer ↔ Checkpoint ↔ ACK 窗口（L2/L3/C1/C2 + B5）
- **L2**：Optimizer 阶段 kill trainer → 无自动重启，恢复路径**双重死锁**：
  ① megatron scheduler 要求恢复配置 num_steps 与 checkpoint 精确相等
  （num_rollout 必须 = 保存时已跑步数，否则 AssertionError 无法启动）；
  ② 即使匹配加载成功（iter 7），fully-async rollout 轮次无法续接 → 0 新 step 空转。
- **C1/C2**：checkpoint 落盘成功（iter 7, 20s）后 kill → 重启：**无 StepToken/ACK/幂等
  识别概念**，"已提交状态识别、绝不重复 apply"验收不满足。
- **B5 开销**：checkpoint 保存 20-31s/次 = step 时间 65%~100%；崩溃重算
  = 丢失步 × 30.8s + rollout 全量重生成（从头恢复 ≈308s）。

### Day 5：B0-B5 矩阵 + Go/No-Go 裁决
- **裁决 GO（7/7）**：见第 4 节。
- g4/g6 采用**协议语义模拟器**（修正文档循环依赖）：全切点 0 Invalid Commit、
  协议开销 2.54%。
- g5 核算：Selective Replay 节省 vs B2 85.9% / vs B5 95.0%。

### 补充实验（2026-08-26，审查补强）
| 补充 | 结果 | 意义 |
|---|---|---|
| 干净基线（1.5B+GSM8K+4卡） | 20 步, 11,179 tok/s, 判对率 12.6% | 配置对齐，横向可比 |
| R3 strict 重复 | 组偏移 **-0.2625**（首次 -0.2562） | 结果可重复 ✓ |
| R3 loose 模式 | 组偏移 **-0.2437** | 结论不依赖 v2 构造 ✓ |
| 文档性说明 | B3 语义映射 / R2 弱化定位 | 严谨性 ✓ |

---

## 3. B0-B5 全矩阵（DAY5_MATRIX.json）

| 基线 | R1 崩溃 | R2/R3 混算 | Q0/L0/L2 窗口 | 重算代价 |
|---|---|---|---|---|
| **B0** 默认开源栈 | 静默丢弃 32 样本 | 完全静默（组偏移 -0.26~-0.28） | 永久丢失/无恢复 | 0（训练损坏） |
| **B1** 组 ID 校验 | 部分拦截 | **穿透**（无版本检查） | 不覆盖 | 需整组丢弃 |
| **B2** 丢组（AReaL 实证） | 安全丢弃 | **穿透**（组完整不触发） | 不覆盖 | 100% Rollout 浪费 |
| **B3** 幂等去重 | 不覆盖 | 不覆盖 | **穿透**（防重复不防窗口） | 数据丢失 |
| **B4** Occupy/Consume | 不覆盖 | 不覆盖 | **窗口存在**（无回滚） | 数据丢失 |
| **B5** 每步 ckpt | 可恢复 | 可恢复 | 可恢复 | 100% 重跑 + 保存开销 65-100% step 时间 |
| **RewardTxn**（模拟） | ABORTED+重算 | ABORTED（0 Invalid） | Reconciler/StepToken | **仅重算 Reward，节省 85.9-95.0%** |

---

## 4. Go/No-Go 门禁（文档 8 节）

### Go 条件 7/7 全部满足

| # | 条件 | 证据 | 结果 |
|---|---|---|---|
| g1 | ≥2 类静默错误（含组校验后） | 3 类（R1/R2/R3）；B1 校验器 20/20 注入组放行、80 样本穿透 | PASS |
| g2 | 真实崩溃窗口 | Q0（Available:0）/ L0 / L2 / C1 四窗口实测 | PASS |
| g3 | 梯度污染可测 | 受控实验：Delta L2=0.260、梯度余弦 0.755 | PASS |
| g4 | 零无效提交 | 协议语义模拟器全切点 0 | PASS |
| g5 | 重算节省 ≥30% | 85.9%（vs B2）/ 95.0%（vs B5） | PASS |
| g6 | 协议开销 <5% | 2.54% | PASS |
| g7 | 无现成等价保证 | 三栈上游 main 逐一核查无 Durable Step Commit | PASS |

### No-Go 条件 7 项全部不成立
组校验/丢组无法消除错误 / 窗口真实存在 / B5 开销高（非"极低"）/ exactly-once
可绑定 Durable Checkpoint / 注入均为真实 kill 与数据流修改（非内存篡改）/
上游未合入等价协议 / 贡献非 Lineage 界面。

---

## 5. 核心结论

1. **问题真实存在**：锁定开源栈上稳定复现 3 类静默/半静默错误（跨版本混算完全
   静默、RM 崩溃半静默丢数据、陈旧结果混入完全静默），组归一化污染可测
   （组偏移 -0.24~-0.28，重复实验稳定）。
2. **崩溃窗口真实存在**：队列侧（Q0 消费标记不可逆）与训练侧（L0/L2 无自动
   重启、恢复双重死锁）均有源码级 + 实测级证据。
3. **单个机制不够**：B1-B5 各基线在声称覆盖的注入下均失败（穿透或代价极高），
   只有"Seal + Fencing + StepToken + Selective Replay"组合能达到 0 无效提交
   且仅重算失败 Reward。
4. **方案可行（模拟级）**：协议语义模拟器显示 0 Invalid Commit、2.54% 开销、
   85-95% 重算节省——需 Phase 2 真实原型实测复核。

---

## 6. 产物清单

```
runs/DAY5_MATRIX.json / DAY5_VERDICT.json / DAY5_SIMULATOR.json / DAY5_G5.json
runs/DAY5_SUMMARY.md / DAY2_SUMMARY.md / DAY3_SUMMARY.md / DAY4_SUMMARY.md
runs/PHASE1_SUPPLEMENT.md / DAY2_FIXES.md / G7_UPSTREAM_AUDIT.md / B2_AREAL_RESULT.md
runs/day5-final/verdict.json            # judge_gates.py 自动化裁决 GO
runs/p1-slime-B0-K8-s42-20260824/       # Day1 基线（0.5B）
runs/p1-slime-B0-1525b-K8-s42-20260826/ # 补充干净基线（1.5B）
runs/p1-slime-{skew,crmcrash,dup}-K8-s42-20260825/   # Day2 三注入
runs/p1-tq-Q0Q1-s42-20260825/           # Day3 Q0/Q1/B3/B4
runs/p1-slime-L0-K8-s42-20260825/       # Day3 L0
runs/p1-slime-L2-K8-s42-20260825/       # Day4 L2/C1/C2/B5（含恢复实验链）
runs/p1-slime-skew-{r2,loose}-K8-s42-20260826/       # 补充实验
scripts/day1-5_*.py/.sh                 # 全部实验脚本（可复现）
```

## 7. 对 Phase 2 的输入

- **待实现 7 模块**：Group Seal / Reward CAS / StepManifest / StepToken /
  Reconciler / Selective Replay / Trace Runner（文档 9.1）
- **实测复核项**：g4（0 Invalid Commit）与 g6（开销 <5%）需真实原型验证；
  8 卡 + Qwen3-4B/Qwen2.5-7B 端到端评测（9.2）
- **复用资产**：custom-rm 注入框架（反向用于 Seal 校验）、day4_inject.sh
  （Trace Runner 基础）、rewards.jsonl 逐样本 instrumentation
  （Selective Replay 判定输入）
