# Day 5 修正方案 (2026-08-26)

基于文档 7 节 Day 5 + 8 节门禁，修正 5 个问题后执行：

## 修正 1: g4/g6 循环依赖 (文档矛盾)
- 文档: Go 条件 4/6 需要 RewardTxn 原型, 但 Phase 2 要等 Day 5 门禁通过
- 修正: Day 5 用**协议语义模拟器** (day5_simulator.py) 评估 g4/g6:
  在 Day 2-4 已落盘的注入数据上重放 Seal/Fencing/StepToken/Reconciler 语义,
  统计 Invalid Committed Step 数与协议元数据开销

## 修正 2: g5 节省率测量定义缺失
- 定义: 重算成本 = 重算所需 Rollout GPU·s (Rollout token 数 × 单位成本)
- Selective Replay: 复用 Rollout 前缀, 仅重算失败 Reward (CPU, ≈0 GPU)
- 对比: B2 (丢组后 100% rollout 重来), B5 (整步重跑, Day 4 实测)
- 门槛: 相比 B2/B5 减少 ≥30%

## 修正 3: g3 Delta 漂移未直接测量
- 受控实验: 用 Day 2 真实注入数据 (rewards.jsonl) 的 reward 分布,
  构造干净组 vs 污染组 advantage, 计算模拟策略梯度 Delta 与余弦相似度
  (同数据、仅 reward 不同 → 归因干净)

## 修正 4: B1/B2 未实现
- B1: 轻量校验器 (Group ID + 组大小), 对 Day 2 注入数据重放 -> R3 穿透验证 (g1 子条件)
- B2: AReaL drop_incomplete_group — 源码级机制审查 + 数据模拟; 若可行补最小实例实验

## 修正 5: 全矩阵工作量陷阱
- 不跑 6×7=42 次实验; B0 全注入数据已有 (Day 2-4),
  每基线仅在声称覆盖的注入下验证

## 执行顺序
1. g7 上游核查 (slime main 是否已合入等价机制)
2. g3 受控实验 (真实数据梯度 Delta)
3. B1 校验器 + R3 穿透验证 (g1 子条件)
4. 协议语义模拟器 (g4/g6)
5. g5 测量填表
6. B2 (AReaL 源码审查 + 数据模拟)
7. 全矩阵汇总 + judge_gates.py 裁决 + DAY5_SUMMARY.md

## 状态: 全部完成 (2026-08-26) - 裁决 GO (7/7), 详见 runs/DAY5_SUMMARY.md
