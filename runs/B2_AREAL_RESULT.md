# B2 基线: AReaL drop_incomplete_group (2026-08-26)

## 机制审查 (pin b83d1f40)
- areal/infra/controller/rollout_controller.py + GroupedRolloutWorkflow:
  drop_incomplete_group=True 时, 组内任一轨迹失败/缺失 -> "dropping entire group"
  -> 整组丢弃, 返回 None (不污染组统计量)
- 语义: 安全丢弃不完整组, 但该组全部 Rollout 作废

## 实证 (容器内运行)
- tests/test_grouped_rollout_workflow.py: 4/4 passed (11.6s)
  含 test_grouped_rollout_workflow_drops_incomplete_group (丢弃行为断言)

## 对照 Day 2-4 注入场景
- R1 崩溃 (32 样本): AReaL 检测组不完整 -> 丢弃 4 组 -> 32 样本 Rollout 100% 浪费
  -> 需重新采样生成 (重算成本 = 整组 rollout, g5 表 B2 列: 31.3s)
- R3/R2 (80 样本): 组完整 (8 条) -> drop_incomplete_group 不触发 -> 跨版本混算
  **仍穿透** (B2 不检查版本一致性) -> 与 B1 相同边界
- Q0/L0/L2: AReaL 无 queue 提交/StepToken 机制 -> 窗口依旧

## B2 矩阵结论
| 注入 | B2 表现 | 代价 |
|---|---|---|
| R1 崩溃 | 安全丢弃 (无污染) | 100% Rollout 浪费 (整组重采样) |
| R3/R2 混版本 | **穿透** (组完整时无版本检查) | 0 (训练损坏) |
| Q0/L0/L2 崩溃窗口 | 无提交机制, 窗口依旧 | 数据丢失 |
