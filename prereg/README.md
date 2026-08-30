# prereg/ — 论文实验预注册目录

本目录存放 RewardTxn 论文实验（P0–P4）的全部预注册内容，与实验代码同 commit 冻结。

## 用途（对应方案 §8.1 与 §15.1）

- **样本量**：每个实验（E1–E8）的重复次数、trace 注入数、每切点下限；
- **排除条件**：允许排除 run 的理由（外部 GPU 抢占、磁盘/网络故障等）与 exclusion log 要求；
- **单一等价 margin**：E7 TOST margin，pilot 后冻结为单一值（禁止事后二选一）；
- **停止规则**：允许提前终止的预注册条件；
- **fault schedules**：E7/E8 播放的固定故障序列（附录 B.2 schema）；
- **eval splits**：E7 固定评测集划分（附录 C.3）。

## 冻结纪律

1. **P0 冻结前**：所有 JSON 内容可修改，状态字段为 `pending_pilot` / `draft`；
2. **P0 冻结时**：`prereg_index.json` 的 `frozen_at`/`freeze_commit` 置为实际值，`tost_margin.json` 的 `status` 由 `pending_pilot` 翻转为 `frozen` 并写入冻结值；
3. **冻结后**：任何字段修改必须走新的 commit 并在 `changelog` 中说明理由，正式统计一律使用冻结版本；
4. fault schedules 与 eval splits 一经生成即不可变（内容哈希写入索引）。

## 文件清单

| 文件 | 内容 |
|---|---|
| prereg_index.json | 注册表 + 冻结状态 |
| E1_E2_correctness.json | 问题复现 + 30k trace 安全性的样本量/排除/停止 |
| E3_baseline.json | B0–B6 公平对比 |
| E4_ablation.json | A0–A7 消融 |
| E5_replay.json | Selective Replay 边界 |
| E6_overhead.json | 正常路径开销与规模 |
| E7_training.json | 5-seed 训练语义 |
| E8_soak.json | 长期 soak |
| stop_rules.json | 全局停止规则 |
| tost_margin.json | 等价 margin 状态机 |
| fault_schedules/ | E7 每 seed + E8 soak 故障序列（生成器：scripts/fault_schedule_gen.py） |
| eval_splits/ | GSM8K 固定划分（生成器：scripts/gen_gsm8k_split.py） |
