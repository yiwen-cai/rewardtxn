# 最小 FT 正式矩阵结果（2026-09-25）

冻结：[FORMAL_FREEZE_20260924.md](FORMAL_FREEZE_20260924.md)。机器可读结果：[FORMAL_RESULTS_20260925.json](FORMAL_RESULTS_20260925.json)，由 `uv run python tests/ft/analyze_ft_formal.py` 从 `minimal_evidence/formal-*-pair.json` 与各 run 的 `cost.json` 生成。范围：AReaL 分支、Qwen2.5-0.5B、每 run 10 次有效更新；A = AReaL 原生恢复，R = A + RewardTxn。

## 主结果（F2，10 对）

| 臂 | 32/32 目标回答以同一执行来源进入保留链 | 精确 95% 区间 | 分类 |
|---|---:|---|---|
| A | 0/10 | [0.00, 0.31] | 10× `safe_discard` |
| A+R | 10/10 | [0.69, 1.00] | 10× `correct_recovered` |

- 10 对全部安全、可判定（无 `invalid_commit`/`safe_stop`/timeout/来源不可判定），满足冻结的检验前提。
- 不一致对 R 独成 10、A 独成 0；双侧精确 McNemar p = 0.00195 < 0.05。
- 解读边界：A 的 `safe_discard` 是安全行为而非错误；结论仅是 R 在 F2 切点下能保留并复用故障前已生成、已评分的工作，而 A 按设计丢弃重算。不外推到 F1/F4、其他模型规模或更长训练。

## 成本（描述性，不相减推断收益）

| 场景 | 臂 | 墙钟中位 s（范围） | GPU·h 中位 | 峰值增量磁盘 | 故障后生成 token 中位 |
|---|---|---|---|---|---|
| F2 | A | 785（772–804） | 0.87 | 7.0 GB | 124,648 |
| F2 | R | 1092（1075–1121） | 1.21 | 34.7 GB | 101,029 |
| 无故障 | A | 530（521–581） | 0.59 | 7.0 GB | — |
| 无故障 | R | 847（838–863） | 0.94 | 20.9 GB | — |

- R 在故障后少生成约 19% token，但完整 run 墙钟约为 A 的 1.4×（F2）/1.6×（无故障）；不得宣称 R 端到端更快。R 的正常路径开销（每步持久提交、分片快照与验收链）是后续优化重点。
- 墙钟为训练至独立验收全程（`cost.json` scope），未拆分训练与验收/清理成本。

## 排除与偏差

- `formal-f2-p02-s8954-20260924`：R 在故障信号前权重同步 CUDA OOM，`fault_not_valid_hit` → 技术无效；按冻结规则以 `-redo1` 同 seed、同顺序完整重做一次并通过，原件全量保留，见 [说明](FORMAL_P02_TECHNICAL_INVALID_REDO_20260924.md)。
- GPU 分配修订为任意四张同时空闲 H100，逐对记录新冻结哈希，见 [修订](FORMAL_GPU_ASSIGNMENT_AMENDMENT_20260924.md)。
- 第 6 对（s938）两臂全量保留，可重新加载复验；其余通过 run 已删大分片。
