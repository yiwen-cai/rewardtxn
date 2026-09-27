# F4′ 正式矩阵补做规则修订（2026-09-27，用户批准）

原冻结规则：技术无效每对最多一次同 seed、同顺序整对补做。

修订：**由外部原因造成的技术无效不消耗补做次数**，可以用同一 seed、同一顺序再做一次整对补做。外部原因限于以下两种：
1. 宿主 runner 被外部终止，事件记录为 `host lease expired`；
2. GPU 监测在所选卡上发现外部计算进程（`foreign_compute_process`）。

实验自身导致的技术无效仍然只允许补做一次。seed、顺序、主判据、统计方法和冻结源码都不变。所有尝试的原始证据全部保留，GPU·h 全部计入 25 GPU·h 上限；由于补做会改名，汇总时需按名称前缀手工累加。

p01 记录：
- 原始尝试：R 臂 host lease expired（runner 随 Codex 会话被杀），A 0.85 + R 0.25 GPU·h；
- redo1：A 臂在约 136 s 时 GPU-40b9b481 上出现外部进程 PID 805549，被监测停止，0.15 GPU·h；
- redo2：由 `tests/ft/launch_ft_f4t_redo.py --attempt 2` 启动，等待任意四张空闲 H100 后开跑，完成后自动继续 p02–p10。
