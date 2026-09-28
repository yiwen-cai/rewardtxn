# F4′ 修订 A3：GPU 监测器把刚被 kill 的自有进程误判为外部进程（2026-09-28，用户批准）

## 问题
p06 原始尝试和 p07 的 A 臂，都在 SIGKILL 之后 1–2 秒被 `gpu_load_monitor` 以 `foreign_compute_process` 停掉。被标记的 PID（1564305、1628956）在 kill 前的采样里都在本容器的 `own_pids` 中，就是跑在 GPU 7 上的 trainer（约 63 GB）。进程被 kill 后，它先从 `docker top` 中消失，而 `nvidia-smi` 还要一会儿才不再列出它的显存。监测器每次只拿当前容器的进程列表做比对，于是误判。这两次都不是外部进程。

p01 redo1 的外部进程 PID 805549 从未出现在本容器中，那次仍是真正的外部占用（chenjunjie 的评测）。

## 修订
`tests/ft/gpu_load_monitor.py` 把本次运行中出现过的所有自有 PID 记为已知集合 `seen_own`。只有不在该集合中的计算进程才判为外部进程，其余逻辑不变。用新规则重放历史采样：p06 和 p07 不再报警，p01 redo1 仍然报警。

## 登记更正
- p06 原始尝试：原因从"外部进程"更正为"监测器误判"。原 `error` 保留不改，新增 `cause_correction_A3` 说明。补做 redo1 已通过，结果不受影响。
- p07：原因同上。按同一 seed、同一顺序做一次整对补做（redo1）。监测器误判属于测量工具缺陷，按 `FORMAL_F4T_REDO_AMENDMENT_20260927.md` 的精神，不占用补做次数。
- p07 起改用新冻结 `FORMAL_F4T_FREEZE_20260928-A3.json`，只有监测器的哈希变化；seed、顺序、判据和统计都不变。
