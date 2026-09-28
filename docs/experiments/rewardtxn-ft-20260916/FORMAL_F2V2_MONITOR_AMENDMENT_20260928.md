# F2v2 修订 A4：GPU 监测器对"刚创建的自有子进程"误判（2026-09-28，用户批准）

## 问题
F2v2 p01 redo1 的 R 臂启动约 120 s 时，监测器在 GPU-1b3ba0aa 上看到 PID 2991363（458 MiB）。它不在同一轮 `docker top` 的结果里，于是被判为外部进程，训练随即被停。证据如下：
- 该 PID 夹在本容器的 PID 序列中间（2991312 与 2991564 之间），产生时间正是容器批量启动子进程的时候。
- 所有 F4′/F2v2 的 R 臂在启动时都会短暂出现 450–1,700 MiB 的自有小进程（CUDA 上下文），以往都被正确认成自有进程。
- 监测器每轮先 `docker top`、后 `nvidia-smi`，两步之间新产生的子进程会被漏认。修订 A3 只处理了"已死亡但显存尚未释放"的情况。

真实的外部占用（p01 原始尝试，xuyouxuan，PID 2968156，69 GB，持续存在，cgroup 属于 user.slice）与此明显不同。

## 修订
`tests/ft/gpu_load_monitor.py`：
1. 对不在已知自有集合中的 PID，读取 `/proc/<pid>/cgroup`；其中包含本容器 ID 的，记为自有进程。
2. 其余可疑 PID，只有在连续两轮采样（间隔约 5 s）中都被判为外部进程，才停训练。样本中新增 `foreign_confirmed` 字段。

单项验证：本机容器进程判为自有；宿主 PID 1 和 xuyouxuan 的 PID 判为外部。真实外部占用仍会被发现，最多晚约 5 s。

## 处理方式（用户决定：不影响结果的修复不整对重跑）
- p01 redo1 的 A 臂在误判之前已完整通过验收（`safe_discard`，0/32），结果不受监测器影响，予以保留。
- 只重跑被误杀的 R 臂：`tests/ft/resume_ft_formal_arm.py` 把原 R 臂证据改名为 `…-redo1-r-invalid1` 原样保留，然后在该对的 A4 冻结（`FORMAL_FREEZE_formal-f2v2-p01-s6293-20260928-redo1-A4.json`，只有监测器和脚本的哈希变化）下，从基座重跑 R 臂。
- p02–p13 使用 `FORMAL_F2V2_FREEZE_20260928-A4.json`。seed、顺序、判据和统计都不变。
