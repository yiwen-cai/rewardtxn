# 最小 FT 正式实验参数冻结

冻结时间：2026-09-24 13:14 CST。机器可读清单为 [FORMAL_FREEZE_20260924.json](FORMAL_FREEZE_20260924.json)，其启动命令锚定的 SHA-256 为 `c0e386e88dda06fcb7231506636506af246cb3ffcfe03718de591e69b280343b`；完整 510 文件源码哈希为 [FORMAL_SOURCE_SHA256_20260924.json](FORMAL_SOURCE_SHA256_20260924.json)。以下参数在第一条正式 run 前确定；四条 pilot 均不计入正式样本。

## 条件与矩阵

- 固定镜像 `sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`；AReaL 基线 `b83d1f40196e5bd7d9f83092563443561870d550`，本地已跟踪补丁 diff SHA-256 为 `f12f7767a69e9a807885e646d622a407d16725bba97d43c5587c404cd84e0a8e`，未跟踪新增源码逐文件列在完整哈希表中。哈希表自身 SHA-256 为 `4a3a44f8ae5bd251b82776c7abc4d9108a832ad5b2aa00ce5e2d42d815e89894`。
- 模型为 `models/Qwen2.5-0.5B-Instruct`，数据为 `runs/diagnosis-20260910/train.jsonl`；各文件 SHA-256、基础 YAML 和正式入口 SHA-256 列在 JSON 的 `extra_sha256`。每次启动前重新核对冻结文件，训练容器内再核对源码及有效输入哈希。A/R 共用 `FT_MINIMAL_BLOCKING_D2H=1`、`FT_MINIMAL_AUTOTUNE_POINTWISE=0`，保留异步完整 DCP、每步保存、原生恢复与严格评分；不启用诊断环境变量。
- 4 张 H100 顺序为 `GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794`、`GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9`、`GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3`、`GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d`；最后一张训练。每 run 前必须四卡均不超过 100 MiB、0% 利用且无 compute process。
- 每 run 10 次有效更新，K=8、U=4。F2 在第 2 次成功 optimizer 更新后等待前代异步保存完成，再于本次更新保存前杀 trainer。F2 原生重试 1 次、无故障 0 次；每 run 控制器上限 2400 秒、恢复观察 900 秒、握手 10 秒、lease 20 秒。有效 argv、环境和调度还会逐 run 保存在 `controller-config.json`、`launch.json`、`ft1-case.json`。

| 顺位 | 场景 | seed | 顺序 | 名称 |
| ---: | --- | ---: | --- | --- |
| 1 | F2 | 250 | A→R | `formal-f2-p01-s250-20260924` |
| 2 | F2 | 8954 | R→A | `formal-f2-p02-s8954-20260924` |
| 3 | F2 | 2381 | A→R | `formal-f2-p03-s2381-20260924` |
| 4 | F2 | 9155 | R→A | `formal-f2-p04-s9155-20260924` |
| 5 | F2 | 9318 | A→R | `formal-f2-p05-s9318-20260924` |
| 6 | F2 | 938 | R→A | `formal-f2-p06-s938-20260924` |
| 7 | F2 | 6992 | A→R | `formal-f2-p07-s6992-20260924` |
| 8 | F2 | 4558 | R→A | `formal-f2-p08-s4558-20260924` |
| 9 | F2 | 6026 | A→R | `formal-f2-p09-s6026-20260924` |
| 10 | F2 | 7259 | R→A | `formal-f2-p10-s7259-20260924` |
| 11 | 无故障 | 8830 | R→A | `formal-nofault-p11-s8830-20260924` |
| 12 | 无故障 | 5459 | A→R | `formal-nofault-p12-s5459-20260924` |
| 13 | 无故障 | 5657 | R→A | `formal-nofault-p13-s5657-20260924` |

13 个 seed 由 `random.Random(20260924).sample([0..9999 排除 pilot seed 419、421], 13)` 在任何正式结果产生前抽取。F2 顺序各 5 对。独立用随机种子 `20260925` 从 10 个 F2 顺位中预抽取**第 6 对**保留两臂全量权重及重新加载所需元数据，名单与结果无关。

## 判定、空间与停止

- F2 每 run 的主结果为目标 32 条原始回答及评分是否全部以同一物理执行来源进入最终保留链；仅 32/32 记 1。A 或 R 安全但 0/32 均照实记录，不按收益筛选。无故障只描述同版本正常路径成本。只有 10 对均安全、可判定时才做双侧精确 McNemar 检验，阈值 0.05，并报告两臂精确 95% 区间。完整 run 成本与验收/清理成本分别记账，不用无故障耗时直接相减推断故障收益。
- 四条 pilot 最大端到端空间降幅 `P=34,727,661,568 B`，因此每条正式 run 前 `/public` 可用空间须至少 `56,202,498,048 B`（`P+20 GiB`）。一次只运行一个训练 run，随后完成故障/状态链/输入/新进程完整状态加载/来源验收；通过后逐分片 SHA-256，第 6 对 F2 的两臂保留全量，其余通过 run 删除大分片并核对释放量。小证据永久保留。
- `invalid_commit`、`safe_stop`、`timeout`、技术无效、来源不可判定或独立加载失败时停跑，保留失败 run 全量原件。技术无效最多允许一次同 seed、同顺序的完整配对补做，必须新目录并登记原件；有效命中而安全但未复用不补做。任何实现或清理路径变更先重新测 `P` 并重新冻结。正式调度器不自动补做或跳过失败配对。

正式入口为 `python3 tests/ft/run_ft_formal.py --freeze docs/experiments/rewardtxn-ft-20260916/FORMAL_FREEZE_20260924.json --freeze-sha256 c0e386e88dda06fcb7231506636506af246cb3ffcfe03718de591e69b280343b`。当前四卡中 GPU5 被外部作业占用，故用 `tests/ft/launch_ft_formal_when_idle.py` 携同一冻结文件及 SHA-256 等待四卡空闲，最长 12 小时；超时只停候卡进程，不启动训练。它逐对串行调用阶段 2 已验证的单臂训练、验收、来源检查及清理组件；正式 run 和 pair 记录使用新的 `formal-*` 名称，`formal_sample=true`。阶段 2 pilot 的训练/清理实现和输出文件系统不变，故沿用其实测 `P`；每条正式 run 仍执行现场空间与 GPU 门槛。
