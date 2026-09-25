# 最小 FT 实验阶段 2：新版四 run pilot 完成记录

2026-09-24。阶段 2 的四个真实 10 步 pilot 已按两对串行完成，均为 `formal_sample=false`；正式样本数仍为 0。旧失败与诊断 run 的原件继续保留，历史停跑和定位过程见 [停跑记录](STAGE2_PILOT_STOP_20260924.md)。本记录只对新版冻结配置的功能可行性与空间作结论，不把四个 pilot 当作统计结果。

## 冻结条件与命令

冻结清单是 [`minimal_evidence/pilot-pointwiseoff-freeze-20260924.json`](minimal_evidence/pilot-pointwiseoff-freeze-20260924.json)。固定镜像 `sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`；两臂均在 minimal 入口启用 `FT_MINIMAL_BLOCKING_D2H=1` 和 `FT_MINIMAL_AUTOTUNE_POINTWISE=0`。`CUDA_LAUNCH_BLOCKING`、`FT_DCP_DIAG_PATH`、`PYTORCH_ALLOC_CONF` 均未设置；async/full DCP、保存频率、F2 切点与原生恢复保留。点算子 autotune 关闭避免了先前 Inductor benchmark 的 pinned CPU 备份路径；它与同步 D2H 是共用实现改动，不能从本次通过反推先前 CUDA 错误的唯一根因。固定镜像定向 CPU 合同测试 8 项通过，独立生成核同 shape 数值检查、单 launcher、host allocation 0 通过；细节见停跑记录的后续诊断及 [`diag-inductor-gpuclone-20260924/`](minimal_evidence/diag-inductor-gpuclone-20260924/)。

四张 GPU 的固定顺序为 GPU1、GPU5、GPU6、GPU7；训练卡为最后的 GPU7。F2 与无故障两对使用完全相同的 UUID：

```text
GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794
GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9
GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3
GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d
```

完整入口命令如下；每臂单独新目录、前一臂完成全部验收和清理后才启动下一臂。首对 seed 419、A→R；次对 seed 421、R→A：

```bash
python3 tests/ft/run_ft_minimal.py --name pilot-f2-pointwiseoff-s419-20260924 --scenario F2 --seed 419 --order A R --devices GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d
python3 tests/ft/run_ft_minimal.py --name pilot-nofault-pointwiseoff-s421-20260924 --scenario no_fault --seed 421 --order R A --devices GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d
```

启动前及最后复核冻结清单中的 7 个源码 SHA-256 全部一致。每 run 的 `ft1-case.json`、`controller-config.json`、`source-sha256.json`、源码归档和 GPU 预检原件均在各自证据目录。

## 逐 run 验收与成本

两份配对记录 [`F2 pair`](minimal_evidence/pilot-f2-pointwiseoff-s419-20260924-pair.json) 和 [`无故障 pair`](minimal_evidence/pilot-nofault-pointwiseoff-s421-20260924-pair.json) 均为 `pilot_pair_verified`，入口退出码均为 0。四 run 的 fault/smoke、chain、input、全新进程完整 native 加载、source 验收均通过；`functional-verification.json` 的 `full_native_reload_verified` 和 `safety_verified_for_retained_chain` 均为 true。两次 F2 均有一次有效 trainer 故障信号、11 次物理优化器成功（含一项被舍弃）、恢复后 9 次成功更新，最终保留 10 次更新。无故障两臂的 `smoke_passed=true`、优化器更新数为 10。

| run 证据目录 | 分类 | F2 目标来源 | 墙钟秒 / GPU·小时 | 磁盘峰值新增 B | 验收后 DCP |
| --- | --- | ---: | ---: | ---: | --- |
| [`F2 A`](minimal_evidence/pilot-f2-pointwiseoff-s419-20260924-a/) | `safe_discard` | 0/32 同执行来源 | 821.19 / 0.9124 | 7,347,187,712 | 1 分片哈希后删除 |
| [`F2 R`](minimal_evidence/pilot-f2-pointwiseoff-s419-20260924-r/) | `correct_recovered` | 32/32 同执行来源 | 1121.91 / 1.2466 | 34,727,661,568 | 4 分片哈希后删除 |
| [`无故障 R`](minimal_evidence/pilot-nofault-pointwiseoff-s421-20260924-r/) | `no_fault_verified` | 不适用 | 839.54 / 0.9328 | 20,963,926,016 | 2 分片哈希后删除 |
| [`无故障 A`](minimal_evidence/pilot-nofault-pointwiseoff-s421-20260924-a/) | `no_fault_verified` | 不适用 | 517.55 / 0.5751 | 7,183,409,152 | 1 分片哈希后删除 |

来源验收均 `verified=true`，每 run 有 320 条 retained rows。F2 A 的 32 个目标输入未以相同物理执行来源进入最终链；F2 R 的 32 个目标输入全部以相同来源进入最终链。来源只按父进程接受的评分尝试和物理执行 ID 判定。墙钟和 GPU·小时取各 run 的 `cost.json`，只覆盖训练容器运行，不含其后的独立验收；清理用时单列，不将这些非配对重复的 pilot 比作性能结论。

每个 run 的 `storage-cleanup.json` 留存每个分片路径、字节数和 SHA-256；按上表顺序，哈希记录覆盖的分片合计为 6,917,735,465、27,670,941,860、13,835,470,930、6,917,735,465 B，均有 64 位 SHA-256，且清理后路径均不存在。对应文件系统可用空间分别观测增加 6,917,718,016、27,666,001,920、13,832,450,048、6,913,908,736 B；哈希与清理耗时分别 8.21、38.56、16.21、7.68 秒。观测释放量是整个文件系统的空间差分，因此可与分片字节和有小幅偏差。原始 `*.distcp` 已删除，后续不能重验其内容；配置、事件、输入/评分来源、链、独立加载结果及哈希清单均留存。旧失败 run 中已有的 DCP 分片没有删除。

## 空间结论与后续边界

0.5 秒采样的 `disk-peak.json` 从训练启动覆盖至独立 input/chain/final-state load 验收结束，测量的是同一 `/public` 文件系统可用空间的端到端降幅，含重启残留和加载临时副本。四 run 最大值 **P = 34,727,661,568 B（32.34 GiB）**，来自 F2 R；按方案正式每 run 的启动门槛为 **P + 20 GiB = 56,202,498,048 B（52.34 GiB）**。最终现场 `/public` 可用 411,842,260,992 B（383.56 GiB），高于该门槛 355,639,762,944 B（331.22 GiB）。四张 GPU 最终均为 4 MiB / 0% 空闲；四个 pilot 容器已不存在。清理后的 `docker inspect` / `docker network inspect` 返回 `No such object`，证明控制器已移除对应容器和网络，并非训练或验收失败。

以上空间结论只适用于当前镜像、源码、配置、四 GPU 和清理路径；若正式 run 前改变实现或清理规则，按方案重新测 P 并冻结。阶段 2 到此完成，尚未启动正式 run；下一步先做独立评审和正式参数冻结。
