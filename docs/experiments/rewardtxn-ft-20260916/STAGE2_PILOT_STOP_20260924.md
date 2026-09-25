# 最小 FT 实验阶段 2 停跑记录

日期：2026-09-24。状态：**未完成四个 pilot，按预设失败条件停跑；正式样本仍为 0。** 未启动 R 臂或无故障配对，未形成新版正式 freeze，也不能计算正式 run 的 `P + 20 GiB` 空间门槛。

## 固定条件与启动命令

- 固定镜像：`sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`。
- 四张 H100 PCIe：`GPU-da816a5b-68f4-71fb-1ba8-d8ad7e505605`、`GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9`、`GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3`、`GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d`。启动前均满足 100 MiB/0% 且无 compute process 的空闲检查；两次启动使用同四张卡。
- F2 pilot：seed 419、顺序 A→R、10 步、第 2 次成功 optimizer 更新后故障、`formal_sample=false`。无故障 pilot 原拟 seed 421、R→A，因 F2 停跑而未启动。
- 阶段 1 相关 CPU 合同在固定镜像内重跑，初次 14 passed；验收器修订后 15 passed。两次均以仓库只读、无网络、可写临时 `/tmp` 和 `/output` 的容器运行。Python `compileall` 与 `git diff --check` 通过。

两次 F2 命令仅 `--name` 不同：

```bash
python3 tests/ft/run_ft_minimal.py --name pilot-f2-s419-20260924 --scenario F2 --seed 419 --order A R --devices GPU-da816a5b-68f4-71fb-1ba8-d8ad7e505605 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d
python3 tests/ft/run_ft_minimal.py --name pilot-f2-s419-v2-20260924 --scenario F2 --seed 419 --order A R --devices GPU-da816a5b-68f4-71fb-1ba8-d8ad7e505605 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d
```

## 已运行结果

| 证据目录 | 容器墙钟 | 端到端空闲空间降幅 | 结果 |
|---|---:|---:|---|
| `minimal_evidence/pilot-f2-s419-20260924-a/` | 734.93 s | 7,004,672,000 B（约 6.52 GiB） | F2 有效命中；11 次物理 optimizer 更新中最终保留 10 次；独立输入、链、完整原生状态加载通过。旧来源验收器误拒绝一次评分 retry，配对停跑。修订后的离线复验为 A `safe_discard`、原始 32 条复用 0/32。全量 DCP 保留。 |
| `minimal_evidence/pilot-f2-s419-v2-20260924-a/` | 528.62 s | 6,985,154,560 B（约 6.51 GiB） | F2 有效命中；恢复后第 3 次物理 optimizer 更新附近，异步 DCP 保存抛 CUDA 错误；无最终状态，完整链/输入/加载验收无法进行，停跑。全量 DCP 保留。 |

容器墙钟取各 run 的 `cost.json`，不含其后的独立验收；磁盘降幅取 `disk-peak.json`，采样间隔 0.5 s，覆盖训练到独立验收结束或停止。第一轮开始前 `/public` 约 406 GiB 可用；第二轮开始前约 399 GiB；两份失败原件保留后约 393 GiB。上述两个降幅不能代替四个 pilot 中最大值 `P`。

最终现场核对：`/public` 可用 **421,795,160,064 B（约 392.8 GiB）**。第一轮目录占用 6,987,170,016 B，第二轮 6,969,426,139 B；各保留 1 个 6,917,735,465 B 的 `*.distcp`，以及 DCP `.metadata`、`common.pt`、恢复元数据、源码快照、原始事件/日志和验收报告。两个目录均没有 `storage-cleanup.json`，未执行验收后分片删除。第一轮有 `final-native-state.json`；第二轮没有，且其现存分片不能据此视为可用终态。

第一轮证据：[`pilot-f2-s419-20260924-pair.json`](minimal_evidence/pilot-f2-s419-20260924-pair.json)、[`fault-verification.json`](minimal_evidence/pilot-f2-s419-20260924-a/fault-verification.json)、[`chain-verification.json`](minimal_evidence/pilot-f2-s419-20260924-a/chain-verification.json)、[`acceptance-status.json`](minimal_evidence/pilot-f2-s419-20260924-a/acceptance-status.json)、[`source-reverification.json`](minimal_evidence/pilot-f2-s419-20260924-a/source-reverification.json)、[`source-verification.json`](minimal_evidence/pilot-f2-s419-20260924-a/source-verification.json)。

第二轮证据：[`pilot-f2-s419-v2-20260924-pair.json`](minimal_evidence/pilot-f2-s419-v2-20260924-pair.json)、[`fault-verification.json`](minimal_evidence/pilot-f2-s419-v2-20260924-a/fault-verification.json)、[`acceptance-status.json`](minimal_evidence/pilot-f2-s419-v2-20260924-a/acceptance-status.json)、[`trainer.log`](minimal_evidence/pilot-f2-s419-v2-20260924-a/areal/logs/caiyiwen/rewardtxn-ft-pilot/r-fault-integration/trainer.log)、[`disk-peak.json`](minimal_evidence/pilot-f2-s419-v2-20260924-a/disk-peak.json)。两个 run 均有 `source-sha256.json`、`training.yaml`、`controller-config.json`、`launch.json`、GPU UUID 和完整容器日志。

## 停跑原因与范围

第一轮仅是验收器判定错误。一个评分子进程 pid 2702 写出 `score_execution_returned`，但父进程记录 `status=error, exit_code=-15`，随后接受重试子进程 pid 2704 的 `status=scored, exit_code=0`。旧验收器错误地要求同一 `sample_attempt` 仅有一次子进程返回。A 入口当前传入原始 `gsm8k_reward_fn`，没有调用 `observed_gsm8k_reward`，所以 `reward_done` 事件也未产生；修订后的来源验收器以父进程已接受的 child pid/执行 ID、物理生成和训练张量/最终链作关联。原始 run 快照中的验收器 SHA-256 为 `049c83d049dded726848b028feb40c99cb52cb20ca5b39071a60fa782b94fbe0`，离线复验验收器为 `0eb6e3214a3180b77677b5dfc1bc21669af329cba4acec627e4e92331eaad400`；这次复验不能改写原配对的 `stopped_for_review` 状态。

第二轮使用新目录重跑完整配对，训练配置与控制器配置的 SHA-256 和第一轮分别完全一致；`source-sha256.json` 中只有 `tests/ft/check_ft_minimal_source.py` 改变。恢复后保存下一代异步 checkpoint 时，Megatron 的 `filesystem_async.py:243` 在 `tensor.to('cpu', non_blocking=...)` 抛 `torch.AcceleratorError: CUDA error: invalid argument`。日志没有给出更早的确定性 CUDA 根因；不能将错误归于验收器变动或特定 GPU。控制器容器 `exitcode` 为 0，但 trainer 异常且 `final-native-state.json` 缺失，必须以验收状态判失败。

只读对照进一步缩小触发点：两次均在故障后第一次完整 DCP 保存处执行同一个 `MegatronCheckpointer.save_checkpoint → AsyncCallsQueue.schedule_async_request → filesystem_async.preload_tensors` 路径。第一轮此处成功并继续到 10 步；第二轮在 `preload_tensors` 内、异步请求入队前失败，因此恢复进程中没有对应的 `Scheduled async checkpoint save #0`。故障前的进程也有编号 `#0`，须按进程与时间区分。故障后该次 `checkpoint_save_state` 快照在两轮均含 680 个同形状、同 dtype 张量（170 个 `bfloat16`、510 个 `float32`，无空维），学习率调度器均为第 2 步；快照没有保存失败张量的键、stride 或设备指针，不能排除张量布局或先前异步 CUDA 错误。`dmesg` 未见 9 月 24 日的新 NVIDIA Xid；所选 GPU 5 有 9 月 15 日的旧 row-remap pending 记录，但没有证据把它与本次训练卡上的 D2H 错误连接。期间宿主 `nvidia-smi` 偶发无法连接驱动、重试后恢复，亦不足以判定根因。

依第 6 节规则，两个停止 run 的大分片均未清理，未启动其余 pilot。后续独立诊断及其边界见下节。

## 后续 CUDA 诊断：仍停跑

在原四卡中的 GPU4 被外部 VLLM 占用后，经父任务确认，单独诊断将推理 GPU4 换成空闲 GPU1，设备顺序为 GPU1/5/6/7，训练卡仍是 GPU7；此硬件差异使它不能作为原四卡的配对 pilot。其他条件为 seed 419、A/F2、10 步、故障序号 2、`formal_sample=false`、同一固定镜像、相同 `training.yaml` SHA-256 `22f547ec6aed31afc62e210bfe7add6e0b0f3a55f9b78340df7c11d2d0085e17`。诊断只在异步 DCP preload 外包一层由 `FT_DCP_DIAG_PATH` 开启的异常记录，原始 async/full checkpoint/recovery 语义未改；未开启预同步。配置 SHA-256 因该诊断环境变量和 run nonce 发生变化，源码快照中还包含诊断入口与 checkpointer 探针。

```bash
python3 tests/ft/run_ft_dcp_diag.py --name diag-f2-d2h-s419-20260924 --seed 419 --devices GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d
```

这次 run 容器墙钟 416.14 s，容器退出码 2，端到端空闲空间降幅 145,211,392 B，验收为 `fault_not_valid_hit` / `technical_invalid`。第一次 backward 时，Torch Inductor 的 `copy_args_to_cpu_if_needed → torch.empty_strided` 抛同类 `CUDA error: invalid argument`；尚未到故障序号或任何 DCP preload，故无 `dcp-diag.jsonl`、DCP 分片和最终状态。这说明错误不限于 v2 所见的 DCP 拷贝位置，不能据此修补 DCP。原始日志见 [`trainer.log`](minimal_evidence/diag-f2-d2h-s419-20260924-a/areal/logs/caiyiwen/rewardtxn-ft-pilot/r-fault-integration/trainer.log)，启动、源码与状态见 [`launch.json`](minimal_evidence/diag-f2-d2h-s419-20260924-a/launch.json)、[`source-sha256.json`](minimal_evidence/diag-f2-d2h-s419-20260924-a/source-sha256.json)、[`acceptance-status.json`](minimal_evidence/diag-f2-d2h-s419-20260924-a/acceptance-status.json)、[`disk-peak.json`](minimal_evidence/diag-f2-d2h-s419-20260924-a/disk-peak.json)。诊断目录约 44,450,614 B，完整保留。

两个失败容器的内核日志都出现 `page allocation failure: order:9`，栈为 `__gup_longterm_locked → get_user_pages → os_lock_user_pages [nvidia]`，随后 `Cannot map memory ... size of 0x40000 pages`（按 4 KiB 页约 1 GiB）。v2 的 `cpuset=81f62119...` 与其 [`container.id`](minimal_evidence/pilot-f2-s419-v2-20260924-a/container.id) 完全一致；诊断的 `cpuset=d8c3b866...` 与其 [`container.id`](minimal_evidence/diag-f2-d2h-s419-20260924-a/container.id) 完全一致。两次内核警告均比相应用户态报错记录晚约 12 秒，不能单凭时间顺序认定是该 CUDA 异常的直接原因；但它们属于失败训练的同一容器和 NVIDIA 锁页路径。失败时内核 `Mem-Info` 中 v2 两个 NUMA Normal 区的 order-9 及更高块均为 0；诊断两个区的 order-9 及更高块也均为 0。精确栈、内存区、cgroup 和 `Cannot map memory` 原文保存在 [`kernel-page-allocation-excerpts.log`](minimal_evidence/diag-f2-d2h-s419-20260924-a/kernel-page-allocation-excerpts.log)。10:28 的事后宿主快照为 MemFree 17,514,180 KiB、MemAvailable 490,567,568 KiB，Normal order-9 块为 0 和 14；它不是失败时快照，见 [`host-memory-snapshot.json`](minimal_evidence/diag-f2-d2h-s419-20260924-a/host-memory-snapshot.json)。可用内存大而高阶空闲块少，提示碎片或分配路径约束，尚不能确定完整根因。

在 GPU7 空闲且宿主 MemAvailable 约 468 GiB 后，使用同一镜像做独立 pinned host D2H smoke：`torch.empty(pin_memory=True)`、`dst.copy_(src, non_blocking=True)`、`torch.cuda.synchronize()`。64 MiB、256 MiB 均通过；1 GiB 一次通过，容器退出码 0，墙钟 9.19 s，期间无新增 `dmesg` 记录。具体 Docker 参数、API、输出和脚本分别见 [`pinned-smoke-small.result.json`](minimal_evidence/diag-f2-d2h-s419-20260924-a/pinned-smoke-small.result.json)、[`pinned-smoke-1g.result.json`](minimal_evidence/diag-f2-d2h-s419-20260924-a/pinned-smoke-1g.result.json)、[`pinned-smoke-1g.stdout.log`](minimal_evidence/diag-f2-d2h-s419-20260924-a/pinned-smoke-1g.stdout.log) 和 [`pinned-smoke-1g.py`](minimal_evidence/diag-f2-d2h-s419-20260924-a/pinned-smoke-1g.py)。独立成功只能证明当前轻载下该 API 和大小可用，不等同于完整训练时 NVIDIA 锁页路径的压力条件。

此时 `/public` 可用 421,674,790,912 B（约 392.7 GiB，10:30 左右）。正式样本为 0，四个 pilot 的 `P` 仍不可计算。

### `CUDA_LAUNCH_BLOCKING=1` 定位补充

按后续诊断要求，在 GPU1/5/6/7 均空闲、`/public` 可用 421,671,854,080 B 后，新目录以相同 seed 419、A/F2、固定镜像、相同训练 YAML、异步完整 DCP 与恢复设置启动；与上次诊断的意图差异仅为容器环境增加 `CUDA_LAUNCH_BLOCKING=1`（run nonce 和源码归档自然不同）。仍未加 DCP 预同步，也未启动 R 或无故障臂：

```bash
python3 tests/ft/run_ft_dcp_diag.py --name diag-f2-blocking-s419-20260924 --seed 419 --devices GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d --cuda-launch-blocking
```

容器墙钟 435.03 s、退出码 2、端到端空闲空间降幅 137,117,696 B；独立验收为 `fault_not_valid_hit` / `technical_invalid`，计划中的第 2 次更新后故障尚未命中。首次正常 DCP preload 内，`tensor.to("cpu", non_blocking=...)` 抛 `CUDA error: invalid argument`。探针记录报错项为优化器 `optimizer.distributed.dp_group_idx_0.gbuf_idx_0.dtype_(torch.bfloat16, torch.float32).bucket_idx_0.exp_avg`，张量 shape `[136134656]`、stride `[1]`、`float32`、`cuda:0`（训练 GPU7）、storage offset 0，约 519 MiB 数据；这是发生异常的项，并不证明其内容/布局有缺陷。`CUDA_LAUNCH_BLOCKING=1` 使用户态报错明确落在此 D2H 调用，但仅凭该栈仍不能证明底层最早原因。原文见 [`dcp-diag.jsonl`](minimal_evidence/diag-f2-blocking-s419-20260924-a/dcp-diag.jsonl) 和 [`trainer.log`](minimal_evidence/diag-f2-blocking-s419-20260924-a/areal/logs/caiyiwen/rewardtxn-ft-pilot/r-fault-integration/trainer.log)。该次在请求入队前失败，无 `*.distcp`、无最终状态；完整诊断目录约 45,074,477 B。

10:39:00 的内核警告再次与本次 [`container.id`](minimal_evidence/diag-f2-blocking-s419-20260924-a/container.id) 完全匹配：`page allocation failure: order:9`，同样经 `__gup_longterm_locked → get_user_pages → os_lock_user_pages [nvidia]`，随后 `Cannot map memory ... 0x40000 pages`；两个 NUMA Normal 区 order-9 及以上空闲块均为 0。警告比 10:38:48 的用户态报错晚约 12 秒，时序不能单独确立因果，但三次训练失败的同容器内核特征一致。完整内核片段见 [`kernel-page-allocation-excerpt.log`](minimal_evidence/diag-f2-blocking-s419-20260924-a/kernel-page-allocation-excerpt.log)；退出/故障/磁盘状态见 [`acceptance-status.json`](minimal_evidence/diag-f2-blocking-s419-20260924-a/acceptance-status.json)、[`fault-verification.json`](minimal_evidence/diag-f2-blocking-s419-20260924-a/fault-verification.json)、[`disk-peak.json`](minimal_evidence/diag-f2-blocking-s419-20260924-a/disk-peak.json)。

最终 `/public` 可用 421,534,859,264 B（约 392.6 GiB，10:42）；v1/v2 与两个诊断目录原件均保留，未做 DCP 清理。阶段 2 仍停跑。可操作门槛是先由节点/驱动侧诊断同容器的长期锁页与高阶分配失败，明确是否为宿主碎片、pin/register 路径或其它资源条件，并用与训练负载足够接近的定向测试验证修复；轻载 1 GiB smoke 通过不足以放行。之后在同四张空闲 GPU、冻结源码/镜像的新目录重新执行四个 pilot，重新测 `P` 和独立验收。当前没有可信的最小代码修复，也没有启动新 pilot 的依据。

## 独立评审

Astra 只读核对两轮 JSON、日志和分片存在性后，确认停跑符合方案，阶段 2 未完成，不能进入正式 freeze。评审指出本记录曾把恢复后的首次异步保存编号误写为 `#1`；上文已更正为恢复进程的 `#0`，并注明故障前进程也有独立的 `#0`。评审未发现其它新增阻塞项；CUDA 保存失败仍是继续 pilot 的技术阻塞。

## 后续同步 D2H 候选与新 pilot 停跑

固定镜像 MCore `FileSystemWriterAsync.preload_tensors(write_buckets, non_blocking=True)` 会对每个保存张量调用 `tensor.to("cpu", non_blocking=...)`，随后仍由原异步 writer 写完整 DCP。诊断模式只把该 partial 的第二个实参改为 `False`，不改张量、checkpoint 内容、保存频率、后台写盘和原生恢复。定向 CPU 合同核对 AsyncRequest 其余字段和 preload 输出一致，固定镜像 8 passed。此处只是尝试绕开失败的 D2H 页锁定路径，不能替代节点故障诊断。

在 GPU4 仍被外部作业占用时，继续固定 GPU1/5/6/7（GPU7 训练），先以与上次阻塞同步定位 run 相同的 `CUDA_LAUNCH_BLOCKING=1`、seed 419、A/F2 条件做单一新增差异的候选诊断：

```bash
python3 tests/ft/run_ft_dcp_diag.py --name diag-f2-blockingcopy-s419-20260924 --seed 419 --devices GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d --cuda-launch-blocking --blocking-copy
```

该独立诊断完成 10 步，F2 有效命中、11 次物理更新中最终保留 10 次；故障前及恢复后合计 10 次同步 D2H preload 均返回成功，原生完整 DCP 加载、输入和链验收通过，A 分类 `safe_discard`。容器墙钟 857.78 s、端到端磁盘峰值降幅 7,584,870,400 B；一份 6,917,735,465 B `*.distcp` 与诊断原件保留，未做 pilot 清理。入口验收为 `functional_verification_written`；首次离线来源验收发现一个评分子进程 pid 1648 写出返回事件，但父进程在故障前没有对应 `scoring_attempt` 记录。来源验收器按既定原则只接受父进程 `status=scored, exit_code=0` 的 child pid，删除“所有 child 返回都必须有父进程记录”的过严反向断言并加定向测试；固定镜像 8 passed。原 run 来源验收器 SHA-256 为 `0eb6e3214a3180b77677b5dfc1bc21669af329cba4acec627e4e92331eaad400`，离线复验为 `2fc881f77d95d9324e5f52520c308df87c8fc71c4080449e3533aed55eb385d8`，结果 A 目标 32 条同一执行来源复用 0/32。源码差异和原始状态分别见 [`source-reverification.json`](minimal_evidence/diag-f2-blockingcopy-s419-20260924-a/source-reverification.json)、[`source-sha256.json`](minimal_evidence/diag-f2-blockingcopy-s419-20260924-a/source-sha256.json)、[`acceptance-status.json`](minimal_evidence/diag-f2-blockingcopy-s419-20260924-a/acceptance-status.json)、[`dcp-diag.jsonl`](minimal_evidence/diag-f2-blockingcopy-s419-20260924-a/dcp-diag.jsonl)。这个成功诊断带 `CUDA_LAUNCH_BLOCKING=1` 和 `sys.settrace`，不能证明不带诊断开关的 A/R pilot 已修复。

随后把同一 `non_blocking=False` 改动限制在新 minimal 实验显式环境 `FT_MINIMAL_BLOCKING_D2H=1`，两臂共用；旧 FT1 默认不设置该环境。新 pilot 不设置 `CUDA_LAUNCH_BLOCKING` 或 `FT_DCP_DIAG_PATH`，不执行 `sys.settrace`。冻结清单见 [`pilot-blockingd2h-freeze-20260924.json`](minimal_evidence/pilot-blockingd2h-freeze-20260924.json)。因 GPU4 仍占用，计划在同四张 GPU1/5/6/7 上做 F2 seed 419 A→R、无故障 seed 421 R→A；实际只启动首个 F2 A：

```bash
python3 tests/ft/run_ft_minimal.py --name pilot-f2-blockingd2h-s419-20260924 --scenario F2 --seed 419 --order A R --devices GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d
```

A 在首次 DCP 之前的 backward 再次抛 `CUDA invalid argument`，用户态位于 Torch Inductor `copy_args_to_cpu_if_needed → torch.empty_strided`，同步 D2H 补丁尚未执行。容器墙钟 403.79 s、峰值降幅 274,583,552 B，验收为 `fault_not_valid_hit`，pair 自动 `stopped_for_review`，未启动 R 或无故障配对，无 DCP 分片可清。11:11:59 同一 [`container.id`](minimal_evidence/pilot-f2-blockingd2h-s419-20260924-a/container.id) 的内核日志再次出现 NVIDIA `os_lock_user_pages` 路径 order-9 分配失败及 `Cannot map memory ... 0x40000 pages`，见 [`kernel-page-allocation-excerpt.log`](minimal_evidence/pilot-f2-blockingd2h-s419-20260924-a/kernel-page-allocation-excerpt.log)。入口和错误原文见 [`pilot-f2-blockingd2h-s419-20260924-pair.json`](minimal_evidence/pilot-f2-blockingd2h-s419-20260924-pair.json)、[`trainer.log`](minimal_evidence/pilot-f2-blockingd2h-s419-20260924-a/areal/logs/caiyiwen/rewardtxn-ft-pilot/r-fault-integration/trainer.log)、[`acceptance-status.json`](minimal_evidence/pilot-f2-blockingd2h-s419-20260924-a/acceptance-status.json)。这明确否定了“仅同步 D2H 就能让无诊断开关的训练稳定完成”的结论。

### pinned allocator 取整与 host-register8 核查

固定镜像 PyTorch 2.9.1 的库包含 `pinned_use_cuda_host_register` 与 `pinned_num_register_threads` 选项字串，未找到 `pinned_max_round_threshold_mb`。在空闲 GPU7、相同镜像中，针对失败诊断 `exp_avg` 的 136,134,656 个 `float32` 元素（544,538,624 B），默认 pinned host 分配与 `PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:128` 均由 `host_memory_stats` 报告分配/保留 1,073,741,824 B；后者没有缩小该路径的取整量。启用 `pinned_use_cuda_host_register:True` 以及再加 `pinned_num_register_threads:8` 时，小型 pinned 分配与 D2H 也都通过，但仍保留 1 GiB；四种小测试均无新增 dmesg。脚本、精确 Docker 命令、输出与内核差分保存在 [`diag-pinned-rounding-20260924/`](minimal_evidence/diag-pinned-rounding-20260924/)。轻载通过不具有训练负载的判别力。

在父任务确认后，以新独立目录进一步测试 host-register8；seed 419、A/F2、GPU1/5/6/7、完整 async DCP/恢复与同步 D2H 诊断均保留，不设置 `CUDA_LAUNCH_BLOCKING`。仅增加 `PYTORCH_ALLOC_CONF=pinned_use_cuda_host_register:True,pinned_num_register_threads:8`，并在诊断模式记录 train_batch 与 DCP 处的 host allocator 统计：

```bash
python3 tests/ft/run_ft_dcp_diag.py --name diag-f2-hostreg-s419-20260924 --seed 419 --devices GPU-40b9b481-bf7d-cd0e-d659-065cfc44e794 GPU-1b3ba0aa-8679-a6d9-8c4f-d5ba5990bfe9 GPU-8c3a30a0-a187-3387-9e90-089a3c851bd3 GPU-6ab6332e-3b12-c9bf-21dc-3e6c4e57f91d --blocking-copy --host-register
```

它仍在首次 `train_batch` 的 Inductor `torch.empty_strided` 失败，未到 DCP preload/F2；`host-alloc-diag.jsonl` 的 `train_batch_begin/failed` 均记录 `reserved_bytes.current=13`、`allocated_bytes.current=0`、`num_host_alloc=3`，不能据此判断失败分配是否已进入统计。容器墙钟 406.87 s、峰值降幅 211,881,984 B、退出码 2、验收 `fault_not_valid_hit`。11:25:09 同一容器又出现 NVIDIA 锁页路径 order-9 分配失败和 `Cannot map memory ... 0x40000 pages`；原文见 [`kernel-page-allocation-excerpt.log`](minimal_evidence/diag-f2-hostreg-s419-20260924-a/kernel-page-allocation-excerpt.log)，配置和批次统计见 [`controller-config.json`](minimal_evidence/diag-f2-hostreg-s419-20260924-a/controller-config.json)、[`host-alloc-diag.jsonl`](minimal_evidence/diag-f2-hostreg-s419-20260924-a/host-alloc-diag.jsonl)。host-register8 不构成可信修复。

至 11:26，`/public` 可用 413,356,371,968 B（约 385.0 GiB）。所有旧失败 run、成功但非正式的同步 D2H 诊断、新失败 pilot 和 host-register8 诊断均完整保留；没有删除任何失败证据。正式样本仍为 0，四个新版 pilot 未完成，`P` 仍不可测。下一步应由节点/驱动侧针对同容器的 `get_user_pages`/NVIDIA 长期锁页与高阶分配失败进行排查和修复验证；在此之前不启动新的 A/R pilot。
