# P3 checkpoint 写入中故障恢复验证（2026-09-21）

状态：**CPU 与真实 GPU 独立验收通过**。checkpoint 写入未完成时 trainer 被 SIGKILL，原生自动重试、安全弃用半成品、完整状态恢复与样本重放均已验证。

[验收摘要](training-midwrite-verification-r1.json)、[GPU 主验收](p3_evidence/training-midwrite-gpu-r1/verification.json)、[补充验收](p3_evidence/training-midwrite-gpu-r1/supplementary-verification.json)。

## 范围与机制

本轮检验单 actor rank、同一 PID namespace、一次真实 checkpoint 写入中 trainer SIGKILL。使用此前用户已批准的公共 job 生命周期补丁，未再次修改 launcher 或其重试逻辑。原生 local_main 负责自动重试，controller 只启动一次 launcher。

`training_adapter` 在原生保存可能 fork 前，持久记录 pending gate：owner epoch/nonce/进程身份、generation、PID namespace，以及从实际祖先链捕获的 job supervisor/root 身份、命令、日志路径和单调时间。恢复要求旧 owner/root/supervisor 均已退出、同 boot/namespace，且唯一匹配生命周期收据的起止时间包含 gate 捕获时间，`cleanup.empty=true` 且无清理错误。缺失、错配或隔离未完成均拒绝接管。

这是受信任本地 job descendant tree 的清理凭据，不是跨 namespace 的存储 fencing。清理不产生 finalize 收据或 checkpoint 提交；原有 `state.select_recovery` 验证已提交链并处理不完整候选。CPU ledger 的 `scope=cpu_contract` 保持不变。

## 真实故障切点

专用入口 `scripts/ft/areal_midwrite_fault.py` 使用与既有 P3 一致的实际训练适配器。step0 先完成保存与提交；step1 optimizer/scheduler 已执行，原生 TemporalAsyncCaller 预加载 CPU 数据并 fork writer 后，在最后 bucket 的首个真实 tensor 写入返回时暂停。最后 bucket 必须仍有后续 tensor，其他 bucket 的线程此时尚未启动。

探针包装实际 MCore `filesystem_async._write_item`，保留函数签名；刷新 Python 文件缓冲用于检查实际 fd 的 inode、大小、路径及进程身份，不替代原生余下写入/fsync/finalize。trainer 从实际 async queue 取 writer PID，核对祖先关系、打开 fd、非零文件长度、剩余 tensor，以及缺失 `.metadata`，然后向既有 controller 提交固定故障事件。controller 通过已认证 trainer pidfd 发 SIGKILL。暂停 writer 的退出由公共 job 清理完成。

独立验收必须确认 writer 被回收、半成品未提交、原生启动器重试、加载 step0 完整状态、重放未提交的32样本、最终96个唯一样本/3次提交，并核对全部 job 和容器/网络清理。

## CPU 证据

- `p3_evidence/midwrite-cpu-r2`：固定已有镜像、无 GPU/网络，4 CPU、8 GiB、256 PID；4项测试通过，39.322秒，无跳过，源码哈希和容器删除核验通过。
- 真实 subreaper/orphan 测试验证：旧 owner 活着禁止接管；trainer SIGKILL 后清理不响应 TERM 的 writer；正确凭据允许 epoch 接管；错误 namespace/epoch/nonce/job/命令/时间、缺收据、cleanup=false 或带 error 均拒绝；旧 gate 不可用于新 epoch。
- 原生 MCore CPU writer 测试：真实两个 tensor 的 DCP bucket，启用探针时首 tensor 已写且 fd 开着、仍有一个 tensor 未写，pidfd SIGKILL 退出 -9；禁用时完整返回两个 WriteResult、退出0、无故障见证。
- 实际 PPO hook 调用合同回归通过。
- 项目 Python 单独执行原有 `tests.ft.test_state`：21项通过，0.830秒，日志保存于 r2。
- r1 保留：新增4项均通过；同时加入的旧状态测试有4 error、2个 subtest failure，原因是镜像子进程解析到其他 `tests` 包，找不到 `tests.ft`。r2 将该旧套件移至项目环境，未修改状态实现或旧测试。
- 固定镜像的 MCore writer/async 源码只读导出并保存在 r2，以支持切点复核。

## GPU 证据

运行目录：`p3_evidence/training-midwrite-gpu-r1`。启动前冻结源码与独立验收器，运行中不修改源码。使用现场空闲 H100 0/1/2/3，配置保持 K=8、U=4、3次有效更新、一次原生重试，32 CPU/128 GiB/4096 PID/16 GiB shm，私有 PID namespace 和内部网络。

| 验收项 | 实测结果 |
|---|---|
| 真实切点 | writer PID1695，fd96，`__0_0.distcp`已写272,437,394字节，剩703个tensor，`.metadata`不存在 |
| 唯一注入 | trainer PID682收到一次SIGKILL，controller以同一已认证pidfd确认退出 |
| 写入进程清理 | 原trainer job收据`cleanup.empty=true`，明确记录writer1695以-15退出；旧trainer/inference均清理 |
| 原生重启 | 仅一次launcher创建，local_main run0→run1；新trainer PID2221，assignment=already_fired，无手动重启 |
| 安全接管 | 新trainer匹配旧job收据后弃用pending候选，加载step0保留状态；模型/optimizer/scheduler/RNG精确相等 |
| 半成品 | g-8aa9ae22a0ca46a99473bfacde5536bd 无token/manifest/.metadata，保留abandoned标记，文件长度保持故障见证值 |
| 更新与消费 | 4次物理optimizer成功、3次有效提交、96个唯一样本；64样本跨进程复用后提交，故障候选的32样本全部重放消费 |
| 文件链 | 三代checkpoint共20,754,258,568字节重新哈希，token/manifest/intent/文件清单/消费链一致 |
| 全部job | 两轮trainer和两轮inference共4份收据齐全，全部cleanup.empty=true，旧job清理先于新job启动 |
| 运行收尾 | 新trainer正常返回，最终trainer job退出0；原生launcher的COMPLETED→JobException/退出1语义未变，controller/container退出0 |
| 资源与冻结 | 源码和验收器哈希全一致；容器/网络完整ID独立确认删除，GPU0–3均0%、各4MiB |

有效generation顺序：step0 `g-ad3ea4635b9b4669b9c28ba981caf8aa`，恢复step1 `g-d0417eaa1569453eb24fb77ec1491c78`，step2 `g-b888dde6f7414f15b210136dd55087c1`。

唯一SIGKILL到下一次optimizer成功 **178.488425730秒**，到下一次commit **244.403191468秒**。整个run **694.959470382秒**、分配 **0.772177189 GPU·小时**。这是单个受控工程样本，包含清理、重试间隔、启动、加载与验证，不能作为正式配对RTO或收益结论。

原生日志`is_recover_run=False`仍由默认RecoverInfo路径探测决定；A+R读取独立retained路径，实际恢复由native_state_loaded精确比对证明。

复算命令：

```bash
.venv-tq/bin/python3.11 tests/ft/check_training_fault.py docs/experiments/rewardtxn-ft-20260916/p3_evidence/training-midwrite-gpu-r1
.venv-tq/bin/python3.11 tests/ft/check_training_fault_supplement.py docs/experiments/rewardtxn-ft-20260916/p3_evidence/training-midwrite-gpu-r1
```

主验收器与补充验收器均在GPU启动前冻结。本轮只有一个GPU尝试，运行中未修改源文件；没有测试失败后的验收条件放宽。

## 边界

不覆盖多 rank、跨 PID namespace、多次连续故障或任意写入时刻。人工屏障将真实保存停在两个 tensor 写入之间，不宣称随机 I/O syscall 中断。正式配对 GPU 样本仍为0，P2的184项待补不变，评分语义尚未正式冻结。历史未打公共补丁时的原生重试 timeout 保留。
