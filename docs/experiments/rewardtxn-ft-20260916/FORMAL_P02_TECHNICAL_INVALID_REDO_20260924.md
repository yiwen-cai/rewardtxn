# 正式 F2 第二对技术无效与唯一重做

原尝试：`formal-f2-p02-s8954-20260924-r`。原冻结种子 8954，顺序 R→A。R 在模型权重同步时失败：`llm_server.log` 记录 CUDA out of memory，`trainer.log` 记录 `/update_weights_from_distributed` 返回 400 及服务断连。随后控制器结果为 `technical_invalid`、`pending assignment lost its process; no reassignment`。`events.jsonl` 无 `signal_sent`，故障验收 `valid_hit=false`，`acceptance-status.json` 为 `fault_not_valid_hit`。故障目标未命中，原 R 不计正式样本；A 未启动。原目录和全部日志保留，不清理或覆盖。

原计划的失败规则已经执行：配对状态 `stopped_for_review`，正式调度器退出，没有继续第三对。错误明确发生在计划故障信号之前，符合冻结清单允许的**一次技术无效完整配对重做**。本次不修改训练、故障、验收或清理实现，不更换种子与顺序。新尝试命名 `formal-f2-p02-s8954-20260924-redo1`，新建 R、A 目录并执行完整 R→A；绝不把原尝试与新尝试拼接成一对。每对使用任意四张同时空闲的同型号 H100，并将设备 UUID 与新冻结清单哈希写入记录。若重做仍失败，停止，不再自动重试。

现场剩余约 383 GiB，高于原冻结的 56,202,498,048 B 门槛。CUDA OOM 的直接证据明确；现有日志不能可靠判定另一显存占用进程的所有者，因此不把资源竞争归因为已证实的外部任务。
