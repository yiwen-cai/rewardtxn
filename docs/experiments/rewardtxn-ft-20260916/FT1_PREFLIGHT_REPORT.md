# FT1 预检准备与当时阻塞（2026-09-21）

> 后续已开始 FT1；当前执行与独立验收见 [FT1_RUN_REPORT.md](FT1_RUN_REPORT.md)。本文保留启动前的历史预检状态。

**未完成完整 FT1，不放行 FT2。** 原有实现已提交为 `24b7285`，归档属性修正为 `6719426`；均为本地提交，未推送。此次新增无故障配对入口和 CPU 检查，GPU run 数为 0。

## 已完成

- 无故障配对入口：`tests/ft/run_ft1.py`。每臂 10 步，试验 seed 401（A→R）、409（R→A），与正式 seed 分离。每对固定同四个 GPU UUID，逐臂重新检查空闲；证据目录存在即拒绝覆盖。每对启动前要求至少 120 GiB 磁盘余量。
- 共同配置：单 Megatron training rank＋3 SGLang 实例，K=8、每更新 4 组；完整 optimizer、scheduler 恢复，async_save=True、每步保存。无故障场景 launcher retries=0。seed 通过共同的 `set_random_seed(seed, 'ft1-trainer')` 设置，实际派生 seed 写入观察记录。
- 共同观察：真实 RLVR 生成返回、实际 strict reward 子进程开始、PID/进程身份、训练入口与成功 optimizer 更新、结束时完整 native state。评分观察写入实际子进程的独立文件，避免父进程异步日志队列滞后。
- A 保持原生异步保存完成时机，A+R 的每次保存等待属于方法行为。两臂只在训练结束时共同 drain（原生 destroy 也会执行），不在 A 的每步保存后新增等待。
- 离线 smoke checker 检查退出、进程清理、10 次成功更新、状态组件和保存证据，另列自然 F4 候选。它**不是**完整 reward authority／保留链／重新加载 oracle，不能据此判正确恢复。

## 验证

固定镜像 `sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`，Python 3.12，CPU 容器（4 CPU、8 GiB、无网络、无 GPU），执行：

```sh
/opt/.venv/bin/python -m unittest test_ft1_preflight -v
```

4 项通过，15.101 秒，退出码 0：真实配置解析、拒绝漂移配置/非预定 seed、F4 顺序与缺样本/重复执行反例、真实 GSM8K 评分子进程的身份继承及回收。最后一项使用真实 RLVR `_compute_rewards` 和 strict subprocess；tokenizer 与生成响应为 CPU fixture，不是 GPU 推理。

证据：`p3_evidence/ft1-preflight-cpu-r1/{launch.json,source-sha256.json,tests.log,exitcode,inspect-final.json,cleanup.json}`。容器删除经第二次独立 inspect 核验，见 `ft1-cpu-verification.json`。新入口及 runner/checker 的 Python 语法检查通过。

## GPU 阻塞与剩余验收

现场快照及采集时间见 `ft1-resource-preflight.json`：8 张 H100 均被占用，没有满足每卡显存 ≤100 MiB、利用率 0 的四卡集合。未启动 GPU 容器，未终止占卡进程，未清理历史实验。利用率瞬时为 0 不代表卡已释放。

资源恢复后，先运行无故障配对（将四个现场空闲 UUID 传入）：

```sh
.venv-tq/bin/python3.11 tests/ft/run_ft1.py --seed 401 --devices GPU-UUID1 GPU-UUID2 GPU-UUID3 GPU-UUID4
.venv-tq/bin/python3.11 tests/ft/run_ft1.py --seed 409 --devices GPU-UUID1 GPU-UUID2 GPU-UUID3 GPU-UUID4
```

这些命令仅运行无故障部分；完整 FT1 仍需：

1. 两臂独立消费/保留链和真实 checkpoint 重新加载验收；不能用 smoke 的文件存在检查代替。
2. F1 实际生成服务 worker 与目标组映射；F2 两臂共同异步保存切点、有效命中及恢复证据。原生 A 在下次 save/load/destroy 才 finalize（`megatron_utils/checkpointer.py` 的 `_reap_finished_async_saves`、`wait_async_saves`），不能假定第二次 optimizer 前已有完成的上一次 checkpoint。
3. F4 自然时序：同组 K=8 全部 generation_complete 后的第 4 次独立评分开始，绑定实际 score child。新 checker 排除缺生成、过早开始和前四次含重复样本。尚无 GPU 候选证据；不人为插入生成屏障，不更换主终点。
4. F3 沿用已审核 v1 路径预先 N/A；X1/X2 实际结果入口适用性还需冻结。`workflow_executor.BatchTaskDispatcher._send_callback` 仅发送 task_id；strict scorer 使用逐次独立 Pipe 并 kill/join。不能借通知 callback 或直接改内部状态冒充真实奖励消息故障。本次未修改能力矩阵或重新分配 N/A 预算。
5. 既有 CPU 合同仍为 408 格已验证、184 待补和 8 项限制，新增 4 项不抵扣该合同矩阵。BASELINES/design/schedule/完整 freeze 与 functional_acceptance 尚未完成。

FT1 上限、正式矩阵与 F4 唯一主比较保持既定方案；预检不计正式样本。当前阻塞包含 GPU 资源和上述尚未完成的工程验收，不能将其概括为“只等 GPU 即可正式启动”。
