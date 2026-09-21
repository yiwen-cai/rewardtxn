# P3 真实训练接入验收（2026-09-21）

状态：**本次真实训练接入工程验收通过**。3 次真实 optimizer/scheduler 更新、3 代完整异步 checkpoint、96 个无重复 consumed 样本；64 个跨进程复用样本实际进入后续更新与提交。新进程完整状态精确加载通过。

可回查摘要：[training-integration-verification-r6.json](training-integration-verification-r6.json)；原始证据：[training-gpu-r6](p3_evidence/training-gpu-r6/verification.json)。

## 实现

专用入口 `scripts/ft/areal_rewardtxn.py` 运行原生 PPOTrainer。`training_adapter.py` 将已校验的 batch identity 绑定到真实 Megatron optimizer 与 scheduler 返回值；每步通过原生 RecoverHandler 保存 DCP、RecoverInfo 和 policy，再等待真实 async finalize，完整哈希/fsync 后发布保留链 token。消费权威只来自提交链，optimizer 成功而未提交的候选不会导致跳过样本。

完整状态包括模型、optimizer master/moments/step、scheduler、Python/NumPy/Torch CPU/CUDA/tracker RNG，以及 policy 和数据抽取记录。新进程通过原生 engine.load 加载，在权重同步和继续训练前重新读取完整状态并做逐内容哈希精确比较。`LocalNonpersistentObject` 明确标记为不持久化，不冒充已保存状态。

`training_replay.py` 从 durable draw 重建完整 K 组，只跳过已提交 consumed。完整且版本有效的旧执行产物经哈希、输入、tokenizer、评分调用和 lineage 检查后重新授权，保留原始调用 nonce；未来版本、过旧版本或未完整接受的样本重新生成。版本检查使用每个实际输出 token 的版本。

`state.py` 扩展最终 checkpoint 文件清单的目录前缀映射，以覆盖原生 DCP 动态 shard 文件名，并在 mutation 临界区退出时显式解除 flock，防止评分 fork 子进程继承描述符后继续持锁；CPU ledger scope 未更名。`batch_identity.py` 抽出准备函数，保留旧 CPU 入口的 PreparedBoundary 行为；`rlvr_replay.py` 新增默认保持原行为的扩展 hook。

## 验收范围

固定 AReaL checkout、0.5B 模型，4 张 H100 PCIe（物理 0/2/4/5；3 个推理服务、1 个 Megatron actor），K=8、U=4，每步 32 样本。仅同一容器 PID namespace、单 actor rank、单数据 epoch、每步一轮 PPO optimizer。

工程流程为首进程提交 step 0 后正常结束，再由 guardian 启动新进程加载该 generation 并完成 step 1、2。没有注入 trainer 故障，不计为自动故障恢复、恢复 RTO、正式配对样本或性能测量。

原生 local launcher 对 COMPLETED 也抛 JobException，原始 launcher 退出码单独保留。入口父进程用实际 Popen.wait() 记录 trainer 退出码；只有 trainer=0 才继续。未修改原生 launcher/scheduler 的判断。

## 测试

最终同版本 `training-final-cpu-r1` 合计 **21 项通过，114.711 秒，无跳过**：缓存 12、batch bridge 1、训练 replay 5、真实 optimizer hook 1、退出码 1、fork 锁回归 1；隔离容器退出 0，删除独立核验通过。宿主 state/fork/exit 另合计 23 项通过，1.189 秒（与前述 fork/exit 重叠，不能相加作为不同测试数）。

- `training-cpu-r3`：5 项通过，73.356 秒；覆盖只从保留链消费、完整组重排、DCP shard 损坏拒绝、未关闭 writer 拒绝接管、跨两次新进程复用及未来版本重生成。
- `training-hooks-r1`：1 项通过，24.341 秒；实际 PPO `input_=` 调用、identity 剥离、真实 torch.Adam 和 StepLR 顺序。
- 宿主 `tests.ft.test_state`：21 项通过，1.093 秒。
- 宿主 `tests.ft.test_training_exit`：1 项通过，0.066 秒；真实子进程退出 0/3 均原样记录与传播。
- `training-regression-r1`：原评分缓存 12 项与 batch bridge 1 项回归测试，合计 13 项通过，39.802 秒，无跳过。
- `test_state_fork_lock`：真实 fork 修复前稳定复现 BlockingIOError；显式解锁修复后，连同 state 与 exit 测试合计 23 项通过，1.189 秒。
- GPU `training-gpu-r6`：独立验收通过；重新哈希 20,754,258,568 字节 checkpoint 文件，确认 3 代保留链、96 个唯一 consumed 样本、64 个已提交 adoption、3 次 async finalize；新进程完整状态 exact_match=true。

CPU r1 因镜像 tests namespace 冲突导致旧 state 子进程导入失败；新增测试通过，未将该混合运行称为全绿。r2 的 3 项通过，r3 才是补全后的 5 项验收。

## 保留的 GPU 调试记录

| 运行 | 结果 |
|---|---|
| r1 | 发现 workflow 类型未继承原生 RolloutWorkflow，启动拒绝；已修复 |
| r2 | 真实生成评分后发现 PPO forward 使用 input_ 关键字，hook 参数不匹配；已修复并加入实际库测试 |
| r3 | 首步真实 optimizer、scheduler、异步 finalize 与 commit 成功；原生 launcher=1 导致旧 guardian 未进入 resume |
| r4 | 为加入实际 trainer 退出码凭据主动停止；保存 operator-stop 原因，不作为完成样本 |
| r5 | 新进程完整状态精确加载通过；评分 fork 子进程继承 mutation.lock 导致后续阻塞，已保存 fdinfo 证据后停止并修复 |
| r6 | 3 步真实更新/异步完整保存/提交完成；新进程精确加载、64 个复用样本实际消费；独立验收通过 |

所有运行使用独立目录与源码哈希，失败证据未删除；运行期间不热改源码。

## 最终 GPU 证据

首 trainer PID 678，恢复 trainer PID 2180，父进程实际 wait 返回均为 0；两次原生 launcher 返回均为 1（COMPLETED 分支），容器最终退出 0。验收器将 PID、generation、实际 update_successful、scheduler、async call ID、physical update ID、manifest receipt 和 adoption destination attempt 逐一绑定。

| step | generation | 说明 |
|---|---|---|
| 0 | g-53c60e57d24c41ebaa6e9ea9ac17a4b3 | 首进程，32 样本，warmup LR=0，真实 optimizer moments 更新 |
| 1 | g-b0cdc2910c274f78973447c11af9b942 | 恢复进程，32 个复用样本，LR=1e-6 |
| 2 | g-c0f9d14c89da43ee931f14938c0b3a96 | 恢复进程，32 个复用样本，LR=1e-6 |

相邻 generation 的模型、optimizer、scheduler 内容均实际变化。恢复前后同一 checkpoint 的模型/optimizer/scheduler/RNG 则要求精确相等。所有运行源码和独立验收器哈希与当前文件一致。

独立核验命令（读取原始证据，不启动训练）：

```bash
.venv-tq/bin/python3.11 tests/ft/check_training_integration.py docs/experiments/rewardtxn-ft-20260916/p3_evidence/training-gpu-r6
```

实际 GPU 启动 argv、training.yaml、phase.json、host-runner.py、guardian.py、固定镜像和 UUID、两次 trainer 退出、完整日志及容器 inspect 均保存在同目录。入口是工程探针专用，要求配置旁存在 phase.json，不能直接替换任意生产训练入口。全部六轮 GPU 容器和网络均以完整 ID 独立核验删除；最终物理 0/2/4/5 为 0% GPU 利用率、各 4 MiB 显存。

## 仍未覆盖

写入开始前持久化 pending writer 标记，只有 wait_async_saves 完成才解除。若在这段窗口崩溃或换 PID namespace，接管明确拒绝，尚不支持孤儿 GPU writer 自动隔离。多 rank、多数据 epoch、真实故障自动重启和独立训练连续性 oracle 仍需后续验收。

评分保持现有 official_call_returned 工程语义，官方函数内部异常/超时返回 0 仍不可辨识；没有替用户冻结正式评分口径。P2 的 184 项待补及正式 GPU 样本 0 保持不变。同步等待、CPU 校验、完整哈希开销均属于该接入，未声称性能无损。
