# AReaL nofault pilot 实现说明

本入口仅验证真实 RLVR→GRPO→同步 Megatron recover checkpoint 的 3 步闭环。未实现 R、oracle、kill、F4 屏障、重投或恢复状态补全；代码交付不代表运行通过。

## 入口与配置

运行镜像 `areal-project/areal-runtime:v2.0.0-sglang`（现场 Python 3.12、Torch 2.9.1+cu129、Megatron-Core 0.17）；不安装或修改依赖。将仓库只读挂载 `/workspace`，全新独立输出目录挂载 `/output`，复制本目录 `pilot.yaml` 到 `/output/pilot.yaml`，预先创建 `/output/areal` 和 `/output/name_resolve`（官方 launcher 在启动前检查它们存在），容器仅暴露已选择的 4 张卡。

```bash
PYTHONPATH=/workspace:/workspace/third_party/areal python -m pytest /workspace/tests/ft/test_areal_pilot_contracts.py -q
PYTHONPATH=/workspace:/workspace/third_party/areal python -m areal.infra.launcher.local /workspace/scripts/ft/areal_pilot.py --config /output/pilot.yaml
```

独立 SPMD：SGLang d3p1t1＋Megatron d1p1t1，scheduler.type=null，actor/rollout v1。Qwen2.5-0.5B-Instruct，本地隔离 train.jsonl（预期 6373 行）映射 prompt→messages、label→answer，保留零基源行 ID，无验证/测试集。U=4、K=8、每训练批32回答、一个 PPO minibatch，LR=1e-6，3步。普通 saver/evaluator 定时器关闭，仅 recover 每步保存，包含 optimizer，使用 checkpoint scheduler。

首次 nofault 的 `recover.retries=0`：当前官方 local launcher 正常退出后也继续 retry 循环。此工程启动验证不测试自动重启；后续恢复探针必须单独配置 retries=3。没有修改 launcher。`async_save=false` 只属于首次 pilot，不能据此取消后续原生 async 验证。最大并发 rollout 8、context_length 2048 是短 pilot 资源界限；正式实验配置尚未冻结，不由此声称异步 pending 恢复正确。保留官方 `max_head_offpolicyness=2`。

## 观察链与语义

- `ObservedRLVRWorkflow` 只覆写 `_compute_rewards`：收到真实 ModelResponse 后排入 generation_done，再调用父类，不改 `_collect_samples`，不等待全K。原始 messages/answer、tokens/logprobs/version 保留在事件中。
- 顶层可 pickle callable 调用官方 `gsm8k_reward_fn`，显式 answer，保留官方 math_verify 默认语义（try_extract_without_anchor=True、precision=6、timeout=5秒，官方异常返回0）。实际 AsyncRewardWrapper 默认 pool/retry 不变；每个真实评分调用独立 UUID，记录评分子进程 PID/starttime/boot-id/cgroup。CPU 合同直接调用 verifier 并不证明子进程，真实运行必须从 PID 验收。
- 原生 GroupedRolloutWorkflow 按 sample_idx 排序后加入三个一维 int64 观察键：源行、原生整数 task_id、sample_idx。非整数 task_id 明确拒绝，未伪造哈希整数。键仅随真实内存 batch 流动，用于 batch_taken/train_batch/optimizer 关联，不是恢复输入。官方 concat/advantage/minibatch 真实库 CPU 合同检查顺序与数值等价；进入训练 engine 前剔除这些键。它们不改变 reward/group normalization。原生 microbatch helper 对非序列1D键整体保留，因此当前身份对应严格限定 `ppo_n_minibatches=1`；未来多 PPO minibatch 不能直接沿用此映射。
- `MegatronPPOActor.optimizer_step` 记录真实返回的 update_successful、LR和grad_norm；train_batch 记录完整样本身份与实际 tokens/mask/reward/version，update_id 贯穿本次真实更新。prepare_batch 取出集合不叫 ACK。
- 真实 Megatron save/load 返回后记录目录全内容 SHA256 清单；RecoverInfo.dump 返回后记录 metadata 文件清单与实际 step，RecoverHandler.dump/load 记录真实调用边界。同步 save 返回不是 async finalize。全内容 hash/CPU tensor 拷贝是显式公共观察成本，两臂应同口径；本轮不做5%开销门禁。
- 每个进程 incarnation 单独 JSONL，唯一 writer thread 串行 append。rollout 热路径只做非阻塞有界队列入队；写错误/队列满会报错，不丢弃后假称证据完整。评分子进程在 finally 做有界 flush，训练入口 finally 有界收尾；被 SIGKILL 时未刷事件仍可能丢失，因此日志缺失不能判为成功。

观察仅追加，从不读取 observer 日志参与恢复，不缓存、重放或替方法补回答。过程内关联用源行/task/sample，sample_attempt 只在 generation/reward 链；重启后的同 task_id 不代表同 attempt，必须结合进程 incarnation、未来 recovery epoch。首次 nofault 无故障因此不推断跨重启保留链。

## 验收与后续缺口

主任务必须检查：真实 3 次 optimizer_end 且 update_successful=1；每次 train_batch 是32个回答、4个完整8样本组；每步 checkpoint_save_returned 有实际非空 DCP 文件且 with_optim=true；metadata_written 包含 dataloader_info.pkl 和 step_info.json；pilot_training_returned 和 launcher 退出码。单看退出码/目录/日志中的 success 不够。此次文件清单不能单独证明 checkpoint 与 metadata 原子一致。

CPU 合同包含合成生成结果，只证明真实分组/处理库合同，不是 GPU RLVR 或恢复证据。尚未新增故障条件或状态摘要 oracle；P0 单 actor state-continuity 证据不能升级为本入口完整恢复通过。

后续恢复探针需要独立进程加载、model/optimizer/scheduler/RNG下一值及下一批比较、全rank保存完成、恢复 epoch 与 attempt 关联、自然F4可达性统计和受控故障身份握手。本轮完全不等待全K。RecoverInfo 只保存 dataloader/state counters，不保存 dispatcher pending；batch_taken/组/实际训练集合可观测，尚不能证明 cursor 前进后的全部 pending 被恢复。该缺口不得记为 PASS。F3无持久ACK接口，继续按已审部署范围 N/A。

主代理 CPU 验证：4 项通过，26.43 秒，退出码0；配置已用真实 GRPOConfig/分配解析器检查。nofault-r1 在 launcher 目录预检处失败，未开始训练；原始证据保留于 `pilot_evidence/nofault-r1/`。仅补齐目录后进入独立 nofault-r2，结果待核验。

环境要求补充：官方 SGLang launcher 的 `gethostip` 需要非回环地址，`network=none` 在 nofault-r2 中实测失败。使用本任务专用 `--internal` Docker 网络，CPU 地址解析及绑定已通过；不使用 host 网络、不开放端口、不修改官方网络函数。nofault-r3 保持源码和训练配置不变，仅修复上述环境。
