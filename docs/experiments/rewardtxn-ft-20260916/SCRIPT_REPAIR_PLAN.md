# FT 实验脚本修复计划（待实施）

日期：2026-09-20。依据：根目录《RewardTxn 已发表方法容错优势补充实验方案.md》（FT-v1）、`HANDOFF.md`、本目录状态及本地源码。本文只规划，不代表恢复已验证；本轮仅做只读源码检查并新增本文，未运行测试、安装依赖或启动 GPU。路径均相对仓库根目录。

## 1. 建议先决定的三件事

1. **A/A+R 优先选原生支持完整恢复的 v1 controller 路径，再实现 R。** `RecoverHandler._ensure_recover_supported()` 明确拒绝 `GatewayTrainController`（`_version="v2"`）。当前 FSDP DCP 保存链只显式包含 model/optimizer，未发现该链保存 scheduler/RNG；Megatron checkpointer 已有 scheduler/RNG 保存加载。建议先验证 v1＋Megatron 能否在选定四卡布局恢复，再冻结后端。若只能用 FSDP，必须先解决完整状态缺口并让两臂共享同一补丁，披露为本项目完整状态补全，不能只修 A+R 或声称未经修改的官方基线。
2. **F4 的屏障可映射性先于主比较实现。** `RLVRWorkflow._collect_samples()` 对每条回答生成完即评分；`GroupedRolloutWorkflow.arun_episode()` 并发执行这些完整 episode，并没有“先完成全 K 生成”的原生组屏障。同步 reward 还经官方 `AsyncRewardWrapper` 进程池执行，杀评分子进程可能由原生重建池直接恢复。先实证识别真实 PID、回答存活范围及全 K 生成／第 K/2 次评分执行的可达事件。不能为了做出收益而另建 RM 服务。若必须重排 workflow 为两阶段，两臂都要同改、明确基线含义变化，并在正式实验前修订 FT-v1；本文不自动批准这种变更或改主终点。
3. **F3 默认是待映射，不能将队列取出当 ACK。** 已检查的 AReaL dispatcher 使用内存待办和结果集合；`wait_results()` 清除选中 task，不构成持久消费确认。尚未发现 FT-v1 所需“checkpoint 完成、ACK 未完成、重投未确认任务”的完整接口。先追踪所选部署路径；若确无接口，预先记 N/A，不给 A/C 人造一个所谓原生 ACK。

实施主线为 A/A+R；C/C+R 在真实 ByteCheckpoint 接口适配可行后进入；RobustRL 官方 artifact 未确认，保持条件分支。不得用旧 B4/B5 或自写重启器代称 RobustRL。

准确率、正常吞吐及 5% 公共开销复测仍关闭，不作门禁；不重跑 LAST_THREE，不改历史统计或 FT-v1 样本矩阵。允许任意 ≥4 张空闲 GPU，实际单 run 布局、设备型号、UUID 和资源成本须成对一致并正式冻结。上海 smoke 的设备是 **L20Z**，不能按配额名称写成 H100。

## 2. 已核实接口与复用边界

本次 `git rev-parse HEAD` 核实：AReaL `b83d1f40196e5bd7d9f83092563443561870d550`；ByteCheckpoint `6f00167c153f3e65a67240aaa5ef4850a1c740fd`。以下是静态源码事实，均不等于运行时验收。

| 来源与真实符号 | 可复用部分 | 缺口／不能直接复用 |
|---|---|---|
| `scripts/phase3_auto_recover.sh` | 历史失败线索、恢复次数上限及禁止 `RTX_NO_SAVE_OPTIM=1` 的检查意图 | 按名称 `pkill`、`ray stop --force`、恢复后清空所有注入环境变量、覆盖 `RTX_EXTRA_MODEL_ARGS`；不能用作新入口；`--override-opt-param-scheduler` 不能直接继承为正确恢复策略 |
| `scripts/phase2_manifest.py`：`weights_hash`、`write_sidecar`、`audit` | 历史清单格式阅读和链关系概念 | 哈希只采文件头尾 4KB；`audit` 以 sidecar 存在认定提交，`n_manifest=0`；没有完整状态、保存完成及消费提交原子性证明 |
| `scripts/phase2_reconciler.py`：`build_plan`、`selective_replay` | 缺 reward／混版本的故障样例 | 仅输出 JSON/CPU 重评分，未送回训练；缺失记录的样本不能只靠扫描已有 reward 行发现；`B5_STEP_S=30.8`、`B5_CKPT_S=25` 不进入新成本统计 |
| `scripts/phase2_seal_rm.py`：`_v1_reward`、`_v2_reward`、`_cas_write_many`、`_seal_register` | 实际不同评分语义、现有 seal/CAS 回归素材 | 当前 CAS 按 logical ID 插入去重，SQLite 与 JSONL 分写；不能据此证明 attempt fencing 或训练状态事务。复用前须验证旧 attempt 抢占、重评分替换和崩溃边界 |
| `scripts/freeze_paper_inputs.py`：`_sha256_file`、`hash_path`、`git_state`、`verify`；`tests/paper/test_repro_freeze.py` | 全内容流式 hash、源码/镜像 provenance、输入变更拒绝用例 | 仅复用函数/验证思路，不直接套用旧 paper 默认资产和旧 freeze。完整 hash 本身也不证明保存快照一致性 |
| `scripts/trace_oracle.py`：`classify_event_log`；`tests/paper/test_trace_oracle.py` | 外部 authority、篡改和缺失证据负例 | 旧逻辑全日志计 `apply` 次数；新实验允许回滚后重算，必须新建按最终保留状态链判定的 oracle |
| `third_party/areal/areal/utils/recover.py`：`RecoverHandler.dump/load`、`RecoverInfo.dump/load`、`check_if_recover` | 原生恢复入口、训练 step、dataloader、saver/evaluator/stats 状态；恢复后权重版本同步 | `RecoverInfo` 多文件顺次写入；恢复 checkpoint 使用固定路径；目录存在不证明跨状态一致。需查选定后端保存完成语义；不得把原生模式开关等同自动拉起全部故障角色 |
| `third_party/areal/areal/trainer/rl_trainer.py`：`PPOTrainer.train`、`_save_training_state`、`_save_recover_checkpoint` | `actor.prepare_batch` 到更新、恢复 checkpoint 的真实接缝 | 必须记录 batch/group → 实际 optimizer 更新 → checkpoint 的关联；训练 global step 是否含多次 optimizer 更新须在所选 PPO 配置下核实 |
| `third_party/areal/areal/engine/fsdp_engine.py`：`_save_to_dcp/_load_from_dcp`；`engine/fsdp_utils/checkpoint.py`：`DCPState` | model/optimizer DCP | `DCPState.state_dict()` 只有 model/optim；不能声称已保存 scheduler/RNG |
| `third_party/areal/areal/engine/megatron_utils/checkpointer.py`：`generate_state_dict`、`get_rng_state`、`load_rng_states`、`save_checkpoint/load_checkpoint`、`wait_async_saves`；`engine/megatron_engine.py`：`save/load` | 原生 model、optimizer、scheduler、Python/NumPy/Torch/设备 RNG 及异步保存支持 | 需核实 `use_checkpoint_opt_param_scheduler`、`with_optim`、RNG 参数实际生效；recover 调用与异步 finalization 的关系以实际路径为准 |
| `third_party/areal/areal/infra/remote_inf_engine.py`：`GroupedRolloutWorkflow.arun_episode`；`infra/workflow_executor.py`：`WorkflowExecutor.submit/prepare_batch`、`BatchTaskDispatcher.wait_results/register_callback` | 原生分组、结果投递与批次汇聚接缝；保留完整组保护 | drop incomplete 会重试 dataloader 的**新 prompt**；不能冒称 FT-v1 的“同 prompt、同 K 合法替代”。callback 为结果回传，不等于 durable ACK |
| `third_party/areal/areal/workflow/rlvr.py`：`RLVRWorkflow._collect_samples/_compute_rewards/arun_episode`；`api/reward_api.py`：`AsyncRewardWrapper.__call__/_recreate_executor` | 真实回答、reward 和 tensor 构建链；原生 timeout/retry/坏池恢复 | 原生有存活回答与局部重试能力，A 必须保留；超时最终返回 0，独立 oracle 仍需判断该值是否正确；不能假设全部 worker 崩溃都需重新 rollout |
| `third_party/ByteCheckpoint/bytecheckpoint/api/save.py`、`load.py`：`save/load`；`checkpointer/meta_type.py`：`CheckpointState`、`SUPPORTED_FRAMEWORK_TYPES` | 官方 `model`、`optimizer`、`extra_state`，支持 `ddp/fsdp/fsdp2` | 不支持名为 `megatron` 的 framework；Slime 现有 Megatron optimizer 不可直接冒充 PyTorch optimizer 塞入 API |
| `third_party/ByteCheckpoint/bytecheckpoint/storage/_storage/local_storage.py`：`LocalStorageWriter.run/write_tracker`；`base_storage.py`：`CKPTCounter`、`_write_checkpoint_tracker` | 等待本地 futures、收集各 rank 写入失败后写 tracker/callback 的完成链 | `fast_saving` 返回不是完成凭证；须提供 `global_steps`，核实 callback 的触发 rank、失败传播、关闭等待与真实部署存储后端；不猜一个 `bcp.wait()` API |
| `third_party/slime/train_async.py`：`train`；`slime/backends/megatron_utils/actor.py`：`train_actor/save_model`；`checkpoint.py`：`load_checkpoint` | C 分支训练/模型及 scheduler 保存加载接缝 | Megatron 保存实际转交依赖库，须核查部署版本；模型保存与 rollout 数据保存分开，需统一 generation 关联 |
| `third_party/slime/slime/rollout/data_source.py`：`RolloutDataSource.save/load`、`RolloutDataSourceWithBuffer.add_samples/get_samples` | 数据游标字段与现有重新放入 buffer 的接口 | 普通 datasource 的 `add_samples` 抛错；WithBuffer 的内存 buffer 未列入父类保存字段，不能把调用成功当持久 replay；还须检查 rollout 插件是否重新生成已完成回答 |

官方测试 `third_party/areal/tests/test_recover.py` 大量使用 Mock；`test_grouped_rollout_workflow.py` 使用合成 workflow。它们可验证真实库逻辑，但不证明真实 checkpoint 或 GPU 恢复。ByteCheckpoint 的 `demo/fsdp_save_reshard.py` 使用 CUDA/NCCL，不能标成 CPU 通过。

## 3. 最小改动布局

保留 Phase 2/3 历史入口及产物；新增隔离 FT 入口，不在旧脚本上叠加更多环境变量。以下文件是**拟新增**职责边界，不要求一次性做完，也不引入通用插件框架：

| 拟新增文件 | 单一职责 |
|---|---|
| `scripts/ft/run.py` | 读冻结配置和 schedule、保存有效 argv/env、注册本 run 进程、监督原生恢复及计量、幂等记录已注入事件；提供只读 preflight/dry-run |
| `scripts/ft/faults.py` | 校验语义屏障、PID 身份和事件 nonce，执行一次精确故障并记录命中凭证；不决策恢复 |
| `scripts/ft/state.py` | R 的不可变状态 generation、全量清单、提交记录、恢复选择与 fencing；不把这套提交协议加给 A/C |
| `scripts/ft/replay.py` | 从 R 自有持久状态恢复组、重评分/补生成、投递、等待消费与提交收据；不从 oracle 取数据 |
| `scripts/ft/areal_adapter.py` | 实际 AReaL workflow/训练/恢复接缝；明确公共观察 hook 与仅 R 启用部分 |
| `scripts/ft/bytecheckpoint_adapter.py` | C/C+R 的 ByteCheckpoint 完整状态集成；仅接口审计通过后新增 |
| `scripts/ft/oracle.py` | 按独立规范离线判定最终保留状态链与受影响工作恢复；不导入 R 的 commit/reconcile 判定函数 |
| `scripts/ft/freeze.py` | 薄封装既有全内容 hash/provenance，生成/核验 FT 专用资产，不复制整个冻结工具 |
| `tests/ft/` 下状态、replay、oracle、runner 测试 | 对应崩溃切点、反例和真实接口层；fixture 与库调用结果分栏 |
| 本目录 `oracle_spec.md`、`design.json`、`schedule.json`、`freeze.json`、`functional_acceptance.json` | 规范、原 FT-v1 设计落盘、正式版本及验收索引；本次均不创建 |

如必须在第三方 trainer/engine 内添加 hook，限于上表真实接缝的小补丁，保留单独 diff/hash。公共 hook 只记录、暂停到故障命中，不提供存储、去重或重投能力；R hook 承担其方法能力和成本。先尝试已有 workflow/config 入口；涉及 AReaL `cli_args.py`、launcher/scheduler 或新依赖的改动，需要依其 `AGENTS.md` 在具体补丁准备后处理相应确认，不能提前假设已获授权。

## 4. 恢复必须成立的不变量

**状态单元。** 一个恢复 generation 绑定 `run_nonce`、配置 hash、父 generation、checkpoint 文件全量内容 hash、模型/optimizer（含 moment/step/master state）、scheduler（计数及下一 LR）、各 rank 的 Python/NumPy/Torch CPU/设备 RNG、data cursor/epoch/shuffle 状态、policy version、逻辑组和实际更新映射。预取已推进的数据游标与尚未消费工作要一起解释：或者保存足够的 pending 状态，或者按记录回滚/合法重新采样；不能只加载一个“最新 offset”而永久跳过工作。

**R 的提交顺序。** 完整组及正确 reward/attempt 校验 → 记录本次消费集合 → optimizer/scheduler 更新完成 → 按同一逻辑切点保存所有 rank 训练状态和 R 的消费映射 → 等实际保存完成并验证完整内容 → 原子发布不可变 generation/StepToken → 才进行该部署真实支持的消费确认。文件临时写入、flush/fsync 和原子发布的具体实现须覆盖元数据；固定路径原生 checkpoint 若会覆写，R 需在后端接缝选择独立 generation，不能边覆写边从旁轮询做 hash。保留原生异步能力，测量 R 额外等待/I/O，不强迫 A 使用 R 协议。

**恢复选择。** 未完成 generation 不可作为已提交状态；“checkpoint 完成、token 尚未发布”的候选必须依据预先持久的消费映射完成验证后才可补提交，否则回退完整父 generation。token 存在但文件缺失/内容不符必须拒绝，不能依 latest 指针静默续跑。已提交但 ACK 丢失时可重新确认，不能再应用已保留更新。并发恢复只允许一个有效恢复 owner/epoch，旧 attempt 即使迟到也不能越过 fencing；单纯 logical-ID 去重不足以处理旧 attempt 先抢占。

**回滚语义。** 物理执行和逻辑提交分开：崩溃后回滚的 optimizer 执行可重算且计成本；同一逻辑更新在最终保留链上应用两次才是 duplicate commit。不能拿同 step 编号或相同权重 hash 单独证明幂等。A/C 使用其原生保留链和 checkpoint 证据，oracle 不要求它们生成 R token，不替它们修复不一致。

**Selective Replay 闭环。**

1. 从 R 自有持久 group manifest 枚举 K 个预期样本；保留 logical group/sample ID，分开 attempt ID、恢复 epoch、policy/verifier 版本。manifest 必须覆盖从未写出 reward 的缺失项。
2. 只复用真实存活或 R 自行保存的回答；保存/恢复不仅是文本，还包括训练需要的 tokens、loss mask、旧 policy logprobs/versions 等。回答完整但 reward 缺失/版本错误才重评分；回答不可得就重新生成并计完整成本。
3. 封组后以 AReaL 实际 workflow tensor/prepare_batch 路径送入训练；不假设 `WorkflowExecutor.submit` 可以直接注入任意 trainer batch。若做 workflow 层缓存返回，必须保持原生分组、normalization、staleness 和 tensor 契约。C 使用真实 datasource/rollout 接缝，同样验证不会把 replay 回答再生成一遍。
4. 记录这组进入哪次真实 optimizer 更新、其结果被哪个完整 checkpoint 保留，以及该 generation 被重启加载后的状态；完成持久提交收据后才标记 replay 完成。`replay_result.json`、进程重启成功、reward 返回都不是终点。

最小固定输入训练需比较“连续执行”和“保存→新进程加载→同一下一批”的 model、optimizer、scheduler、RNG 后续取值及下一数据批。在线异步实验核查上述逻辑不变量，不要求 A 与 A+R 随机轨迹逐字相同。

## 5. 故障、oracle 与资源隔离

故障控制器使用冻结 `event_id`/seed/目标逻辑组，worker 到达真实代码边界后报告并等待释放；控制器核实命中状态再注入。F2 必须在真实 optimizer 完成而 checkpoint 未完成；F3 必须有真实 ACK 前窗口；F4 必须说明“已完成 K/2 评分”还是“第 K/2 次评分执行中”的精确切点并于正式冻结前消除歧义，两臂相同。X1 使用真实不同评分函数/config，X2 仅走真实 callback/重试入口，不能直接改对手内部状态。

每个目标登记 PID、`/proc` start time、boot ID、进程组/容器或 cgroup、run nonce、父子关系及角色；信号前再次校验，PID 重用或身份不符拒绝。优先独立 job/container/cgroup；Ray 服务、端口、临时目录均 run 级隔离。只清理登记且身份匹配的本 run 进程；不按名称杀进程，不关闭共享 Ray。控制器、只读 oracle 和证据盘不在故障目标范围内。

有效 argv 以数组落盘；恢复只变更经过验证的恢复路径和 attempt 元数据，其余与冻结配置逐项比对。scheduler/RNG 不靠覆盖参数绕过断言。schedule 追加记录 armed/fired/observed，重启只跳过已确认注入的 event，保留第 50/80 步的后续故障；故障控制器自身崩溃或 fired 状态不明记技术无效，不盲目再次杀进程。

独立 observer 拷贝输入、原始回答/训练入口证据供**离线**审计，authority 包含 label、回答 hash、真实 verifier 版本及独立重评分。oracle 根据训练 tensor、实际 optimizer/保存加载事件重建保留链；不能信任被测系统写的 `success` 或 `authoritative_reward`。authority、observer 输出和完整回答副本不挂载到被测方法的可读恢复目录；控制通道只发送屏障控制，不发送恢复数据。R 只读自己的持久状态，其在线 hash/写盘均计成本。证据记录缺失导致无法证明进入 optimizer 时为 `unverifiable`。

原生丢弃不完整组可安全但无受影响组提交：AReaL 当前代码的“换新 prompt”不能自动作为 FT-v1 同 prompt 替代。先核查是否有原生可用映射；没有则如实记录安全丢弃、浪费与目标工作未恢复，或在正式前修订共同工作量规范，不由 observer 补回原 prompt。

报告分别判定安全性（是否错误提交）、训练继续能力（是否继续产生有效更新）与原任务恢复能力（受影响 prompt 或预先定义的合法替代是否提交）。安全丢弃后换新 prompt 继续训练，可以前两项通过而第三项未通过；不能将其写成安全性失败或训练无法继续。沿用当前终点时，相关优势仅解释为原任务恢复覆盖，不外推为普遍容错优势。

## 6. 分阶段实施与验收

| 阶段／依赖 | 操作及交付 | 通过条件／失败出口 |
|---|---|---|
| P0 接口决策（首先） | 补充实际部署源码/包对照；确定 A v1 后端、checkpoint 完成、F1/F3/F4/X1/X2 真实入口；查 C 的 optimizer 表示和 B artifact；写接口审计与 oracle spec | 每个 cell 有源符号、事件/PID 映射、运行时待证项；F3/F4 无映射按下述策略退出，不能以模拟接口占位 |
| P1 安全 runner＋冻结骨架（P0） | 实现 argv/schedule、进程登记、故障握手与只读 preflight；复用 hash 工具；用普通子进程验证多次恢复 | 冻结参数不丢失，PID 重用/外部进程拒绝，多故障 schedule 不清空，无需 GPU |
| P2 状态＋oracle 合同（P0，可与 P1 顺次集成） | 实现完整 generation、消费映射、fencing、独立保留链 oracle；先固定小模型/数据 | 部分写入、保存完成/token 前后、ACK 丢失、旧 attempt、并发恢复、伪造 reward/链等反例全被正确分类；合法回滚重算不误判 |
| P3 A/A+R 接入（P1/P2） | 基于下述早期 GPU 可行性结果，接 R workflow/replay/训练接缝；开启原生组保护、reward retry 和 RecoverHandler；包含短程 GPU 集成验证 | A 原生恢复凭证完整；A+R 在选定最小用例证明“故障→恢复/重评分→投递→optimizer→持久 checkpoint→再加载”闭环，再进入各 cell 预检；不要求所有方法在所有 cell 成功 |
| P4 C/C+R 补充分支（P2，独立于 A 正式可行性） | 验证 ByteCheckpoint 原生小例；映射 Slime model/optimizer 和 extra_state；同一公共 restart 包装供两臂使用 | 真实 optimizer/scheduler/RNG/data cursor round-trip；不支持 Megatron 表示则记录阻塞，不伪造 framework 参数。后端转换若必要，先固定同底座成对方案与补丁范围再实施 |
| P5 FT1 与正式冻结（P3；C 按 P4） | 按下列分层验收；冻结 capability/design/schedule/code/package/config/hardware/oracle，独立审查原始证据 | 正式 run 前关闭所有实现缺口；pilot 不并入正式；资源到位但工程没完成仍不能启 FT2 |
| P6 正式执行（P5） | 仅依 FT-v1 固定顺序运行、收集真实 RTO/成本；失败及 N/A 全保留 | 不换主检验、不追追加 seed；修代码则结束该版本批次、重新冻结，修前后不拼成正式配对 |

**早期 GPU 可行性检查：** P0 完成静态定位后，只补足最小隔离启动、进程身份检查和必要观察 hook，就用短程 GPU 运行验证 A 的原生保存加载及恢复路径、F4 事件是否可达。此检查先于完整 P2/P3 适配，不等待全部 CPU 开发完成，也不等待完整正式 runner。不可达时先按既定退出规则处理，避免投入不适用的完整实现。P1 与不依赖这些接口的 CPU 合同工作可继续。早期探针和 P3 集成运行单列为工程验证，记录次数、GPU·小时与预算，不冒充 FT1 成功凭证或正式样本；未核实实际成本前不声称原总预算已覆盖它们。

**CPU 分层，不能互相替代：**

- 合同 fixture：六 cell 各 100 个预定调度，成对同 payload/顺序；覆盖缺样本、全 K 不完整、部分写入、token/ACK 切点、重复重投、旧 epoch、并发恢复、oracle 篡改、controller 重启及后续注入。无法映射 cell 的 fixture 只证明规范模型，不使 N/A 变 supported。
- 真实库调用：后续运行 AReaL `tests/test_recover.py`、`tests/test_grouped_rollout_workflow.py`、`tests/test_async_reward_wrapper.py`，逐项标 Mock 范围；新增实际 `RecoverInfo` 文件 round-trip/损坏测试和可在 CPU 支持的 ByteCheckpoint save/load 小例。真实库依赖 GPU/进程组的项目留 GPU，不 mock 其 checkpoint 后声称实测通过。
- 固定小模型：在可用的真实 checkpoint 库路径验证更新后重新加载、optimizer moment/step、下一 LR、RNG、数据游标与保留链；只验证状态连续性，不开展终点准确率评估。每项注明 fixture／真实库／真实后端及执行命令、退出码、依赖环境。

**GPU FT1 仍沿用 FT-v1 上限：** A/A+R 最多 16 次（各 2 次无故障 10 步，六 cell 各 1 对）；C/C+R 最多 8 次；B 若 artifact 可用最多另 8 次。原生最小恢复路径与完整状态 round-trip 单独验收通过后，再做各 cell 故障预检；若需额外调试 run，独立标工程调试并记录预算，不挪入正式分母。每个适用 cell 的工程通过条件是：故障确实命中、目标进程或消息行为有证据、观察链完整、独立 oracle 能可靠分类、runner 能结束并收集结果。成功恢复的判定另需受影响工作真实提交和保存加载证据；有效命中后的安全丢弃、方法性超时或错误提交是 pilot 结果，不能仅因此删除 cell、改为 N/A 或补强基线直到成功。R 的实现缺陷须修复并重新验证、冻结；其他配置或观察缺陷也须解决后再进正式矩阵。pilot 结果不并入正式统计，恢复收益大小不作为功能预检门槛。

正式 A 70 对/140 run、C 20 对/40 run、A 重复故障 6 run 及 B 条件分支保持 FT-v1。F4 仍是唯一预设 RTO 主比较，样本数、seeds、统计阈值和置信区间方法不由脚本修复修改。F3 若无真实映射预先 N/A，预算不改投别的 cell；F4 若无映射，原主检验记未执行，在产生正式结果前另行修订并冻结设计，不能改选收益好的次要 cell。后续能力表状态更新须有证据；本文不改 `capability_matrix.csv`。

## 7. 冻结、结果分类与剩余风险

正式 freeze 必须记录：完整源码 commit＋dirty/untracked 补丁内容 hash、两臂公共/R 差异、镜像 digest、实际部署 tar hash、有效 argv/env/config、依赖版本、数据/基座模型完整 hash、后端与 rank/角色拓扑、checkpoint/重试策略、独立 oracle 版本、故障语义与 schedule、计时来源、真实 GPU 型号/UUID、CPU/内存/磁盘预算。每对连续运行且硬件一致；换硬件不能混作原冻结配置。

现有冻结工具默认核验源码仓库干净，工作区已有大量未跟踪历史产物。实施时创建只含所需源码和明确补丁的隔离快照，记录它与当前来源的对应关系并验证部署包内容；不删除历史文件，也不为满足冻结门禁把全部历史产物提交进仓库。

当前 `OVERLAY_PIN.json` 的 rewardtxn/slime commit 为空，不能作正式 freeze。9 月 17 日 `shanghai_overlay_smoke_audit_dlc5yvbiisbeyv8e.json` 绑定 tar `bee5d1a6…`、明确 commit 和 L20Z，证明那一部署包的环境 smoke；正式运行须核对实际新包，不能复制旧 smoke 的 PASS。此前 `70c4392d…` 的旧包审计只保留历史用途。完整 optimizer checkpoint 的磁盘峰值须实测估计，再按 FT-v1 留余量；不清理历史实验释放空间作为默认动作。

结果至少输出：`correct_recovered`、`safe_stop`、`invalid_commit`、`timeout`、`technical_invalid`、`not_applicable`、`unverifiable`；分别记录安全性、首次恢复、目标工作量完成，不能压成一个 success。RTO 从实际注入到“角色可服务且受影响工作首次 oracle-valid 持久提交”，其他组提交不算。30 个最终保留更新达标但目标故障工作未解决也不算正确恢复。注入至恢复 GPU·秒包含保留/备用卡，另报重复 token、丢弃回答、verifier 次数/CPU 秒、回滚更新、I/O/checkpoint 和整 run 成本，取消历史常数节省估计。

有效命中后的方法超时、错误提交或自身资源耗尽留在分母；未命中、外部抢卡或 observer 损坏为技术无效，原记录保留，同 seed 最多补一次。单故障 900 秒、30 步 run 45 分钟、多故障 100 步总 120 分钟及最多 3 次原生自动重试沿用 FT-v1；各层重试分别记录，避免 wrapper 乘上原生 reward 重试却隐去成本。

尚待实证的关键风险是：v1 后端完整恢复及四卡拓扑；A 的 F1 原生重启/重连实际路径；F3 ACK 与 F4 屏障；AReaL 换 prompt 丢弃策略与目标恢复定义；R 缓存 tensor 到训练消费的接缝；异步保存跨 rank 完成；Slime Megatron optimizer 到 ByteCheckpoint 的兼容表示；RobustRL artifact。任何一项都不能由环境 smoke、fixture 或历史 Phase 3 结果替代。

工程时间仅作依赖估计：P0 约 0.5–1 工作日，P1/P2 约 1–2 日，P3 约 2–3 日，P4 约 1–2 日，P5 约 0.5–1 日加 GPU 排队；接口不兼容需重新估时。原 FT-v1 的 5–8 日工程预算和 209–352 GPU·小时仍是规划值，不是资源承诺。下一步应先交付 P0 的真实接口决策和最小补丁设计，再进入实施；本次规划到此停止。
