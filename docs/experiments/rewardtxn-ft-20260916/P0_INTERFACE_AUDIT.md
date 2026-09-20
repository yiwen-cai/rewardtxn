# P0 接口审计与决策

采集日期：2026-09-20（Asia/Shanghai）。范围：FT-v1、SCRIPT_REPAIR_PLAN 的 P0；本文件是静态源码审计，**没有宣告任何真实后端恢复 PASS**。本轮运行证据由主任务另行登记；CPU fixture、真实库、GPU 后端结果不得相互替代。未实施 P1+、未启动正式矩阵。

## 决策

1. A/A+R 优先验证 **v1 + 非 LoRA Megatron + 完整 optimizer + 同 TP/PP 拓扑**。`weight_format="dcp"` 确实有 Megatron 分支，不是仅适用 FSDP 的参数。配置默认 `async_save=False`、`use_checkpoint_opt_param_scheduler=True`。先跑原生同步最小连续性检查，再单列异步完成语义检查；这不是批准正式实验禁用高效异步能力。最终 checkpoint 策略须在正式前两臂共同冻结。
2. 默认同步路径静态有保存返回/全 rank barrier，再写 RecoverInfo 的顺序；异步路径仅 schedule 后返回，RecoverInfo 可先于 finalize。两者均不能仅凭目录、日志 `Saved recover checkpoint` 或 step_info 判断完整 generation，固定路径覆写/多文件元数据仍有崩溃窗口。不得给 A 偷加 R 原子发布能力。
3. **F3 在本次所审 v1 RLVR/GroupedRollout + 内存 dispatcher 路径为 N/A**：没有 checkpoint 后持久 ACK 和未确认重投接口。不是宣称所有 AReaL 部署都无队列，也不是方法故障后改 N/A。A+R 不得靠单臂添加 ACK 把该 cell 变成配对适用。
4. **F4 尚未证明运行时可达，不能定为支持或不可达。** 并发 episode 每条生成后立即评分，并无全 K 生成屏障；调度上可能自然出现全 K 生成且评分尚未完成。先观察自然事件；若须暂停 reward dispatch 到全 K 生成才可构造条件，须标原定义未验证并先修订设计，不把这种约束默认为普通观察。将 workflow 改为两阶段是设计变更，不在 P0 内实施。主 cell 无映射时按 FT-v1 留主检验未执行，不能换主 cell。
5. C 保留条件分支：ByteCheckpoint 不接受 `framework="megatron"`；不能因它提供 generic state_dict 协议就认定 Slime Megatron optimizer 直接兼容。需转换模型 chunks、分片 optimizer/master state 与 scheduler/RNG/data cursor，并做真实同拓扑 round-trip。另一种选择是改用其支持的 FSDP/DDP 后端，但必须另定两臂共同底座，不能暗换框架。
6. X1 的 reward callable 重试是真入口；controller 的 rollout_complete callback **没有 reward payload**。X2 的重复通知可映射，旧 attempt 结果跨恢复竞争的映射尚缺。仅 POST 同一个 task_id 不能证明 FT-v1 X1/X2 已实施。

## 源码及完成链

以下路径均相对仓库根，行号对应本次 checkout；`A:` 为 `third_party/areal/areal/`，`C:` 为 `third_party/ByteCheckpoint/bytecheckpoint/`，`S:` 为 `third_party/slime/`。

| ID | 实际源码证据 | 审计结论 |
|---|---|---|
| A1 | A:utils/recover.py:198、266、418、445；A:infra/controller/train_controller.py:471、690；A:engine/megatron_engine.py:1009、1043、1897 | v2 明确拒绝；v1 save/load RPC 收集 worker 返回。RecoverHandler 固定 dcp，with_optim 由 no_save/load_optim 取反；Megatron dcp 转 checkpointer。非 LoRA 才创建 checkpointer；HF 无 optimizer，不能替代恢复格式。 |
| A2 | A:engine/megatron_utils/checkpointer.py:350、428、512、576、593；A:api/cli_args.py:943 | model.sharded_state_dict；optimizer.sharded_state_dict(is_loading=…, dp_reshardable)；with_optimizer 下 scheduler；默认 with_rng=True。load 在 scheduler 开关为真时加载。同步 save 后 barrier；异步在下一次 save 非阻塞 reap，load 前/engine.destroy drain；RecoverHandler 无显式 drain。 |
| A3 | A:engine/megatron_utils/checkpointer.py:258、408；A:utils/recover.py:53、97、292；A:engine/fsdp_utils/checkpoint.py:33 | Python/NumPy/Torch/设备 RNG/tracker 有保存加载。DP RNG 默认 data_parallel_random_init=False，只构造单份 DP 状态，不能未经验证声称多 DP 各 rank RNG 连续。actor 单卡缩小此风险。RecoverInfo 顺次写 step/saver/evaluator/stats/frequency/dataloader，不是原子快照。FSDP DCPState 仅 model/optim，完整状态证据不足。 |
| A4 | A:trainer/rl_trainer.py:826、867、900、968、1374；A:engine/megatron_engine.py:1091、1390 | ppo_update 到保存有真实接缝；实际成功 optimizer.step 在 engine rank。外层 global_step 不是天然一 optimizer step，必须记录 update_successful 和内层更新序号。 |
| A5 | A:workflow/rlvr.py:84、112、139；A:infra/remote_inf_engine.py:87、107、125、962、1003；A:api/reward_api.py:76、117、141 | 一条 agenerate 返回后即评分，gather 等完整 episode。回答在 workflow 所属进程，sync reward 经 ProcessPoolExecutor，坏池可重建和重试；kill reward 子进程不必丢 workflow 中已完成回答。请求 HTTP retry 不等于生成进程自动重启。 |
| A6 | A:infra/workflow_executor.py:326、359、367、570、655；A:infra/controller/rollout_controller.py:665、736、865、893 | pending results/callback 都在内存。wait_results 移除 active IDs；wait_for_task pop。callback 仅唤醒 Future，随后 RPC 取 trajectory；重复 callback 无 pending future 时不产生第二个结果。无持久 ACK/replay。 |
| C1 | C:checkpointer/meta_type.py:73、87；C:api/save.py:72；C:api/load.py:66；C:checkpointer/ddp_checkpointer.py:112、131、215；C:planner/common.py:147 | 支持 ddp/fsdp/fsdp2；DDP model 必须 state_dict，optimizer 经过 _init_optim_state 和 state_dict，加载 optimizer 再 model。Slime model 为 chunk sequence、optimizer 为 MegatronOptimizer，不能直接承诺兼容 PyTorch optimizer 初始化/布局。 |
| C2 | C:storage/_storage/local_storage.py:64、137；C:storage/_storage/base_storage.py:233 | local futures 全完成后经 counter 到 tracker；global_steps=None 直接不写 tracker，成功聚合仅 coordinator 写 tracker/callback。fast_saving 返回不是完成；须实测 callback rank、失败传播和所有 components 完成。不存在本审计可据以使用的通用 bcp.wait()。 |
| S1 | S:slime/backends/megatron_utils/actor.py:89、559；S:slime/backends/megatron_utils/model.py:937；S:slime/backends/megatron_utils/checkpoint.py:7 | save 最终导入部署环境 megatron.training.checkpointing.save_checkpoint，不能仅用 Slime commit 代替该依赖版本。模型/optimizer/scheduler 到 ByteCheckpoint 的适配未实现。 |
| S2 | S:slime/rollout/data_source.py:123、138、171、199 | 保存 cursor/epoch/group/sample index/metadata；WithBuffer.buffer 不在该保存字段中，add_samples 不等于持久重投。 |

## 故障位置、数据和 PID 边界

| Cell | 候选真实切点 | 目标及数据范围 | P0 结论/待证项 |
|---|---|---|---|
| F1 | A5：至少一个 response 返回、同组仍有生成请求在途 | 实际 SGLang/vLLM 生成服务 worker PID；不是 controller 或 reward pool PID。已完成 response 可能仍在 workflow 进程；未完成 KV 属生成进程。 | source-supported 边界；runtime-unverified 重启/重连、存活回答与最终训练。HTTP retry 不充当角色重建证明。 |
| F2 | A4：实际成功 optimizer.step 返回后、对应完整 checkpoint 完成前 | trainer engine rank PID；多 rank 必须记录其他 rank 是否连带退出；更新已在 GPU，旧 checkpoint 保留范围需查实。 | source-supported 边界；runtime-unverified 完整状态加载和回滚。杀 controller 不自动等价杀 optimizer rank。 |
| F3 | A6 在结果 pop 后不存在独立 durable ACK | consumer/controller 的内存队列没有未确认持久集合；不存在可合法对应的 ACK PID 屏障 | unavailable，仅本次所审 v1 路径 N/A。C 应用重启包装和消费接口尚未定义，分支阻塞；不能据此预先宣告未来 C 部署 N/A，更不能用 ByteCheckpoint callback 冒充消费 ACK。 |
| F4 | A5：同组 K 个真实 response 完成，K/2 评分执行切点 | 原生 sync reward 的 pool 子进程；workflow parent 存活则 response 仍可能可用，pool rebuild 可能影响多个在途 reward。两臂登记同类 PID 和损失范围。 | runtime-unverified；必须先明确“第4次执行中”与“4条评分已完成”含义，冻结前不得混用。K=8，计数按唯一 sample+attempt，不能混别组或把重试当新评分。 |
| X1 | A5：真实不同 verifier callable/config 的迟到结果在 reward retry 完成链返回 | reward 子进程→workflow scalar/tensor；版本还须独立记录，不靠版本标签造不同值 | runtime-unverified 注入映射；A6 callback 不带值，不能用它直接实现混版本。未适配的负例不能报基线失败。 |
| X2 | A6：callback→future→wait_for_task 真实链 | rollout worker 结果在其内存，controller 仅有通知；task_id 不天然是跨重启 logical ID/attempt/epoch | runtime-unverified；重复通知本身 source-supported，但“旧结果与新attempt竞争”尚未建立，不能篡改 pending_results 冒充公开接口。 |

N/A 限定接口缺失或预设分支不在范围，不能用于有效注入后的安全丢弃、方法超时或错误提交。能力表分别列静态接口、运行验证和正式矩阵资格，不用一个 supported=true 混淆。

## 版本与内容证据

审计开始、本文写入前采集（父任务同时工作，根目录不是冻结快照）：

| 仓库 | HEAD | tracked `git diff --binary HEAD` SHA256 | untracked 文件数 |
|---|---|---|---|
| rewardtxn | 7e6c1dd7755e2a9354ecc9f0f5f83b86996c186b | 0a9c511479e4c6919640e2b33db655a7d54b763d6bf042ff0e38ca5244a5bdb0 | 541 |
| areal | b83d1f40196e5bd7d9f83092563443561870d550 | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | 0 |
| ByteCheckpoint | 6f00167c153f3e65a67240aaa5ef4850a1c740fd | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | 0 |
| slime | a6272da0d4f3d0a08520c99a2f3b4f6c887960dc | ebf4e3999f0139f336820aa1883665f2ee86c205efb113021b18728cbcf14cdb | 1 |

areal/ByteCheckpoint `git status --short` 为空。Slime dirty 涉及 workflows、model shell configs、sglang_engine、fully_async_rollout、sglang_rollout，未跟踪 tests/test_fully_async_rollout.py，不是干净官方基线。untracked manifest 为 `git ls-files --others --exclude-standard` 顺序的 `[path,全内容SHA256]` 列表，UTF-8 JSON、ensure_ascii=False、separators=(',',':')；根 hash `dd9397dd1147be0631a0b55a2b89d524dad852000c95cef5992c6a180c6673f5`，Slime hash `a368124564997d2812ed857b42ed9fc1b8e8a79b1838b3d8d255af2fdf31b51e`。这些是采集时快照指纹，不是可部署 freeze，后续须保存完整清单及 diff。

## 建议运行验证（主任务负责，以下不表示已执行）

- CPU 真实库：在匹配 checkout 的环境执行 `python -m pytest tests/test_recover.py tests/test_grouped_rollout_workflow.py tests/test_async_reward_wrapper.py -q`，逐项披露 Mock。独立 RecoverInfo 临时目录 round-trip/缺文件，不将之称完整 checkpoint。
- Megatron：actor 单 rank 最小完整模型，固定一批执行→保存→新进程加载→同下一批；对照 model、moment/master/step、scheduler 下一 LR、所有 RNG 下一值和下一数据批。必须通过真实 engine dcp 路径，不能只 torch.save 字典。配置 non-LoRA、no_save/load_optim=False；原生同步与 async 各记结果。
- Async：记录 schedule、各 rank finalize、RecoverInfo 文件出现顺序，等待必须在所有 rank 调用 `checkpointer.wait_async_saves()`；probe 的额外等待不能悄悄加入正式 A。固定恢复路径连续两个保存是否覆盖未完成前代也需观察。
- F4：先以实际 RLVR + 原生 AsyncRewardWrapper 记录 generation_done/reward_start/reward_done 的 group/sample/attempt/PID；K 完成条件必须由真实事件证明。用小真实 callable 可验证 pool 机制，但不能冒充正式模型/评分成本。未满足切点则不注入、不计方法失败。
- ByteCheckpoint：先用官方支持的真实小模型保存加载验证 tracker/callback，再核查部署 MegatronOptimizer 的 state/shard 布局。前者 PASS 不使后者 supported。

本轮具体命令、退出码、容器 digest、依赖、GPU UUID、日志及 RobustRL artifact 搜索请以主任务新增的 P0 运行证据索引为准；能力 CSV 在无证据时仍为 runtime-unverified，不复制历史 smoke 的 PASS。

## 本轮已收到的运行证据

- `p0_evidence/official_tests_r2.log` / `.exitcode`：三份官方测试 **62 passed，40.27秒，exit 0**。包括 Mock 和合成workflow，不能证明真实后端故障恢复。首轮 `official_tests.log` 因容器USER/passwd解析失败在collection退出，保留未覆盖。
- `p0_evidence/cpu_contracts.log` / `.exitcode`：真实 RecoverInfo 文件 round-trip、损坏JSON拒绝、缺cursor拒绝；ByteCheckpoint `framework=megatron` 真实 ValueError。未保存模型/optimizer，也不是原子性证明。
- `p0_evidence/source_provenance.json` 保存来源、commit、状态及关键源文件全内容hash。`runtime_imports.log` 记录 torch 2.9.1+cu129、Megatron-Core 0.17.0、mbridge 0.15.1、Transformer Engine 2.16.0、ByteCheckpoint 0.0.3；仅可导入不等于可恢复。
- `p0_evidence/runtime_snapshot.json` 为现场资源；镜像内容 ID 为 `sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`，RepoDigests为空，未声称有registry digest。`robustrl_search.json` 记录本轮作者主页/会议页检索，官方artifact仍未确认。
- `p0_evidence/probe_megatron_recovery.py` 是新建P0单actor验证脚本：两个独立进程，真实engine两步→同步DCP→下一步，与新进程load→下一步比较model/optimizer/scheduler/RNG全内容摘要；固定2×16 token输入、一份checkpoint。**不包含RecoverHandler、data cursor、RL目标组或进程故障**。执行入口 `python probe_megatron_recovery.py --model <本地模型> --output <全新目录> --mode baseline`，完成后同路径 `--mode resume`；设置PYTHONPATH指向当前AReaL。初始化日志表明engine内部effective seed为42（覆盖脚本早期seed211），两进程同配置；不得把211写成实际engine seed。r1真实两步optimizer成功，但探针未创建checkpoint目录导致保存FileNotFoundError，属于探针设置缺陷；原日志和脚本快照保留，不计方法失败。主任务已加创建新checkpoint目录并用独立r2目录重试，结果以其日志/JSON为准，本文不提前标PASS。

## 最终运行验收补充

GPU r2的保存成功后因探针不支持LocalNonpersistentObject而未完成比较；修复摘要后使用新r3目录。r3 baseline与独立新进程resume均exit0，model/optimizer/scheduler/RNG保存状态、下一RNG和下一步状态全内容比较一致。证据见 [P0_REPORT](P0_REPORT.md) 及 `p0_evidence/verification.json`。仅单rank同步MegatronEngine后端通过，不包含RecoverHandler/data cursor/故障注入或完整R。原文“待运行/不提前标PASS”为上述最终结果前的记录。
