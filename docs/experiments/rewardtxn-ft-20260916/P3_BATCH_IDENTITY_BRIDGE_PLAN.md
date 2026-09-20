# P3 下一单元：真实 prepare_batch → train_batch 输入身份桥

2026-09-20；只读规划，未改生产/第三方、未运行 GPU。前提是已验收 DrawLoader、CallReturnRLVR CPU 合同；r2 修复后 r3 定向真实库通过 11.655 秒，不能把它另写成新版全套通过。评分仍限于 `official_call_returned`，内部 fallback 正式语义等待用户裁决。

## 决策

下一实现只做**方法内身份随真实训练输入传播，以及 train_batch 入口 update-intent**。直接使用原异步 `WorkflowExecutor.prepare_batch`、原 Grouped/优势计算/PPO microbatch 切分，不再加孤立缓存 fixture。CPU 在真实 train_batch 调用边界停止；不伪造 optimizer 成功、consumed 或 commit。GPU 接续时把同一边界包装委托给原 Megatron train_batch，接成功 optimizer/scheduler 与完整 async checkpoint 后，才形成真实消费权威。

## 已核实的路径与身份风险

| 实际源码接缝 | 结论 |
|---|---|
| `infra/workflow_executor.py:prepare_batch` | 首次缓存 dataloader/workflow/filter 的 generator；原 dispatcher 并发投递与预取。返回仅 `r.trajectory`，task_id 消失。不能按提交或完成顺序反查缓存；R loader/workflow 必须首次调用前安装。 |
| `infra/remote_inf_engine.py:GroupedRolloutWorkflow` | sample_idx 排序后 concat。tensor RLVR 路径不调用 `_normalize_group_rewards`（该函数只在 interaction 对象分支调用）；drop_incomplete_group 可整组丢弃。一般过滤由 executor 的 should_accept_fn 返回 bool，整条 trajectory 接受/拒绝，并非证明训练消费。 |
| `engine/megatron_engine.py:prepare_batch` → `infra/dist_rollout.py:DistRolloutCoordinator` | DP head 收集轨迹并搬至 current_device；`redistribute_trajectories` all-gather、按 attention_mask 长度分组分配给 DP ranks，再 broadcast 到 CP/模型并行组。整组顺序可变化，不能假定仍等于 loader 顺序。 |
| `utils/data.py:concat_padded_tensors/concat_batch/split_and_unpad_tensor` | tensor 按 dim0 concat/pad/split；list 虽会 concat，反拆时却整份 deepcopy 到每个轨迹；普通对象只取首值。因此普通 dict/list sidecar 不可靠。 |
| `trainer/ppo/actor.py:compute_advantages` | `batched_call` concat→原 `_compute_advantages`→split；reward scaling/clip/normalization、logprob/mask/turn 对齐与 advantage 计算改变实际输入，不能要求训练前 tensor 与缓存七字段全相等。原 dict 返回保留额外字段。 |
| `PPOActor._ppo_update` → `utils/data.py:split_padded_tensor_dict_into_mb_list` | 删除 rewards/tot_rewards/kl_rewards，按长度平衡重排 microbatch。只把 numel==B×L 的 tensor 放入随样本切分集合；一维 B 长度 ID 留在 not_to_split，原样复制到每个 microbatch，连 n_mbs=1 也可能顺序错配。 |
| `MegatronEngine.train_batch` → `_normalize_batch_input/_prepare_mb_list/forward_backward_batch` | 又一次 normalize、microbatch/pack/pad，再 optimizer_step。文本 `megatron_utils/packed_context_parallel.py` 构造 model_kwargs 只取 input_ids/position_ids/attention_mask/packed_seq_params 等及受限视觉字段，不把整个输入 dict 展开给 model；但额外字段仍可能进入 pack/pad/loss，不能仅据此宣布未知 metadata 安全。 |

## 最小载体与 train_batch 核验

项目窄 workflow 包装在 CallReturnRLVR 返回原七 tensor、真实 accept receipt 已存在后，添加四个固定 reserved tensor 字段 `_r_receipt_0..3`，每个 int64、shape=(B,L)。它们用四段有符号 int64 **无损表示完整 256-bit receipt hash**，每一有效 token 重复同一行身份，padding 由原 concat 填 0。不要使用截短 hash、进程 task_id、临时行号或可覆盖的 sample→latest 指针。

receipt hash 对应 R 自有不可变 receipt artifact，含完整 group/sample/attempt/epoch/owner/version 和 response/reward/tensor 引用；当前 `state.accepted` 仍用于有效授权核对，R 自己保存 receipt blob。hash 载体不是信任根：训练边界必须从方法自己的 BlobStore 读取、验全 hash、核当前授权和原 tensor 内容，不读 observer。四字段加在缓存七字段之外，避免改变 CallReturnRLVR 原 tensor 编码合同。

选择 token 同形是为了使用已存在的 concat/pad/split/长度平衡重排，非构建新调度协议。边界核验每行有效 token 的四字段恒定、完整 K、slot 0..K-1、无重复/缺失/旧 attempt；结合 input_ids/attention_mask 与原缓存按有效长度核对，不能仅检查 receipt 存在。相同 tokens 的两个不同 occurrence 必须仍可区分。

在 prepare_batch 返回与优势计算返回处记录方法自有输入/变换摘要；到原 engine.train_batch **调用入口**再次核身份及实际变换后 tensors 的完整 hash，持久 update-intent，再从传给原引擎的独立 dict 移除四个 reserved 字段。额外字段不进入正式 train_batch 的 model/pack/loss。compute_logp/compute_values 等前向入口也用小的共享 stripping helper 仅移除 reserved 字段，保留调用者原 batch 中载体；原前向返回仍由真实代码写回相同轨迹。CPU 必须测试 stripping 不修改原 batch、不改模型输入字段，GPU 再核实际 forward kwargs 无 reserved 字段。不要通过让字段一路进入 model 来假定兼容。

若 RPC/前向返回替换整个轨迹而丢掉载体，边界拒绝并定位该具体接缝，不靠内存对象 id 或排列顺序补回。首单元仅单 actor 的 CPU 数据路径；多 DP/CP 传播不默认通过，后续 GPU 验证要覆实际拓扑。四 token 同形 tensor 传输/存储成本归 R 并计量，不能宣称零成本。

## 下一 bounded CPU 实现（一个模块＋一份真实库集成测试）

建议新增 `scripts/ft/batch_identity.py`：receipt 载体、校验/移除函数和不可变 update-intent，复用 BlobStore；测试文件直接串现有 DrawLoader/CallReturnRLVR。无需改第三方/state，不建 dispatcher 框架。

1. 真实 `WorkflowExecutor.initialize/prepare_batch/destroy`，明确 synthetic engine 实现 agenerate/get_version；保留原 staleness/后台线程，调用两次 prepare_batch 验证首次 generator 缓存、真实预取与过滤后的映射。DrawLoader 只由原 producer 线程首次迭代，不能先在主线程 next 再迁移（现有单线程约束）；窄 loader 外层在 item 投递前按真实 draw 为 K slots 授权。
2. 原 Grouped，混合磁盘 tensor 命中/response 重评分/新 synthetic 生成、0/1；制造异步完成次序不同、长度不同、内容相同但 occurrence 不同的组；执行原 filter 拒绝一个整组并保留其 pending 身份，不能把 reject 当 consumed。正常生成仍立即评分，不能改成同步 rollout_batch。
3. 真正调用 `PPOActor.compute_advantages` 和 `ppo_update`，冻结 CPU 可执行 PPOActorConfig（明确 prox/ref 数据来源为 synthetic fixture）；使用原 batched_call 和实际 microbatch splitter，不自己仿制 reshape。终端 CPU engine 只在 `train_batch` 接口接收输入、核验、写 intent 后抛显式“到达 CPU 边界”信号，不返回伪 optimizer stats。测试 n_mbs=1 及一项 n_mbs=2 重排反例，后者只验证身份工具，不宣称放宽正式配置。
4. `DistRolloutCoordinator` 的通信需要真实分布式/current_platform；首轮可用原 `redistribute_trajectories` 在两个 CPU Gloo 进程核 DP 重排和 tensor 容器传输，再进入原 CPU PPO 路径。若当前平台绑定 CUDA，保留这条未覆盖，不能 monkeypatch synchronize 后称完整 GPU coordinator 已验证；不让它阻塞已能运行的真实 executor→PPO 桥。
5. 负例：交换载体但不交换 tokens、同 tokens 不同 attempt 错配、丢字段/半组/重复 slot、CAS 后到达、非法 padding 身份、sidecar list／一维 ID 在原 splitter 的错配演示。必须在 train_batch 委托前拒绝，真实 filter/reorder 的合法路径不得被误拒。

交付的 intent 至少含 parent generation（未接入时明确 null）、logical update ID、unique physical invocation ID、精确 group/sample/attempt/receipt refs、入参 tensor hash 和状态 `prepared_not_applied`。生成逻辑 ID 需由有序逻辑组集合及冻结更新配置确定、与物理执行分开；不得用 global_step 单独命名。没有实际 optimizer 就不往 state.prepare_generation 伪填成功证据。

## consumed authority 接续与 GPU 必需验收

当前 DrawLoader 全部 durable draw 均重放；`state._sets` 要求 drawn=consumed⊔pending，pending 仅 regenerate，`prepare_generation` 要求 consumed=已提交父代 consumed∪本次完整成功更新样本且无重复。**executor accepted、reward receipt、prepare_batch 返回、update-intent 都不能推进 consumed**。

下一 GPU 单元复用本 intent，绑定原 optimizer 的 update_successful 和实际 scheduler 调用；失败/回滚物理更新不能消费。随后原 async checkpoint 的真实 staging 切点、全 rank finalize、RecoverInfo/data snapshot/组件 full hash 与同一组 intents绑定，commit 后才成为 consumption authority。必须独立完成 GPU ownership/旧 writer fencing，不能用 cpu_contract scope 装 GPU。

恢复 loader overlay 接受的唯一排除依据是已验证 retained generation 的 consumed sample 集合；必须另行实现整 K 完整组排除与部分 draw batch 重打包、完整 prefix 覆盖，并与 snapshot 同切点。不能把当前 DrawLoader 的“全部重放”假称已支持 consumed，也不能把当前 pending regenerate 宣称跨 epoch tensor adoption。

GPU 必需：真实生成与评分→同样载体经过实际 actor.prepare_batch/DP分配/前向/优势计算→train_batch 无额外 model 字段→真实 optimizer/scheduler→完整异步保存→commit→新进程实际 DCP+RecoverInfo 加载→下一 update。先无故障再一故障；native r4 单 trainer kill 未触发 retry 的事实仍是独立门禁，不能假定重启已可用、也不能顺手改 launcher。原 async/prefetch/staleness 与正式矩阵不缩减，CPU synthetic 桥通过只解除输入身份接缝的不确定性。
