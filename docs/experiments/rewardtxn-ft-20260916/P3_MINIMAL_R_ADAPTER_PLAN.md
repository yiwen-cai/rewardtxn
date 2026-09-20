# P3 最小 A+R 接入计划

2026-09-20；仅规划。当前正在运行的 native GPU probe **尚未作为通过前提**，本文不修改其源码/配置、不启动测试/GPU。下一单元先做 R 的 CPU 接口与持久化合同，再做无故障真实闭环，最后才做一次 trainer 故障。所有运行仍属工程验证，不扩大 FT-v1 矩阵、不冻结正式结论。

## 1. 决策与最小成功条件

限定既有真实 RLVR、v1、Megatron、一个 actor rank、固定拓扑、固定 verifier、K=8、U=4、ppo_n_minibatches=1；保留官方 AsyncRewardWrapper、GroupedRolloutWorkflow、staleness 和原生 SPMD launcher 恢复所有权。R 只在项目侧入口启用。A 保持原生数据/保存/重试行为，仅共享只读观察；不替 A 持久化 pending、重投、补 ACK 或发布 token。

闭环必须实证：真实生成/评分产物进入 **R 自有持久存储** → 新进程读取该存储并沿真实 workflow 返回 tensor → 原生 prepare_batch/优势计算/train_batch/成功 optimizer 与 scheduler → 不可变完整 generation → commit → 又一个进程经原生 engine/RecoverHandler 实际加载该 generation → 下一批真实 update。只输出 replay JSON、分数、commit token 或模型目录均不算通过。

最小故障先选 **一次 F2 工程切点**：已有完整父代，下一次真实成功 optimizer 后而对应新 generation 未完成时，精确杀单 trainer；原生 launcher 重启，R 回退父代并处理其持久未消费义务。不是 F4，也不以杀 reward pool 证明 trainer 状态恢复。F2 是否能正式纳入仍取决于完整状态/独立 oracle 验收，而非仅控制器命中。

正式 async 不改成 sync。既有同步 pilot 可作为单独工程对照，但同步闭环通过不解锁 async cells。最小 GPU R 接口应同时保留原 async staging/写入流程，以真实 finalize 作为 commit 前提；任何 R 额外等待计入 R 成本。

## 2. 已读源码给出的真实接缝

| 接缝 | 事实与接入决策 |
|---|---|
| `RLVRWorkflow._collect_samples/_compute_rewards/arun_episode` (`areal/workflow/rlvr.py`) | `_collect_samples` 先 `await engine.agenerate(req)`，随即评分；`arun_episode` 构造 input_ids/loss_mask/logprobs/versions/turn_ids/attention_mask/rewards。项目 workflow 覆盖这两个小接缝，保留原生成→评分顺序；不得重排成全 K 生成后统一评分。 |
| `GroupedRolloutWorkflow.arun_episode` (`infra/remote_inf_engine.py:87`) | 上下文提供 sample_idx，asyncio.gather 后按 index 排序并 concat；保留原分组保护/归一化。task_id 是进程内投递身份，不是跨恢复逻辑身份。 |
| `WorkflowExecutor.prepare_batch` (`infra/workflow_executor.py:1395`) | 内部 generator 首次缓存 dataloader/workflow，至少保持两批 pending。必须在首次调用前放入 R loader/workflow；后续换参数不会替换缓存。不要直接写 dispatcher 私有 pending 字典。 |
| dispatcher `wait_results/register_callback` | 内存移除/回调不是 durable ACK；F3 保持 N/A。R 的“消费提交”是自身状态协议，不改名成原生 ACK。 |
| `PPOTrainer.train` (`trainer/rl_trainer.py:699,826–827`) | `actor.prepare_batch` → compute_advantages → `actor.ppo_update` → `actor.step_lr_scheduler` → 版本更新/保存。必须分别观察实际 optimizer 和真实 scheduler 调用；不能在 optimizer 返回时把 scheduler_applied 写 true。 |
| `MegatronPPOActor.train_batch/optimizer_step/lr_scheduler_step/save/load` | 使用实际 tensor 与 update_successful、LR；一个外层 step 未必一个更新，限定配置也须运行时断言。DCP SaveLoadMeta.path 可在 R-only 项目包装内定向到唯一 generation，保留其他参数。 |
| `RecoverHandler.dump/load`、`RecoverInfo.dump/load` | 保存/加载 dataloader/saver/evaluator/stats 与 step_info；原生路径固定，不能让 R 从被覆写目录旁路 hash 充当不可变快照。实际 load 返回的 RecoverInfo 是 step/data 证据，目录存在不是。 |
| `MegatronCheckpointManager.generate_state_dict` (`engine/megatron_utils/checkpointer.py`) | 实际包含 model、optimizer、lr_scheduler、rng_state；必须检查 optimizer moments/master/step 和各 RNG/tracker 的真实 key/覆盖，不能只根据这些顶层键宣布完整。load 会调用 `wait_async_saves()`。 |
| 同类 `save_checkpoint/_reap_finished_async_saves/wait_async_saves` | async 请求 `schedule_async_request` 返回 call_idx；`maybe_finalize_async_calls` 返回 finalized IDs，内部有 collective。API async 返回或 queue depth=0 单独不够。项目 R-only 包装实际请求/完成函数，关联 call_idx→generation；不增加不对称 collective，也不虚构 `wait()` API。部署镜像中的 MCore 类签名/返回值仍须 CPU 导入核验。 |

PILOT_REPORT 已观察到 12 组训练时 sampler.samples_yielded=28、存在完整及部分未消费组；RecoverInfo 没有保存 dispatcher pending。该缺口是真实 R 需求，但不能据此先宣称 A safety 错误或所有恢复失败。

## 3. R 身份、授权与产物：先于生成写入

R 的单一持久写入者是 trainer 中的 R owner；其协程/线程串行提交 WAL。pool worker 只计算并返回结果，不继承有效 owner、不直接写权威 ledger。文件写/fsync 放到有界专用 I/O 执行器并 await 完成，不在 asyncio loop 同步长时间刷盘。

- logical_group_id 由 run_nonce、数据源全 hash、逻辑 dataset epoch、**该次 draw occurrence** 构成；同一 row 再次抽到必须新 occurrence。logical_sample=`group:index`，预先持久 K 个预期 slot。不能从已有 reward 行反推哪些样本缺失。
- execution/recovery epoch、owner nonce、attempt ID 独立于逻辑数据 epoch；在请求提交/评分前由 owner CAS 授权。真实请求 rid、workflow task_id、pool invocation 都作为 transport 对照，不替代逻辑 ID。
- 生成返回立即保存完整 response：prompt/input/output tokens、原 output_logprobs、逐 token output_versions、stop reason、请求采样参数/版本及内容 hash。之后按原顺序调用原评分器；返回后持久 reward 与 verifier 定义 hash，最后持久 tensor schema/dtype/shape/bytes/hash并 accept_result。
- 首版 verifier 在 run 内固定，不实现 X1。`state.authorize_attempt` 的 policy_version 是显式授权合同，不允许把恢复后的当前 version 填给历史回答。实际 output_versions 可能跨 token 变化，须逐 token 保存并按原 staleness 规则验证。**现有 scalar 字段不足以证明该轨迹授权**：CPU 接口阶段确定 schema2 的 request/admission version 与 response version-trace hash分别存储；不能把混版本轨迹伪装成单模型生成。若实际实现无法证明合法性，只能拒绝复用/按同 prompt 新 attempt 生成，不能抹掉 versions/logprobs。
- 迟到结果必须校验 owner epoch+attempt CAS+冻结 verifier/实际轨迹版本。重复同 attempt 同内容幂等，冲突内容拒绝；旧 attempt 先抢占也不能成功。
- 跨进程复用旧产物必须新 epoch 明确授权新的 adoption attempt，保留 origin attempt/原模型轨迹/内容 hash；重新登记不代表重新生成。staleness 不合法、回答缺失或无法独立验证时，以预先允许的同 prompt 同 K 合法替代重新生成并计成本。

R 存储 layout 最小为 `r/data_wal`（draw/update obligations）、`r/groups/<id>/manifest`、内容寻址 response/tensor blobs、`r/generations/<gid>`。atomic publish 必须先 fsync blob、再发布引用；恢复不读取 observer 目录。此处不是创建通用数据库/插件框架。

## 4. cursor 与 pending：R loader overlay，不替换调度器

项目入口将原 StatefulDataLoader 包为一个窄代理，保留 batch_size/len/state_dict/load_state_dict/iterator 合同，**在 RecoverHandler 初始化/首次 prepare_batch 之前安装**。优先项目侧 trainer 子类/构造后接缝，若无法在原 load 前接入，使用 R-only RecoverHandler.load 包装安装代理后再调用原方法；不能先训练再补装。

1. 代理对一次底层 `next(batch)`、完整 batch 的 draw WAL 写入及 after_state 快照使用同一锁；只有 WAL 与快照已落盘才向 workflow generator yield 任一 item。一个 batch 中尚未 yield 的其他 item 也已列入 pending。
2. `state_dict`/generation 数据快照共享该锁，避免 checkpoint 捕获已推进 cursor 而 draw WAL 未落盘的窗口；崩溃在 WAL 前时，恢复到上一份持久 loader state，不承认纯 RAM 游标。
3. WAL 枚举所有 drawn、已 commit consumed、pending（包括未提交、生成中、结果已回但未消费和回滚更新输入）。不能只记录成功完成的组；保留实际原生 sampler epoch/shuffle/state_dict blob，并区分“逻辑sample计数”与原生prompt游标，不能把 `state._sets.cursor=len(drawn)` 冒充 sampler.samples_yielded。
4. 恢复从所选完整 generation 的 native data state起，合并 R WAL 的后续 durable draw/rollback obligations。若将底层游标恢复到最新 durable after_state，必须证明该前缀内每个 draw 都由 consumed 或待重放覆盖，再优先 yield 未消费逻辑组；否则回退较早快照并按确定的 draw identity 去重。不能简单“原生 cursor 继续＋另塞 pending”导致以后重复抽取同次 occurrence。
5. replay 不直接塞 train_batch，也不篡改 dispatcher。loader 先供应这些真实 group 数据，R workflow 在原 `GroupedRolloutWorkflow` 内逐 sample 返回经过检查的已持久 tensor，或从持久 response 经原评分及原 tensor builder 完成缺失部分。真实 `actor.prepare_batch` 接收并走原 normalization/staleness/优势计算。旧 pending 失效时重新生成相同工作，不靠换新 prompt 掩盖义务。
6. 普通无故障 cache miss 每个样本仍按原“生成→评分”立即执行；不为了可重放冻结全部 reward。动态 batch/filter、多个 epoch、部分组丢弃的复杂映射不在首版，遇到未实现分支明确停止。

先核验代理能否被真实 `cycle_dataloader`、StatefulDataLoader恢复与 prepare_batch 缓存接受（含 `len/batch_size`、state schema、`sampler`访问）；若失败，不能退而采用只会同步的 `rollout_batch` 伪装正式异步路径。

## 5. optimizer → generation → commit → 实际 load

**更新映射。** 进入实际 train_batch 前，持久 update-intent WAL：logical update ID、group/sample/attempt集合、实际输入 tensor全hash、parent committed generation。真实 optimizer成功后追加physical update证据；scheduler真实返回后追加scheduler证据。F2若发生在generation intent创建前，义务仍可从这个WAL找回。失败/回滚的physical update不得自动标 consumed，重新执行保留logical ID并使用新physical ID。

**不可变路径。** R-only 接管本次 recover save 的路径解析：为 DCP 与 RecoverInfo 同时分配 `generations/gid/checkpoint/{dcp,recover_info,...}`；采用调用局部的 SaveLoadMeta/恢复配置副本或单实例包装，不永久修改全局 Saver staticmethod、不把所有 save都重定向。原 saver/RecoverHandler/engine调用仍各一次。恢复时仅 R 根据验证后的 generation 指向这两个路径，再调用原 engine load/RecoverHandler load。官方 launcher 的固定路径 `check_if_recover` 只是其启动提示，不能作为 R 选择 authority；须CPU证明即使原固定目录不存在，原 launcher仍会调用项目入口并由R完成真实load，不改 launcher判断。若做不到，停在此接缝审计，不能创建假“完整原生checkpoint”骗过检查。

**同切点。** data snapshot、消费映射、原 RecoverInfo、policy version、scheduler/RNG等必须关联到同一冻结 snapshot ID；DCP async staging时取得的状态与随后 metadata引用的 loader快照需使用同一份不可变快照，不能在异步 finalize 时重新读取已推进游标。freeze data metadata 的短锁和额外I/O是 R成本。

**真实完成。** 为每个 async request登记call_idx→gid→rank与实际writer身份；项目包装 MCore实例的原 `schedule_async_request` 和 `maybe_finalize_async_calls`，在原函数成功返回相应 finalized IDs 后才记 writer_closed。保存目录必须没有未登记写者；故障时未知/仍存活旧写者对应candidate不得promote。多rank暂不支持，不能把rank0完成推广为全rank。

不改 async_save 配置、不在每个save返回后无条件等待。首版 state 只允许一个 unresolved generation：允许下一段训练与这一代原生异步I/O重叠；**若下一次save需要新generation而上一代尚未finalize**，R-only在这个边界经官方 `wait_async_saves` 做有界背压，验证并commit上一代再准备下一代。这不是A的行为，也不是隐藏成本；记录等待时间/队列深度。后续多代队列不是本单元目标。原自然reap/退出flush完成时同样处理已有generation，无额外不对称collective。部署 MCore没有可绑定的request/finalize接口则 async接入阻塞，不退回强制sync冒充成功。

完整组件覆盖后复用 state 的 `record_evidence/commit_generation`，原子发布StepToken；没有ACK。新进程先验证 committed chain/所有权，完整候选可按预定规则promote，否则回退父代。记录实际RecoverInfo返回、DCP加载fullhash、scheduler计数/下一LR、policyversion与data overlay选择，再执行同一固定下一批。绝不能仅以R manifest内自报状态证明原生load生效。

## 6. state.py 复用边界：不能把 scope 改名就上 GPU

可复用的存储机制：安全相对路径、不可变原子文件、目录fsync、owner/attempt CAS、全内容inventory、父token链、未完成candidate验证/回退、最终链重复更新拒绝。保留现有CPU合同与负例。

必须新增的窄适配合同（另行实现，不在本文放宽）：

- GPU ownership scope：现 `acquire_owner` 明确只允许cpu_contract，participants是caller合同且静态列表。GPU版须关联P1实际kernel登记/完整PID身份、owner与可能异步writer，旧writer退出/finalize的实际证明；不能传scope=cpu_contract绕过。单trainer为唯一权威ledger writer；pool/inference结果经owner校验，不能拥有commit权限。native launcher负责重建，controller只提供生命周期身份事实，不提供恢复数据。
- 动态异步writer登记及未完成generation隔离；新owner不能接受旧进程迟到写入/重新commit。P1只看到旧trainer死不等于其孤儿writer已死。若无法获得writer退出证明，保守弃用其独立candidate并验证已提交父代不可能被该writer修改；scope授权须把这种隔离规则明确实现和测试，不能声称已有GPU fencing。
- 当前pending只支持 `action=regenerate`；先可复用该安全回退，但不能称 selective replay。schema2增加受hash保护的 response/tensor引用与origin/adoption授权，覆盖首次评分缺失与完整tensor复用。
- 当前payload只有三种hash，不保存文件；必须实际持久 blob并验证可读取、类型/dtype/shape、版本和prompt一致。不能只写hash或使用observer副本。
- `_complete` 当前要求固定组件集合，未包含显式data组件，且checkpoint所有文件必须被组件映射。真实DCP/RecoverInfo需要版本化schema增加data及明确auxiliary metadata清单，并对inventory全集覆盖；不要将RecoverInfo所有文件硬贴model或把所有文件列入所有组件以骗过检查。DCP一个shard可包含多个真实状态键，但映射须由实际sharded state key/metadata证明。
- prepare_generation/commit的成功与finalize receipts仍是caller合同；GPU适配必须从真实optimizer/scheduler/async finalize来构造。现有CPU receipt不升级为GPU证明。

## 7. 独立 oracle 与物理隔离

新公共 observer 将原始生成/评分/实际tensor和save/load证据发送到host侧只追加collector；原始证据根**不bind进方法容器**。socket只接收事件与返回有限控制/接收确认，不返回回答/reward/replay指导。A和A+R公共观察字段/成本一致。P1 pilot放在output的历史观察日志只作工程线索，不能作为P3满足隔离规范的最终布局。

独立规范化器不导入 `state.select_recovery/commit_generation` 的判定：从observer侧原始输入/label/真实verifier源码hash离线重算reward；保留实际 prepare_batch 与 train_batch tensor内容、dtype/shape、IDs、tokens/mask/old logprobs/versions，以及优势/有效行映射。旧pilot `batch_evidence` 只列部分字段，且未完整表达dtype/logprobs，不能原样称充分。

逐条链接：真实组K样本/授权attempt → 真实训练tensor → physical optimizer成功＋scheduler事件 → snapshot/finalize →实际保留generation →新进程实际load。R token是待校验声明，A用原生save/load链；两者都需实际状态证据。fixed-next-batch probe用独立导出检查model/optimizer/scheduler/RNG后续值与下一数据批；R在线hash/写盘不能记为公共observer开销。在线随机轨迹不要求A/A+R逐字节一致。

## 8. 分段最小实施与验收

| 单元 | 最小新增/修改范围 | 验收与停止条件 |
|---|---|---|
| P3a：先CPU | 新 `scripts/ft/replay.py`（R blob/group/draw WAL与loader overlay）；新 `areal_r_adapter.py`（项目workflow/窄接缝）；state.py仅显式版本化适配，不修改P1当前源；对应tests | 真实RLVR tensor schema/GroupedRolloutWorkflow和loader接口；CPU真实官方评分；完整/缺reward/缺response分别走真实调用或明确合成生成fixture；CAS迟到/重复、WAL crash、prefetch未yield项覆盖、旧epoch adoption拒绝/授权、worker不可写。CPU generation fixture不称GPU完成。核验真实MCore finalize签名/返回和原scheduler接缝。 |
| P3b：无故障GPU闭环 | 新独立R入口/YAML及observer evidence bridge；A共用只读observer；既有GPU profile只在前序运行结束后按独立授权接入 | 先A+R真实生成→持久→实际train/update→完整async generation/commit；干净结束后新进程从R持久pending加载至少一个真实已生成组（禁止observer读），进入实际train_batch及下一update/commit，再新进程加载。必须记录cache命中回答hash不变/对应真实生成调用为零，原评分缺失才调用原评分器。无持久pending则该项未测，不能拿空队列PASS。固定下一批检查完整状态连续性。 |
| P3c：一次故障GPU | 单event F2，单trainer；复用P1精确PID注入，无自定义launcher restart | 父代完整commit；后续实际optimizer成功未完成save时kill；旧owner/writer fencing→原生重启→验证父代→R WAL完整义务replay/合法重生成→原训练入口→新update→commit→再次load。独立oracle分别输出safety/continuation/affected work。实测超时/失败保留为method outcome，不补强A。 |

最小新增文件至多 `replay.py`、`areal_r_adapter.py`、独立入口、observer证据bridge/离线规范化器及各自定向测试；不一次造通用后端/资源插件。现有oracle.py只消费新规范化证据，不能反向给方法提供计划。需要改state schema的具体补丁须与现cpu_contract回归一起提交审阅，不就地取消scope保护。

GPU启动前必须完成当前native probe独立验收；其结果不由本文推断。P3 GPU均独立run、相同4卡拓扑/镜像/资源，启动前现场空闲检查，单故障deadline900s与原生重试上限分别冻结；工程无故障多进程验证按实际分配GPU秒计成本。scope不满足、无法绑定async finalize、data前缀覆盖不完整、observer可读反哺或真实tensor→update无法关联时停止，不开始矩阵。

## 9. 未支持 cells 与 AskFirst 边界

首版不支持F1/F4自然窗口、F3 ACK、X1 verifier切换、X2跨真实公开接口的消息异常、多故障/多epoch、多actor/多rank、异构拓扑、critic、dynamic_bs/filter变化、C/C+R、RobustRL。一次F2工程闭环也不解锁这些cells。自然F4不得通过冻结所有reward或改两阶段制造；以前CPU pool原生重试不代表F4。

已读取 `third_party/areal/AGENTS.md` 的明确规则：**“Ask first”** 包含 `Modifying config structures in areal/api/cli_args.py`、`Adding new dependencies`、`Changing launcher or scheduler logic`、`Deleting or renaming public APIs`。本计划避免上述修改：R配置单独项目文件、现有库、项目entry/workflow/loader代理和实例包装；官方launcher/scheduler源码不动。若需要向cli_args添加R字段、修改dispatcher调度/原生retry/check_if_recover行为或新增MCore/存储依赖，必须先形成具体可审查补丁并处理该明确边界，不能以“monkeypatch”绕过同一逻辑变更。调用既有公开workflow/loader接缝与R-only数据缓存不等于自动获准改调度器。

本单元仅此文档；未写实现、未运行CPU/GPU/矩阵、未修改运行中的P1任何文件。

## 10. 收窄后的 P3a 第一单元：draw WAL / loader overlay / blob（可实施合同）

本节优先于上表 P3a 的较宽范围。**首单元只新增一个项目模块 `scripts/ft/replay.py` 和对应 CPU 测试/fixture**，实现本节存储与loader合同；不同时修改 state.py/schema、GPU ownership、异步checkpoint、observer、真实reward workflow或训练入口。没有 GPU、选择性评分重放、训练消费提交能力的声明。这里只证明“持久记录了哪些 draw，以及崩溃后能以原身份重新交给loader调用者”。

### 10.1 限定与 API

限定单机POSIX、单进程owner（实际flock排他）、同进程两类线程：一个迭代线程与可并发请求snapshot的线程；num_workers=0、world_size=1、固定batch_size、固定drop_last/shuffle/seed与数据源hash、一个逻辑epoch=0、普通map-style数据集及JSON可表示的list-of-dict batch。不支持RDataset/远端prefetch worker、多迭代者、多epoch、动态batch、模型tensor序列化。blob接口仅收/返 `bytes`，不自行torch.load/pickle反序列化未知对象。

最小接口（命名可按实现风格微调，语义不能放宽）：

- `BlobStore.put(bytes) -> {sha256,size}`；`get(ref) -> bytes`：只允许内部digest路径，完整重算校验。
- `DrawLoader(base, root, run_nonce, source_sha256, loader_config_sha256)`：独占该root的owner锁；明确实现 `batch_size`、`__len__`、`sampler`窄代理、`__iter__/__next__`、`state_dict/load_state_dict`、`close`，不使用任意 `__getattr__` 转发可变底层操作。
- `state_dict()` 返回当前 **durable** draw-prefix descriptor，不返回正在修改的底层dict引用。`load_state_dict(descriptor)` 只用于新代理、首次交付前；加载后创建/恢复底层iterator并启用pending overlay。不得在运行中清空generator并rewind。
- 对外返回原batch数据的独立拷贝，并附项目命名空间 `_r_draw`（与原有键冲突即拒绝），至少含record sequence、occurrence、group ID、K；原prompt字段/顺序不变。本单元只交给CPU调用者，尚未声称真实workflow接受该字段。

真实调用接缝已经从本checkout核对：`utils/dataloader.py:create_dataloader` 产生 torchdata.StatefulDataLoader，普通训练使用 DistributedSampler、collate默认 `lambda x:x`、config.num_workers；`PPOTrainer._create_dataloader`只是该函数封装。`utils/data.py:cycle_dataloader`从epoch=0开始，对暴露的sampler调用set_epoch，然后yield from loader；`WorkflowExecutor.prepare_batch`首次缓存task_input_generator，其中每次取完整batch后逐item yield，后续参数变化不会更新缓存。RecoverHandler.dump调用dataloader.state_dict，load调用load_state_dict。因而代理必须提前装入、实现这些显式接口，不能只拦 workflow.arun_episode。

sampler代理仅允许首次/恢复准备时幂等 `set_epoch(0)`；顺序为设置底层epoch，再load所选底层state（不得在load后重置其进度）。cycle尝试epoch1时清楚报“首单元不支持多epoch”，而不是静默从0重播。有限fixture dataset须足够完成测试；该拒绝不是正式无限训练实现。所有iteration/state/sampler操作共用一个RLock；`base.next`不是锁内调用者之外可访问的公开接口。**这提供代理边界串行化，不声称torchdata本身线程安全**；不得让另一个组件持有base直接迭代/取state。

部署镜像torchdata实际 `state_dict/load_state_dict/iter` 恢复行为仍需下面真实CPU API测试确认：不能仅根据类型注解宣布已验证。num_workers=0消除底层worker持续预取，但不消除调用者已取batch里尚未yield的item，这正由完整batch draw记录覆盖。

### 10.2 最小磁盘结构与稳定 occurrence

不用通用append-log/数据库/后台WAL服务。一个draw事务对应一个不可变JSON文件：

```
r_draw/
  owner.lock
  genesis.json
  blobs/<sha256>
  draws/000000000001.json
  draws/000000000002.json
  ...
```

genesis固定schema/run/source/loader-config/K、初始底层state blob以及其摘要；owner排他锁须跨整个代理生命周期，fork子进程不能沿用owner写入权。新进程先锁定再恢复；锁冲突直接拒绝。这里不是state.py恢复epoch/GPU fencing。

每个成功发布的draw记录包括：连续 `sequence`、前记录完整hash（第一条指向genesis）、epoch=0、batch payload blob ref、底层after_state blob ref、本batch每个slot的 `occurrence` / group ID、K与prompt完整hash。JSON使用固定canonical编码、不接受重复键/非有限数。记录内容不能依赖PID、torchrun次数或本次重启时间。

occurrence取“此前所有已发布draw记录的item数量＋batch中slot序号”；group ID对 `(run_nonce,source_sha256,epoch,occurrence)` 的canonical内容做hash；同一source_row_id在不同slot/抽取次数出现仍为不同组。同一持久draw恢复重投时occurrence/group ID完全不变。K个sample逻辑slot可以预先列出，但本单元**不授权sample attempt，不声称评分完成**。

sequence/occurrence不单独原地更新计数器，而从连续有效记录推导。事务未发布且没有对外yield时，重启可复用该未生效sequence/occurrence；已经发布但还未yield时，必须读取原记录重投，不能重新抽取并覆盖相同ID。

### 10.3 严格的 `next → after_state → fsync → yield` 顺序

首次准备时，在epoch=0及实际底层iterator初始化条件下取得可恢复的“第一条draw之前”state，并原子持久化genesis；CPU round-trip测试须证明此state恢复出的第一批一致。不能假定调用state_dict不创建/改变iterator，须在实际torchdata版本上核对。

每个fresh batch只执行：

1. 获取代理RLock；确认本代理无另一活动迭代者、没有未完成的本地事务、没有优先待交付的replay batch。
2. 调用 **一次** `next(base_iterator)` 得到完整batch；若耗尽不写draw记录。此时仍不把任何item交给调用者。
3. 在同一锁内调用一次底层 `state_dict()`，立即序列化成独立immutable bytes；记录它是这次next之后的after_state。batch转受限canonical JSON bytes，记录prompt/hash与全部slot（包括调用者稍后才会访问的item）。
4. 分别写batch和after_state blobs：临时文件完整写→flush/fsync文件→不可覆盖的atomic publish→fsync所在目录。然后构造draw JSON，写临时文件→fsync→不可覆盖publish最终sequence文件→fsync draws目录。**最终draw记录及其引用全部持久之后**才更新内存durable highwater。
5. 从独立bytes构造返回batch/身份，释放锁，再把完整batch交调用者；不把原底层返回的可变dict或状态引用泄漏给另一线程。

步骤2之后、最终record durable之前若抛异常，代理进入poisoned状态，拒绝后续next/state_dict并要求关闭重开；不能捕获错误后在已推进的RAM iterator上继续。正常进程SIGKILL后，RAM游标消失；新进程只恢复最后durable after_state（或genesis），不会承认未发布的draw。无法把这条软件顺序升级成所有文件系统/存储故障下保证；声明POSIX本地文件与fsync模型。

state_dict线程必须拿同一RLock：它只能看到这次事务前或后一个完整durable prefix，不能看到“游标已前进、draw尚未发布”。慢next/fsync时snapshot会等待，这一开销属于R。禁止把长事务交给后台线程然后提前yield。未来真实async workflow非阻塞I/O问题不在此首单元；当前这是同步loader线程中的有界持久事务。

### 10.4 snapshot/WAL恢复覆盖与 consumed authority

snapshot descriptor最少包括schema/run/source/config、`wal_sequence`、该条record/genesis hash、其after_state ref、durable occurrence count；只描述draw，不包含“训练已提交”字段。它可以被独立保存再传给新代理的load_state_dict；descriptor本身不是权威commit。

恢复必须先完整校验genesis、从1连续到M的record hash链和所有引用blob（size＋全hash）、身份计数/slot覆盖。给定snapshot N时要求0≤N≤M且snapshot准确锚定第N条；WAL缺失/截断到N之前、同N不同hash、source/config/run不符都拒绝。M>N时不能无视durable尾部draw；本单元恢复base到第M条after_state，并将 **1..M的全部draw** 按原batch顺序装入replay队列，再从base的M之后继续fresh next。一次恢复过程内每个replay batch只交付一次；再次崩溃可再次重投，这是未提交工作重投，不是“exactly once训练”。

**首单元不接受任何生产consumed参数/布尔标记，不新增commit文件，也不导入state.select_recovery。** 因为尚无真实checkpoint retained-chain authority，全部durable draw都当未提交义务；CPU调用者返回“已处理”不能让记录从恢复队列消失。纯读取API可报告 `drawn = pending`、`consumed = ∅`。未来P3接入后才允许验证R committed generation所证明的consumed集合，并证明前缀覆盖；那是单独适配，不在本次提前设计通用token接口。

后续需要部分batch已消费时必须另行规定重打包与batch_size合同；本次没有consumed，完整重放原batch，不填充假prompt、不改变原batch大小来绕过prepare_batch要求。drawn“item/组计数”与未来sample集合及原生sampler计数分别命名，不混写成同一cursor。

### 10.5 blob与尾损坏处理

- blob地址仅由完整SHA256派生，引用带size；禁止外部任意路径、symlink和非普通文件。先写同目录唯一临时文件，再通过不覆盖已有文件的发布操作（例如link）发布；并发同hash同bytes幂等，已有同名但size/hash不符视为corruption。hash相等不免除读取时完整校验。
- draw记录同sequence同canonical bytes可幂等确认；同sequence不同内容一律冲突拒绝，不能后写覆盖。发生发布/目录fsync异常时poison代理，不自行猜测事务已回滚。
- 这里故意不用可截断JSONL尾日志：未发布的 `.tmp-*` 半文件和无record引用blob不进入prefix，可报告为orphan，首版不做GC；**最终命名record即使恰是最后一条，JSON截断、缺blob、hash错或序号缺口都fail closed**，不“修好尾巴”后跳过已对外发过的draw。process crash可能留下完整record但未完成目录fsync：若重启后它完整存在且引用校验通过，可作为新发现durable draw重投；若不存在则恢复上一prefix。不会把可见但损坏record当未发生。
- snapshot引用的prefix必须存在，不能因WAL更短回退后继续声称该snapshot有效。支持的是process-crash模型；底层持久介质丢写/破坏保留为拒绝恢复。

### 10.6 pickle的受信来源

batch/prompt和record使用受限JSON；只有**底层StatefulDataLoader自己产生的state_dict**可作为opaque pickle bytes保存。反序列化前必须校验本run自有root、owner、schema/run/source/config以及blob引用与全hash，再交pickle和base.load_state_dict。hash不是真实性/安全证明：root和genesis必须来自本方法当前受信进程/先前run，不接受用户上传、observer目录、任意下载checkpoint或未知插件提供的pickle。方法容器内同UID恶意写者不在此首单元威胁模型；不要把路径/哈希检查宣传为安全pickle沙箱。需要接入外部不可信state时停止，另定非pickle格式，不能顺手开放。

### 10.7 精确CPU验收（先机制、再真实loader，均非GPU）

使用普通子进程和显式fault gate，在每个点以SIGKILL结束持有loader的进程，然后新进程重开；mock loader只测磁盘顺序，真实torchdata/官方create_dataloader另列结果。fixture不能通过方法observer目录恢复。

| 故障/负例 | 必须断言 |
|---|---|
| genesis之后、第一次next之前 | 重新打开的第一批与无故障真实loader一致，occurrence从0开始。 |
| base.next之后、after_state序列化前 | 外部尚未收到batch；恢复从前一prefix，相同输入再次抽到且未跳过。 |
| 第一个blob写到一半；blob文件fsync后发布前 | 临时尾被忽略，prefix不增；不读取partial pickle；不存在可yield事务。 |
| 两blob已durable、draw临时JSON写到一半 | orphan blobs不当作draw；恢复此前cursor。 |
| 最终record发布后、draws目录fsync前 | 依据重启时完整record是否存在进行前一/后一prefix恢复；任一路径均不缺工作，不能硬编码只接受一种文件系统结果。 |
| record目录fsync之后、yield之前 | 恢复从原record重放，same occurrence/group/blob；不调用base.next补造这一批。 |
| batch已交付，但只读过第一个item | batch全部slot仍为pending，恢复包括尚未读出的item；不能只登记workflow已进入的slot。 |
| 两线程：next在序列化/fsync gate中阻塞，另一线程state_dict | snapshot在锁外等待，释放后只有完整前/后prefix，不出现advanced cursor与旧WAL混合；第二个并发迭代者拒绝。 |
| snapshot锚N，之后又durable两批到M再crash | 校验1..M完整覆盖；全部M批replay后，首个fresh batch等于真实loader从M之后的下一批，既不跳过也不重复该fresh occurrence。 |
| replay第一批后再次crash | 重启仍把所有未获commit权威的draw当pending；不因上次“已交付”删除义务。 |
| 同一prompt/row在不同slot或重复draw | group ID不同；同一record跨恢复的ID相同；task_id/PID变化不影响。 |
| snapshot与source/config/run/highwater/hash不符；缺中间record；最后record截断；blob改一字节/size错；symlink | 全部拒绝恢复，包含“只有最后一条坏了”也不静默截断。 |
| 重复发布相同blob/record与冲突发布 | 相同内容幂等；不同内容拒绝且不覆盖原文件；错误后当前iterator不能继续。 |
| owner锁被另一个进程持有、fork子进程写入 | 拒绝；不声称GPU backend fencing。 |
| 真实create_dataloader＋cycle_dataloader＋load_state_dict | num_workers0、shuffle固定、batch4、普通Dataset；记录实际版本/源码hash，逐batch核对连续运行和prefix恢复，len/batch_size相同，sampler.set_epoch0不覆盖恢复进度，epoch1明确拒绝。 |
| WorkflowExecutor.prepare_batch缓存接缝 | 最小CPU依赖fixture核对首次固定代理被缓存、后来换参数无效；不能用假的生成结果声明真实reward/tensor replay。若需要GPU才能构造完整executor，本项保留未覆盖，先测试原cycle和task-generator源码合同，不把缺项写PASS。 |

本单元交付前只要求上述机制与真实CPU loader通过，明确列出未覆盖的executor完整实例。未来P3b之前必须完成实际executor/训练入口合同；不能拿这次draw重投测试解锁GPU selective replay。首单元没有任何GPU、reward重算、state commit或训练收益验收。
