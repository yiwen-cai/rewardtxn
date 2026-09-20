# P3a-2 最小 RLVR CPU 产物读取／输出合同

2026-09-20；仅规划，不实现、不启动测试/GPU。依赖P3a-1已验收DrawLoader/BlobStore，不修改launcher、第三方、state schema、GPU ownership、observer或checkpoint。native r4的自动trainer恢复失败仍是独立事实，不因本计划改变。

## 1. 本单元只证明什么

在**同一仍有效的CPU owner/epoch/authorized attempt**内，让项目侧RLVR缓存包装经过真实官方RLVRWorkflow与GroupedRolloutWorkflow，证明三个分支：

| 已持久产物 | 底层生成调用 | 评分调用 | 返回路径 |
|---|---:|---:|---|
| response完整、reward缺失 | fake engine = 0 | 真实官方GSM8K scorer = 1（成功路径，不含显式原生重试） | 原RLVR `arun_episode` tensor builder构建并持久tensor |
| response/reward/tensor全套完整且一致 | fake engine = 0 | scorer = 0 | 读取并校验原dtype/shape/bytes，返回完全相同tensor值 |
| response不存在 | fake engine = 1 | 真实官方scorer = 1 | 原生成→立即评分→原tensor builder，依次持久 |

完整response＋reward但无tensor作为相邻必需分支：生成0、评分0，复用真实RLVR tensor builder补tensor，不从缓存手写另一套拼接公式。

fake engine显式标注为CPU确定性生成fixture，不是模型推理/GPU；评分必须调用未修改 `areal.reward.gsm8k.gsm8k_reward_fn`，不是mock score。输出是workflow tensor，不是进入optimizer、更不是已消费/已提交。所有durable draw依旧pending；不调用prepare_generation/commit_generation。

## 2. 源码接口与最小包装

已读 `areal/workflow/rlvr.py`：`arun_episode`使用get_input_ids_fn构建ModelRequest，随后 `_collect_samples` 返回 `(ModelResponse,reward)`，最后在原函数内创建七个torch.Tensor并unsqueeze(0)。`_collect_samples` 是 `await engine.agenerate(req)` 后立刻 `await self._compute_rewards(...)`；`_compute_rewards`将tokenizer.decode(output_tokens)和原task_data传给AsyncRewardWrapper。

`ModelResponse`（`api/io_struct.py:64`）的训练有效字段为input_tokens、output_tokens、output_logprobs、output_versions、stop_reason；tokenizer在重建时使用当前冻结的真实tokenizer实例。不要pickle整个ModelResponse/tokenizer：latency/ttft默认inf，processor/images/routed_experts等字段不在本单元；遇到这些非空任务类型明确拒绝。

拟新增一个 `scripts/ft/rlvr_replay.py`，只包含项目RLVR子类、七字段CPU tensor编解码和窄产物索引，不建插件框架：

1. `arun_episode`入口从DrawLoader的 `_r_draw` 与workflow_context.sample_idx得到logical sample，验证K范围和当前CPU授权。完整tensor命中时做全部引用/输入/版本检查后直接返回；未命中调用 **原 `super().arun_episode`**，返回后持久其原生tensor。不要复制原tensor builder源码。
2. `_collect_samples`仍调用原 `super()._collect_samples`，只给它一个该sample专用、只实现agenerate的窄代理：有完整response则验证并返回重建的真实ModelResponse；否则 `await` 底层fake engine真实方法、持久response完成后返回。父函数仍马上调用评分，无需额外组屏障。该代理不实现调度/重试/模型版本更新。
3. `_compute_rewards`：已验证reward存在则返回原值；缺失则 `await super()._compute_rewards` 执行真实官方评分链，成功返回后持久reward。不主动调用_recreate_executor、不mock ProcessPoolExecutor。真实AsyncRewardWrapper timeout可能返回0，不能仅凭0判断评分成功或正确；CPU验收用原评分调用完成证据和独立预期核对，失败/timeout不伪装有效cache hit。
4. 使用原 `GroupedRolloutWorkflow(project_workflow, group_size=K, ...)`。其sample_idx通过ContextVar分配，gather后按index排序并concat padded tensors。每sample状态从上下文/局部变量取，不能存在共享 `self.current_sample` 导致并发串槽。
5. 异步文件持久化通过有界I/O执行器await完成；不在event loop长时间fsync。没有全K生成完成屏障；正常每个miss仍生成→立即评分。测试中的gate仅控制fake engine完成次序，以证明sample0已评分时sample1仍在生成，不能迁移到生产调度。

冻结本地Qwen tokenizer资产/配置hash，离线加载真实tokenizer，CPU不加载model权重。测试fake engine按req.input_ids构造真实ModelResponse，输出token来自真实tokenizer对已知回答（例如既有CPU评分探针 `The answer is \\boxed{4}.` / answer=`4`）的编码；记录“synthetic generation”。提供长度匹配、有限的假logprobs和固定单一policy version V。不将这些值当实际模型logprob或state连续性证据。

## 3. 产物和授权合同

复用BlobStore原子发布/全hash读取，另有每 `(logical_sample,owner_epoch,attempt)` 的不可变response/reward/tensor索引；索引路径使用canonical tuple的完整digest，不把外部ID直接当路径。可复用项目现有不可变发布小函数，不新增数据库/通用WAL；phase引用必须在blob durable之后发布，缺失/截断最终索引fail closed，不把损坏等同cache miss。

共同绑定字段：schema、run/source/config、DrawLoader group/occurrence/K、sample_idx、attempt授权完整字典、prompt/input token hash、generation config hash、tokenizer hash、固定verifier定义hash。每阶段显式引用上游产物hash：

- response：原请求input tokens、output tokens/logprobs/versions/stop reason；input必须等于这次原tokenizer构建的req.input_ids，三个output数组等长，tokens合法整数、logprobs有限、stop_reason只接受明确支持的stop/length，abort/tool/multimodal拒绝。保留request rid作transport证据，但cache命中新生成的req.rid不能覆盖原origin rid。
- reward：原float值（有限）、verifier定义与参数hash、被评分response hash、实际decode文本hash和评分输入绑定。存在reward但response缺失/不匹配是损坏，不假装允许复用独立reward。
- tensor：七字段 `input_ids/loss_mask/logprobs/versions/turn_ids/attention_mask/rewards`。允许int32/float32/bool，记录字段名、dtype、shape、连续CPU原始bytes blob引用；完整tensor descriptor hash绑定所有raw bytes。每sample前六个序列字段shape=(1,L)，rewards=(1,)；prompt区域mask/versions/turn_ids与原builder一致。拒绝未知dtype/字段、shape不一致、无穷NaN、非CPU/非稠密数据。用明确白名单编解码，不使用torch.load或新的pickle格式。
- “完整tensor复用”必须连同有效response/reward链、当前请求输入和当前授权全部验证；不能只凭tensor文件存在返回。返回新tensor对象，调用者原地修改不能污染blob或后续cache结果。

attempt不能来自task_id或旧pilot临时sample_attempt。测试driver作为当前CPU state owner，在K个slot进入workflow前调用真实 `state.authorize_attempt(expected_attempt=...,new_attempt=...,expected_policy_version=V)`；授权表由该owner持有，workflow执行阶段前及await返回后重新检查当前epoch/attempt，避免异步执行期间被supersede。并发同一sample同attempt须串行或明确拒绝；重复完成后的同内容读取/accept幂等，冲突结果不可覆盖。

初次完整tensor持久后，可以在**现有CPU scope**调用真实 `state.accept_result`，传入response_sha256/reward_sha256/tensor_input_sha256、policy_version=V、verifier_version。这仅是结果授权receipt，不是训练消费/commit。Blob中的实际内容/版本检查由新适配承担；state.py当前只检查hash字段及scalar版本，不替适配验证tensor。

现有state没有公开read/validate_attempt API；本小单元若采用现有内部 `_locked(owner)` 读回control，则必须把这个明确的项目内耦合写入实现与测试，在该锁下比对control.attempts中的完整授权，禁止用调用者保存的旧dict自证当前有效；不假设存在新API。持久发布与最终accept之间发生CAS时，accept必须拒绝；阶段文件仅保留在原attempt不可变命名空间，不更新任何current指针。读取已有阶段时也必须重新校验授权，不能把验证留到tensor输出后。record中旧attempt即使内容相同，若已被CAS取代也不能继续输出。所有结果写入由当前owner完成；真实reward pool只计算，不持有state Owner，不把owner或打开的FD随task_data传进池进程。

## 4. epoch/adoption/version：本单元与未来schema界限

本单元的正例是有效owner持有期间重新创建workflow/从磁盘重读，或受控中断某个coroutine后在同epoch恢复；**不是owner被杀后的训练恢复**。另用新进程只读decode可检查跨进程序列化，但不因此授权其调用accept或宣布跨epoch replay。

- 现有CPU acquire_owner/authorize_attempt/accept_result可直接验证同epoch CAS、迟到拒绝、版本匹配和幂等；不用修改state.py。
- 旧owner退出后新epoch能获得新owner，不意味着旧response/tensor自动有效。旧阶段绑定旧epoch，首版一律拒绝adoption；允许新epoch新attempt从缺response路径重新生成fake结果，旧产物保持不可变。不要悄悄把旧record epoch改为当前epoch。
- **跨epoch复用**需要独立schema/授权修改：origin attempt/epoch、adoption attempt、旧artifact hash、实际token版本与新的admission version、允许复用/需重评分条件以及冲突规则。旧数据不能只靠“新authorize_attempt＋同一payload”洗成当前生成。本单元不实现。
- **真实异步混合token policy versions**超出现有scalar expected_policy_version。首版固定V，要求每个response输出token版本都为V；混合、缺失、未来或错误version均拒绝，不归一化/删去trace。支持真实混合轨迹前需定义request/admission version与逐token版本trace的授权schema及原staleness判定，不能把V填进去就宣布已验证。
- run内verifier固定；切换verifier、X1或在旧reward上改version标签必须先设计schema。当前只是检测不匹配并拒绝。
- `state._sets.pending` 仍仅允许regenerate；本单元不向generation里塞response/tensor引用，不放宽required components，不涉及GPU scope。上述缓存产物存在也不解锁持久消费恢复。

如果实现者发现仅为了同epoch读取就必须放宽state的授权检查，先停止给出实际冲突；不绕过state receipt、不用observer的success/authority覆盖检查。

## 5. 最小CPU验收与交付边界

最多新增一个项目模块 `rlvr_replay.py`、一个CPU fake engine/计数scorer fixture和一份测试文件/实施记录；已有DrawLoader/BlobStore只复用，不扩consumed接口。使用既有镜像Python3.12、真实AReaL/torch/tokenizer/GSM8K依赖，USER/LOGNAME/HOME/PYTHONPATH明确，readonly repo＋独立output、不暴露GPU、不安装依赖。测试subprocess/池数量有界，记录真实wrapper max_workers/max_retries；不通过重写第三方私有恢复函数控制池。若需要限制池，使用其公开构造参数并明确为CPU工程fixture配置，不宣称与正式默认运行资源相同。

精确验收项：

1. 无cache：fake engine 1次、真实scorer1次；输出与一个未启缓存的官方RLVRWorkflow在同固定fake response下的七字段dtype/shape/每字节一致，且response早于reward、reward早于tensor durable。
2. response已durable后受控中断、reward缺失：同owner新workflow生成0次、真实scorer1次，原builder输出等于上项。response不借用fixture的内存对象，必须从BlobStore重新读。
3. response/reward durable但tensor缺失：生成0评分0，原builder重建完全相同tensor；与仅完整tensorcache的返回路径区分统计。
4. 完整链：新workflow磁盘读取，生成0评分0；每字节相等、返回对象互不共享可变存储。篡改任一上游hash/bytes、tokenizer/prompt/verifier/gconfig不匹配均拒绝，不能转成“miss所以再算”掩盖损坏。
5. 缺response：走底层fake engine真实 `agenerate` 方法而非直接测试塞tensor；只允许确实未发布response的样本。partial临时文件可按BlobStore规则忽略，最终索引截断明确失败。
6. GroupedRolloutWorkflow：K=8，其中完整tensor、response-only、全缺三类混合；逐slot调用计数符合表格，输出排序、padding、reward/mask/versions匹配未缓存官方对照。不调用真实训练引擎，不把concat结果叫train_batch。
7. 并发顺序：让sample1的fake生成等待gate；sample0的真实评分必须能先返回，随后释放sample1。只证明没有全组两阶段屏障，不声称真实GPU自然F4可达。
8. 授权负例：旧attempt先完成、await期间CAS supersede、错误owner/epoch、不同内容重复、错误V/混合token版本、verifier错，均不能发布当前有效完整结果/accept。可能已经留下旧attempt自己的不可变stage作为审计，不允许其污染当前attempt索引。
9. owner正常关闭后新epoch：旧产物adoption明确拒绝；新attempt从fake生成重走完整路径可以通过。独立只读进程读取bytes不计入方法恢复成功。
10. 官方score正负样例：正确boxed4/answer4与错误回答，独立直接调用官方函数核对expected值；instrumented callable必须调用原函数一次，不能回传预写score。timeout/官方错误回退结果保留证据并使相应正确性验收失败，不拿原生0回退当缓存正确的证明。

结果按“synthetic generation / real official scoring / real RLVR tensor building / real grouped concatenation / CPU authorization receipt”分别标注。没有optimizer、完整checkpoint、commit、新进程训练load、GPU或正式selective replay的通过结论；不读observer，不以本单元替代独立oracle或修补原生launcher。
