# P3 CPU 训练输入身份桥（待真实库验收）

2026-09-20。新增仅 `scripts/ft/batch_identity.py`、`tests/ft/test_batch_identity.py` 和本文。未修改第三方、state、DrawLoader、CallReturnRLVR、既有证据、正式评分语义；未启动 GPU。

## 实现边界

`IdentityBridge(workflow, root, policy_version=...)` 包装现有 CallReturnRLVR：原七字段结果与当前 `state.accepted` receipt 及方法 BlobStore 的 response/reward/tensor 全 hash 链逐一核对，复用原 tensor decoder/版本/评分 envelope 检查后发布方法自有身份记录。记录包含 receipt、原 task_data、三个 artifact 引用。四个 int64 token 同形字段 `_r_receipt_0..3` 无损携带该记录完整 SHA256；每行有效 token 恒定，padding 必须为 0。记录通过完整 receipt 绑定真实 attempt，不能用 observer 或 dispatch 顺序找回。

`authorize(data)` 在单 loader producer 给一个真实 draw 的 K slots 授权，已有同 epoch 的当前授权可用于重读缓存；错误 epoch/version 拒绝。桥只接受 CPU tensor，state scope 仍 cpu_contract，不是 GPU fencing。

`begin_update(trajectories, config)` 在**全 update 边界**验证完整 K、各 slot 唯一、receipt/当前授权/原 tokens 与 versions，记录实际输入顺序和变换后逐行 tensor 指纹。logical ID 来自排序后 logical sample 集合与冻结 config，与行重排、物理 retry/attempt 无关；physical invocation 使用独立 UUID。它不标记 consumed。

`at_train_batch(data)` 核验 microbatch 为该 update 的合法子集，不要求 microbatch 内完整 K；n_mbs=1 时额外要求完整 update 集合。按身份核对变换后 tensor 指纹（只排除原 PPO 在入口前删除的 rewards/tot_rewards/kl_rewards），避免重排或内容变化被接受。发布前在当前 owner 锁内再验整个 update 授权，持久 `prepared_not_applied` intent 后显式抛 `PreparedBoundary`。没有 optimizer stats、scheduler 成功、checkpoint、commit 或消费推进。

`strip_identity` 生成独立 dict/list，仅删除四个 reserved 字段，不原地修改 carrier/tensor。CPU 终端 forward/train_batch 用该 helper 验证不向模拟模型输入传字段；forward 返回明确 synthetic zeros。**没有调用真实模型前向，也没有证明 GPU engine adapter 已安装**。测试要求真实 PPO 数据处理和 train_batch 调用发生，终端停止不冒充真实 optimizer。

使用现有项目私有 `_locked` 与 RLVR decoder 是明确窄耦合。I/O 包装仅一个线程，workflow await 后只进行小型当前 owner/CAS 文件读，完整 blob/tensor 校验在专用线程内；CPU 校验会反复读取 full blobs，成本尚未优化/量测，不宣称低开销。

## 一个主链集成测试

`FT_BATCH_IDENTITY_CPU=1` 才启用真实依赖测试。使用真实 tokenizer、官方 GSM8K/AsyncRewardWrapper、原 Grouped、真实 `WorkflowExecutor.initialize/prepare_batch/destroy` 后台线程；fake engine 只负责 synthetic tokens/logprobs/version 和测试完成次序。先用独立 DrawLoader 实例发布缓存，再以新实例仅在 executor producer 迭代，遵守既有单线程 loader 合同。

两次原异步 prepare_batch 获取八组，每组 K=8；第二次故意传无效替换 loader/workflow，确认官方首次缓存 generator 继续生效。原 should_accept_fn 拒绝一个组，拒绝的八个样本仍是未提交义务，不作 consumed。预取/完成顺序不作为身份。随后真正调用 PPOActor.compute_logp/compute_advantages/ppo_update（公开 PPOActorConfig，synthetic forward，无 recompute_logprob），由原 batched_call 和原 microbatch splitter 到达终端。

同测试检查：

- 全 update 32 个 sample、reorder 下 logical ID 稳定，physical ID 改变；真实 train_batch 入口写未应用 intent。
- 原 n_mbs=2 splitter 输出的身份集合并回后完整，允许 microbatch 跨组；该项只验证工具，不改变正式 n_mbs=1 范围，不伪造第二次优化。
- 半组/重复样本、缺载体、换身份不换 tokens、非法 padding 身份、优势值被改、CAS supersede 均拒绝。
- 方法 state head 仍为空；输出结果明确 optimizer_executed=false、consumed_authority=false。

尚未做 Gloo/DistRolloutCoordinator/实际 DP-CP broadcast；未覆盖 GPU前向 kwargs、真实优化器/保存加载、consumed overlay、跨 epoch adoption。同一主链已加入相同 token、不同 occurrence 的身份区分断言，以及原 splitter 的一维 ID 被整份复制反例。普通 list sidecar 反拆错误目前只有源码诊断，没有运行反例；同 tokens 但伪换当前有效 receipt 的逐行交换并未单列用例（完整集合/当前授权有独立检查）。所有上述测试均待主任务运行，不能把覆盖设计写成已通过。

## 验证命令

worker 仅做两份文件 `py_compile` 成功，**尚未声称真实主链通过**。真实库如在 CPU 不支持某个 PPO 接口，保留失败位置，不 mock 掉原优势计算或 split。

主任务沿用既有无 GPU CPU 容器，4 CPU、8 GiB、180 秒总 deadline，readonly `/workspace` 与独立 `/output`，保留完整 CID/cleanup、原始 stdout/stderr 和源码 hash。准确容器内命令：

```sh
env FT_BATCH_IDENTITY_CPU=1 FT_BATCH_IDENTITY_EVIDENCE_DIR=/output/batch-bridge-r1 \
  FT_RLVR_TOKENIZER=/workspace/models/Qwen2.5-0.5B-Instruct \
  PYTHONPATH=/workspace:/workspace/third_party/areal:/workspace/tests/ft \
  USER=cpu LOGNAME=cpu HOME=/tmp PATH=/opt/.venv/bin:/usr/bin:/bin \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /opt/.venv/bin/python -m unittest discover -s tests/ft -p test_batch_identity.py -v
```

每次新建 evidence 子目录；方法产物、intent、评分日志保留。该桥直接针对 replay→训练输入接缝，但只有后续真实 GPU optimizer/scheduler→同切点 async finalize→完整 commit→新进程 load 验证才能创建 consumed authority。正式评分 fallback 语义仍未替用户决定。

`tensor_hashes` 使用 CPU tensor 的 NumPy bytes，仅覆盖本单元原 RLVR/PPO dtype；没有 bfloat16 或 GPU dtype 通用兼容声明，不在此提前泛化。

## 主任务验收结果

batch-bridge-r1真实库CPU集成测试1项通过（30.849秒），完整日志与方法产物保留在p3_evidence/batch-bridge-r1。两次原prepare_batch共交付8组/64样本；一个被过滤组的8样本仍为pending；实际PPO入口发布32样本prepared_not_applied intent；原n_mbs=2拆分为16+16。生成和前向为明确合成fixture，未执行optimizer、consumed或GPU。容器full ID独立确认已移除，verification.json记录版本及清理。
