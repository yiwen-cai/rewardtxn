# P3a-3 official-call-return CPU envelope（待真实容器验收）

2026-09-20。本单元接受官方 scorer **返回的**合法 0/1，包括官方内部异常/超时回退为 0；envelope 仅证明 `official_call_returned`，**不证明 verification_success**。不裁决 GPU/正式实验对内部 fallback 的语义，不接训练、消费提交、跨 epoch adoption 或混合 token 版本。

## 历史与文件

修改前将 rlvr-cpu-r1 的 `rlvr_replay.py`、fixture、test 原始 bytes 归档到 `p3_evidence/rlvr-cpu-r1/source/`（保留路径），逐份核对原 verification.json 的完整 SHA256；`source/manifest.json` 保存匹配表。历史 8 项 / 26.199 秒仍只适用于该正分版本。

当前新增 `scripts/ft/reward_return.py`，窄改 `rlvr_replay.py` 与原 fixture/test。类改名 `CallReturnRLVR`，不保留旧 PositiveScoreRLVR 别名或并行策略。`P3A_RLVR_IMPLEMENTATION.md` 留作历史，零分诊断文档纠正了范围限制归属：此前“不改第三方”是主任务 bounded 委派边界，不是用户全局禁令，不产生新审批流程。

state、DrawLoader、P1、launcher、第三方及 oracle 均未改。官方 AsyncRewardWrapper、GSM8K 和 RLVR builder/Grouped 保持真实调用。

## 精确合同

`ReturningReward(scorer)` 是项目级 pickleable callable，实际调用 scorer 一次，不捕获异常、不自行重试。它把官方正常返回的有限 0/1 放入冻结 dataclass `RewardReturn(schema=1, invocation_nonce, input_sha256, verifier_sha256, status='official_call_returned', score)`。scorer 外逸异常照常由官方 wrapper 的原循环处理；本 CPU 类仍显式采用公共 max_workers=1/max_retries=0/默认 timeout=15s，未更改私有 retry 实现。

reserved `_r_reward_request` 随方法调用提交，wrapper callable 在调用官方 scorer 前移除它；原始任务若已有该字段则拒绝。input hash 绑定原 prompt/completion 字符串、两组 token ids、完整原始 task_data。callee 从实际输入重算，caller 从冻结 response/task_data 独立重算，verifier 与当前授权一致。

`invocation_nonce` 是 `SHA256({schema:2, 完整 authorized attempt, input_sha256})`：表示同一授权和冻结输入的**逻辑评分调用身份**，不表示随机每次调度或 wrapper retry 序号。发布前已可推导，新 workflow 从冻结输入和当前授权重算同一预期值；读取缓存不以 record 自报的 nonce 自证。错 nonce 的实时返回和改 hash 后仍自洽的持久 envelope 都拒绝。不新建 call WAL。

项目 `_compute_rewards` 继续 await 原 RLVR `_compute_rewards`；后者实际返回 Python 对象，不做 float cast。项目在对象到达原统计器/builder 前校验并解包 float。await 后先重验 owner/CAS 授权。裸 wrapper 0 或任何非合格 envelope 抛 `UncertainReward`，保留已 durable response，不发布 reward/tensor/accept；不擅自补重试、跳过 sample 或换 prompt。timeout 之后池任务即使最终返回 1，也不会异步补发布。

artifact 共同 binding 升为 **schema 2**；reward payload 为 `{response_sha256, return:完整 envelope}`，tensor 校验从该 envelope 得到 0/1。旧 schema 1 及旧标量 reward 不能直接升级；需新 evidence/output root，或遇到旧产物时 fail closed。合法 0 与 1 使用同一持久、缓存、tensor、accept 路径；Grouped 不筛选零分，原 masks/padding/order 保持。

## 验证与命令

worker 已完成四个 Python 文件的 `py_compile`。**未运行真实库新版套件**，交主任务一次完整 CPU 容器验证。沿用已验收资源：4 CPU / 8 GiB / 180 秒 deadline，无 GPU，readonly repo、独立 output、无网络及既有完整 CID cleanup。不要复用 rlvr-cpu-r1 evidence。

```sh
env FT_RLVR_REPLAY_CPU=1 FT_RLVR_EVIDENCE_DIR=/output/call-return-r1 \
  FT_RLVR_TOKENIZER=/workspace/models/Qwen2.5-0.5B-Instruct \
  PYTHONPATH=/workspace:/workspace/third_party/areal:/workspace/tests/ft \
  USER=cpu LOGNAME=cpu HOME=/tmp PATH=/opt/.venv/bin:/usr/bin:/bin \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /opt/.venv/bin/python -m unittest discover -s tests/ft -p test_rlvr_replay.py -v
```

共 12 项，复用原四阶段/native builder、磁盘新 workflow、真实新进程 owner、错误 epoch/verifier/mixed versions、CAS、损坏等合同，另验证：

- 合法错误答案由原 GSM8K 返回 0，发布完整链且再次读取不生成/不评分。
- 原 Grouped K=8 的真实评分 1/0 交替；与原未缓存 RLVR+Grouped 的七 tensor 字节相同，完整保留八个 sample。
- 实际官方池返回 envelope 的 pickle 往返；实时 nonce/input/verifier 错配拒绝。
- 已持久 envelope nonce 被改且重新计算 blob hash 仍拒绝；旧 schema 1 不升级。
- 真实评分 await 期间 CAS 失效，返回 envelope 也不发布 reward/tensor/accept。
- **测试注入的内层异常**：只在 fixture 中让真实 MathVerifyWorker `_verify_impl` 抛 RuntimeError，原 verify 捕获并原 GSM8K 返回 0；envelope 为 call_returned 0，正常缓存。它明确展示无法证明内部成功，不是自然异常或真实 timeout 测试。
- **真实外层 timeout**：fixture sleep 0.5 秒后实际官方评分，官方 wrapper timeout=0.02 秒折裸 0；适配器拒绝，最终 scorer 日志为 1 也无发布。这与注入内层异常分开记录，不 mock timeout/float。

fixture 评分日志与 source/provenance 只供测试诊断，方法不读取；生成始终明确 synthetic CPU。真实依赖库和分数完成事件不等于 GPU selective replay 或正式 oracle 验收。如何处理内层 fallback、oracle 自身失败与 pending 重评分仍按零分诊断中的未决设计边界保留。

## r2 真实验收失败与最小测试修复

主任务真实 CPU r2：12 项 / 28.685 秒，11 项成功、1 项 ERROR；原始 `p3_evidence/rlvr-cpu-r2/tests.log` 保留，不将此轮计为完整通过。失败项 `test_cached_nonce_and_old_schema_cannot_self_authorize` 改了 reward envelope nonce 并重新发布 reward blob，却未更新 tensor descriptor 的 reward_sha256。生产 `_load_tensors` 因此正确地先报 `tensor upstream binding mismatch`，该轮并未到达缓存 nonce 校验，后面的旧 schema 断言也未执行。

修改前归档 r2 四份源码到 `p3_evidence/rlvr-cpu-r2/source/` 并记录完整 SHA256。两个生产模块和 fixture 的 hash 与该 run provenance 一致；旧测试文件自身未被该 provenance 记录，manifest 如实标 null，归档的是修复前未变的工作区原文件及其 hash，不谎称另有独立测试源 hash 证明。

仅修测试，不改生产校验顺序：保留第一次完整链引用不符的精确错误断言；随后将 tensor descriptor 的 reward_sha256 改为被篡改 reward 的 hash，再重新发布 tensor blob/index，使整条引用链自洽。此时仍必须抛出 `UncertainReward`；用 wraps 原函数的 spy 确认 `_validate_return` 实际调用一次，并明确比较收到的错误 nonce 与从冻结输入/attempt 推导的预期 nonce不同，且无生成/评分调用。之后恢复原引用再执行旧 schema 拒绝断言。这不是放宽异常类型，也不通过删除 tensor 避开完整缓存路径。

worker 仅 `py_compile` 通过；新版真实运行待主任务。建议先在同隔离 CPU 环境、全新 evidence 根定向执行：

```sh
# 其余 env/resource 设置同上，PYTHONPATH 必须包含 /workspace/tests/ft。
/opt/.venv/bin/python -m unittest test_rlvr_replay.RLVRContracts.test_cached_nonce_and_old_schema_cannot_self_authorize -v
```

定向成功后是否重跑完整 12 项由主任务按验收需要决定；本修复没有修改正式评分语义或第三方。

## 主任务真实库验收补记

r2运行12项，11通过/1 ERROR（28.685秒），失败由测试先触发tensor上游hash绑定检查导致，未证明原nonce负例。修正仅改测试，保留明确hash拒绝断言，再构造引用自洽但nonce错误的完整链，并用wraps真实校验器确认调用一次及预期nonce不匹配；没有放宽异常类型掩盖问题。r3定向复验1项通过（11.655秒），包括旧schema负例。生产源码与fixture未改，无需重复已通过的11项；不声称同一次12项全绿。两轮原始日志、源码归档及独立容器移除证据分别保留在p3_evidence/rlvr-cpu-r2与rlvr-cpu-r3。正式评分口径问题仍待用户答复，当前仅CPU方法内返回合同。
