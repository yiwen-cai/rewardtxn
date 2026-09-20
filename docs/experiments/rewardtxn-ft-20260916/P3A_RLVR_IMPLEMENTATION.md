# P3a-2 正分 RLVR CPU 产物合同（CPU 容器已验收）

历史版本：本文记录 rlvr-cpu-r1 的正分实现；后续 official-call-return 实现见 P3A_CALL_RETURN_IMPLEMENTATION.md。原三份源码已精确归档到 `p3_evidence/rlvr-cpu-r1/source/`，与原 verification.json 的 SHA256 一致；当前类不再保留无用正分并行实现。

2026-09-20。此交付只实现同一有效 CPU owner/epoch/attempt 内的 response/reward/tensor 持久读取和输出。**只能缓存精确 1.0 正分，不能接真实 RL 训练**：官方 AsyncRewardWrapper 的错误/超时回退和 GSM8K scorer 内部异常均可能返回 0，公开 float 接口没有可区分的完成证据；因此所有合法错误答案 0 也被拒绝。此策略会选择性丢弃负样本、改变训练分布，不是通用 reward 缓存完成方案，不能迁移至 A+R GPU。后续必须在方法内取得“评分正常完成／失败”证据并保留合法 0；本次不修改官方评分或重试实现。

## 文件与接口

- `scripts/ft/rlvr_replay.py`：`PositiveScoreRLVR(RLVRWorkflow)`；构造参数为 `owner, root, attempts, tokenizer_sha256, reward_fn, gconfig, tokenizer`（keyword-only），可显式指定 `timeout_seconds`；其余关键词交原 RLVR 构造。仅接受 callable scorer、真实 tokenizer 和七字段 CPU tensor，`close()` 在无 active coroutine 后回收唯一 I/O 线程。
- `tests/ft/rlvr_replay_fixture.py`：明确 synthetic 的 `FakeEngine.agenerate`，真实 tokenizer 编码已知答案，合成 logprob 和固定 policy 3；pickleable `counted_gsm8k` 实际调用未修改官方 GSM8K，额外记录测试调用完成证据。方法绝不读取计数日志。独立进程 fixture 只发布 response 后退出。
- `tests/ft/test_rlvr_replay.py`：8 项 opt-in 真实库 CPU 合同。默认跳过，不能把 skip 当真实依赖验证。

不改 state、DrawLoader、旧 pilot、P1、launcher、第三方或训练调度。没有 optimizer、checkpoint、commit、consumed、GPU scope 或 pending recovery 接口。

## 数据路径与权威边界

输入必须来自已信任的 DrawLoader JSON batch，并有 `_r_draw`；logical sample 为 `group_id:workflow_context.sample_idx`。当前 owner 在进入前显式 `state.authorize_attempt`；适配器持有 attempt 的副本，使用项目私有 `state._locked(owner)` 每阶段／每次 await 前后检查当前完整授权及 CPU scope，不以保存的旧 dict 自证授权。私有 API 耦合是已知限制，未扩 state schema。

共同 binding 包含 schema 1、run nonce/config hash/scope、draw identity、完整 attempt（epoch/nonce/scalar policy/verifier）、全部 task_data hash、真实 input token hash、tokenizer asset hash、实际单 sample generation config hash。调用方负责冻结 scorer 及其依赖与 verifier 定义；该 CPU API 不自动证明任意 callable 与 verifier hash 对应。测试使用真实源码 hash、显式 callable。tokenizer hash 在加载时冻结；运行期间 tokenizer 不得修改。

产物位于 `root/samples/SHA256([sample,epoch,owner_nonce,attempt])/` 的 response/reward/tensor 不可变索引；索引引用 `root/blobs/` 的完整 hash/size。沿用 BlobStore 的原子发布/fsync/校验，损坏最终索引或 blob 一律报错，不降级成 miss。仅支持纯文本 stop/length；mixed/缺失/错误 token policy version、multimodal/MoE 路由信息拒绝。七 tensor 保存明确 dtype/shape 与原始 CPU bytes，不新增 pickle。response 和 reward hash 串联到 tensor descriptor。

| 已有阶段 | fake 生成 | 官方评分 | tensor 输出 |
|---|---:|---:|---|
| 无 | 1 | 1 | 原 RLVR builder |
| response | 0 | 1 | 原 RLVR builder |
| response + reward | 0 | 0 | 原 RLVR builder |
| 完整三阶段 | 0 | 0 | 校验并加载原 dtype/shape/bytes 的新对象 |

miss 调用原 `RLVRWorkflow.arun_episode` 和 `_collect_samples`；只给原生成接口一个窄缓存代理，保留每个 sample 生成后立即评分，不引入 K 组生成屏障。原 GroupedRolloutWorkflow 完成并发、排序与 padding。长产物 I/O 交一个 worker 的 ThreadPoolExecutor；K 有界时排队有界，不建立通用任务系统。授权失效后不返回 tensor、不 accept；已 durable 的旧 attempt 阶段仅保留在旧命名空间。成功 `state.accept_result` 只是 CPU 结果 receipt，不是训练消费。

真实 AsyncRewardWrapper 使用公开 `max_workers=1,max_retries=0` 参数（测试与本窄 CPU 类均明确如此）；原 retry 方法未改且未调用私有重建。原 RLVR 构造可能创建无任务 executor，但实际评分仅投递单 worker 共享池。fixture 日志不是方法恢复数据。

## 验收清单与执行

主任务已用真实 AReaL CPU 容器通过 **8 项 / 26.199 秒**，无 GPU；资源为 **4 CPU、8 GiB、180 秒总 deadline**。worker 此前只做 `py_compile`。证据位于 [rlvr-cpu-r1/verification.json](p3_evidence/rlvr-cpu-r1/verification.json)，包含三份实现/fixture/test 的冻结 hash；原始输出、每项产物及调用日志同目录保留。[独立 cleanup](p3_evidence/rlvr-cpu-r1/independent-cleanup.json) 确认完整 CID 已不存在。通过范围严格限于下述正分 CPU 合同，未解除合法零分限制。

8 项覆盖：

1. 缺产物、response 后受控中断、reward 后受控中断、完整命中；新 workflow 从磁盘读取，对比原 RLVR builder 七字段逐字节，返回对象原地修改不污染缓存。
2. K=8 混合完整／response-only／无缓存，真实 Grouped 输出和官方未缓存对照排序/padding/bytes 一致；生成和实际 scorer 调用计数匹配。
3. CPU fake sample 1 gate 未开时 sample 0 的真实评分已返回，证明没有全组生成屏障。这不是自然 GPU F4。
4. 改 task_data/tokenizer hash/gconfig 与截断真实 response blob 均 fail closed。
5. 错误 epoch/verifier、混合 token versions、真实 await 中 CAS supersede 均拒绝，无有效 accept。
6. 独立真实子进程 owner 发布 response 后退出；父测试凭持久 PID 身份退出证明取得 epoch 1，旧 attempt adoption 拒绝，新 attempt 必须重新调用 fake engine + 真实 scorer，旧 response 保留。
7. 官方直接验证错误答案为合法 0；工作流拒绝该 0，无 reward/tensor/accept。
8. fixture 评分函数实际 sleep 0.5 秒后调用官方 scorer；公开 wrapper timeout=0.02 秒、retries=0 真实超时折 0，适配器拒绝。之后实际 scorer 完成得到 1.0 的测试日志也不能让方法补发布；再次验证无 reward/tensor/accept。没有 mock float 或 mock timeout。

使用既有 AReaL 镜像 `/opt/.venv/bin/python`（3.12）、现有 torch/transformers/math-verify/tokenizer；不新增依赖。主任务使用已验收 CPU namespace 容器隔离和完整 CID 清理，readonly `/workspace`、可写独立 `/output`、network none、cap-drop/no-new-privileges、无 GPU，本次实际使用 8 GiB/4 CPU 并通过，不主张 2 GiB/2 CPU 已验证。总 deadline 180 秒；新 owner fixture 单独 timeout 60 秒。所有任务完成后官方 executor 正常 atexit 收尾，总 watchdog 仍由容器负责。

容器内准确命令（cwd `/workspace`，`/output/rlvr-r1` 必须为本次新目录）：

```sh
env FT_RLVR_REPLAY_CPU=1 FT_RLVR_EVIDENCE_DIR=/output/rlvr-r1 \
  FT_RLVR_TOKENIZER=/workspace/models/Qwen2.5-0.5B-Instruct \
  PYTHONPATH=/workspace:/workspace/third_party/areal:/workspace/tests/ft \
  USER=cpu LOGNAME=cpu HOME=/tmp PATH=/opt/.venv/bin:/usr/bin:/bin \
  HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  /opt/.venv/bin/python -m unittest discover -s tests/ft -p test_rlvr_replay.py -v
```

每 test 在 evidence 根新建唯一目录，保留 state/draw/artifacts/真实评分日志/来源记录/owner 子进程日志；不覆盖失败。主任务另保存原始 stdout/stderr、源码与镜像 hash、完整 argv/env 白名单、退出码及 CID cleanup。该命令不自行向 shell 环境倾倒秘密。

## 明确未支持

合法零分的一般成功识别、任意 verifier／多模态、跨 epoch adoption、混合 token 版本、async staleness admission、独立 oracle 的 GPU tensor/update/load 证据、消费提交、实际生成与 selective replay、GPU 恢复，均未实现。DrawLoader 底层 loader 状态仍属于此前受信本 run pickle 边界；本产物格式只使用 JSON 与原始 bytes。测试中正常 workflow 重建与 CPU owner 新 epoch 再生成不构成训练恢复。native r4 的 launcher timeout 事实不变。
