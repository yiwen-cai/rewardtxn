# 合法零分的最小接入决策（只读诊断）

2026-09-20。仅更新已有验收文档并提出下一单元设计；未修改实现、第三方、oracle 或 state，未运行 GPU。P3a-2 已通过真实 CPU 8 项 / 26.199 秒，但正分限制仍在，当前实现不能训练。

## 1. 已核实的三层返回行为

依据当前 checkout：

| 层 | 实际行为 | 外层能推出什么 |
|---|---|---|
| `areal/api/reward_api.py:AsyncRewardWrapper.__call__` | 直接 `return await asyncio.wait_for(future, ...)`，没有 float cast；外层 TimeoutError 最后一次返回裸整数 0；BrokenProcessPool 和一般异常按原循环重试，最终抛出；循环末还有裸 0 | 可以通过类型化 envelope 区分“callable 返回对象”和“wrapper 未交付 callable 结果”。不能从裸 0 推出具体哪个内部评分阶段成功 |
| `areal/reward/gsm8k.py:gsm8k_reward_fn` | 调 `get_math_verify_worker().verify(str(completions), str(answer))`；本函数捕获 Exception 返回 0.0 | envelope 只能证明这个函数正常 return；正常 return 包含它自己的 fallback |
| `areal/reward/__init__.py:MathVerifyWorker.verify` | `_verify_impl` 正常算出真/假或无法解析都可返回 0/1；`future.result(timeout=5)` 的 TimeoutError 和其他异常也折 0 | 同样的 float 0 无法区分答案不等、不可解析、内部超时、内部异常 |

`ThreadPoolExecutor` 的 with 退出会等待工作线程：5 秒不是强制执行截止。`_verify_impl` 甚至可能在超时判定后完成为 1，`verify` 仍最终返回 0。因此事后看到线程完成、日志中出现 1、或重新评分为 1，都不能把本次返回 0 改为“当时评分成功为 1”。本诊断未审计所安装 math-verify 的全部 parser/grader 内部 fallback；即使未来增加 AReaL 层状态，也不能据此宣称所有依赖内层异常都已识别。

## 2. 可立即审查的最小 envelope 接口

若下一 CPU 单元选择 **official-call-return 语义**，只新增一个项目侧 pickleable callable 和一个固定结构，不修改官方函数或官方 wrapper 的重试：

```text
RewardReturn(schema=1, invocation_nonce, input_sha256, verifier_sha256,
             status="official_call_returned", score=0.0|1.0)
```

调用方在当前授权下创建 invocation nonce，冻结 prompt/completion/token ids/label 的 canonical 输入 hash 和 verifier 定义。pool callable **恰调用一次原 gsm8k_reward_fn**；它正常 return 后验证 finite 且在 {0,1}，然后构造 envelope。callable 外逸异常继续抛出，由原 wrapper 处理；不要 catch 后制造 envelope，也不要在 wrapper 外新增重试。原 wrapper 重试可能多次执行 callable，同一 invocation nonce 不代表某个具体 retry 序号；未经额外接口不得宣称掌握该序号。

项目 `_compute_rewards` 可保留 `await super()._compute_rewards(...)`：已读 RLVR 原函数只是 decode、await、return，直到原 `_collect_samples` 才将 reward 交统计器。项目在这个边界校验并解包 envelope，使原 `_collect_samples` / tensor builder 仍看到 float。这样实际 0 和 1 都走同一缓存与 tensor 路径，不按标签正负筛选、不改生成→立即评分顺序。

收到裸 0、错误类型/nonce/input/verifier 或异常时不发布 reward/tensor/accept。外层异常仍保持原 wrapper retry/raise 行为；原 wrapper timeout 裸 0 显式记 `wrapper_result_unavailable`，不能仅从裸 0 推断唯一原因。该 CPU 小单元可停下当前 episode 并保留 response，不能偷偷跳过样本或换 prompt；训练中的 pending 重评分/安全停止策略尚未设计。

方法内 envelope 由实际评分 callable 返回，经方法自身 await 通路接收；不是 observer 证据、不是 controller 恢复数据。await 后仍检查现有 CPU owner/epoch/attempt。reward artifact 应绑定完整 envelope/输入/response；适配器不把它称 authoritative reward。采用新 artifact schema 或新 root，拒绝旧正分产物被无证据升级成 envelope 产物；不必修改 state 的 CPU receipt schema，也不解除 adoption/mixed-version 限制。

**该接口能保留合法零分，但其状态名称只能是 `official_call_returned`，不能命名 `verification_succeeded`。** 内层 fallback 0 也会被缓存，这是沿用官方 scorer 返回语义的明确结果。不能一边声称保持所有原函数语义，一边暗中把内层 fallback 0 全部改成失败。

## 3. 当前独立 oracle 的能力与缺口

已读 `scripts/ft/oracle.py:_reward`：`areal-gsm8k` allowlist 冻结 GSM8K / reward `__init__.py` 源码 hash、math-verify 版本、extraction/precision/5 秒参数，随后直接调用同一个官方 `gsm8k_reward_fn` 并转 float。`audit_run` 对方法 reward 与这个独立重新计算值进行比较。`oracle_spec.md` 要求 reward 与独立 authority 一致，但尚未明确宣布“内部运行错误回退也算正常数学评分”。

因此现有代码的**操作行为**允许两边都返回 fallback 0 时值比较通过，不能将此解读成规范已经认可内层 timeout/error。方法为 0、独立重算为 1 时，当前实现记录 `independent reward mismatch`，不是自动洗成 PASS；方法缓存一次偶发 timeout 0 可能让这种差异持续存在。即使方法和 oracle 都是 0，也只能证明本次标量比较相等，不能证明两个内部调用都无故障。

选择 official-call-return 语义时可以继续用现有独立重算做值一致性检查，且必须保留所有 mismatch、不按收益重跑挑选；但若目标是证明“正确数学 reward／评分内层成功”，当前 oracle 信息不够。oracle 自身超时/异常也可能造成与方法的差异；在更细状态未建立前不能仅靠 float 说明哪一边失败。对 GPU/正式结果应先冻结分类合同，不能事后改变已观察失败的含义。

## 4. 若要求识别内部 fallback，额外信息不可省略

仅 envelope 不够；外层计时、捕获日志、重算一次、仅观察 `_verify_impl` 返回都不能可靠识别 `future.result` 是否已经超时。日志是诊断，不是方法内完成权威；也不建议全局 monkeypatch Future/ThreadPoolExecutor 来推断隐藏分支。

在“严格保持未修改官方 GSM8K 调用”这一边界内，当前公开接口没有所需信息，不能实现可信内部状态。两个可审查的后续路径都属于**额外设计裁决**，本次不实施：

1. **窄上游兼容状态接口**：为 MathVerifyWorker 增加内部/并行的状态返回入口，在现有 `try/with/future.result/except` 的同一分支记录 `returned / timeout_fallback / exception_fallback`；原 `verify` 仍只返回该结果的 score。GSM8K 相应提供状态入口，标记 get-worker/string conversion/调用本身异常 fallback，原 `gsm8k_reward_fn` 仍返回相同 scalar。保持原线程等待、超时、默认参数和外层 retry；不能为方便状态报告改变控制流。既有 scalar API 保留，源码/hash 变化明确披露；状态缺口仍含第三方 parser/grader 内部吞错。需两臂共同冻结 verifier 适配、验证正常/异常/真实超时的输出与旧版本一致，不能只给 R 更可靠评分器后称原 A 对照。
2. **项目侧新 verifier adapter**：显式重建同一 parse/verify 调用与 timeout 分支并返回状态。无需编辑第三方，但它不是“仍调用原 gsm8k_reward_fn 且完全未改评分器”的方案；即便通常数值一致，也要新 verifier ID、依赖/参数/源码冻结、两臂共同披露和独立重算适配。不可复制后默认认为语义等价，不建议作为下一个最小单元直接实施。

即使拿到 internal fallback 状态，也不能把 fallback 样本直接丢掉后继续训练。必须预先定义保留该 response 的重评分/安全停止策略、预算及两臂行为；正常合法 0 应保留。此策略和状态接缝超出当前 CPU 产物合同，不能偷偷补强 baseline。

第三方 `AGENTS.md` 的 Ask-first 条目是 config structures、新依赖、launcher/scheduler、删除或重命名公开 API；上述保留旧 API 的状态接口不自动等同删除/重命名。但此前主任务对该 bounded 委派限定只读且不修改第三方；这是本单元的任务范围，不是用户全局禁止，也不制造额外用户审批。若后续主任务选择上游接口方案，应明确扩大实施范围，不能在当前只读诊断内自动实施。项目 envelope 路径不涉及这些条目。

## 5. 建议下一单元与待裁决项

建议先做**CPU official-call-return envelope 合同**，明确只证明合法 0/1 均可持久和复用、外层 timeout 不伪造完成；不接 GPU/训练，不声称消除了内层 fallback。最小新增一个项目 callable/envelope 模块，窄改新 RLVR 类的 reward 编解码和类型校验，增加 CPU 正负例；不构建通用状态框架，不改 state/launcher/retry。

最小验收：官方正确答案 1 与错误答案 0 均持久/完整 tensor 复用；真实外层 timeout 裸 0 不发布；pickle 往返、wrong nonce/input/verifier 拒绝；错配/截断 artifact 拒绝；旧 owner/CAS 回归；用明确标注的测试注入内层异常验证 envelope **仍为 call_returned 0**（这是暴露限制，不是评分成功证明）；官方 Grouped 含 0/1 的原始 reward/mask/tensor 不被过滤。真实超时与测试注入异常分别记账，不能 mock fallback 后声称真实超时。

正式使用前必须明确的设计裁决只有两项：

- verifier 合同究竟是“原 GSM8K API 返回值，包括内层 fallback”还是“已识别无内部 fallback 的评分结果”？前者可用最小 envelope，但冻结并披露运行相关零分与 oracle 重算不一致风险；后者必须先取得额外状态接口。
- 对已识别的外层未交付／将来内层 fallback，方法如何保留工作并停下/重评分，以及 oracle 如何区分方法错误与 authority 自身失败？不能用丢负样本、observer 反哺或事后改分解决。

本诊断不代替这两项实验语义裁决，也不因 CPU 8 项通过而推荐直接跑 GPU。
