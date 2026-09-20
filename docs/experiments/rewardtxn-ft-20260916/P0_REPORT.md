# P0 交付与验证报告

日期：2026-09-20（Asia/Shanghai）。**P0 接口审计交付完成；单 actor 同步 checkpoint 的新进程连续性验证通过。系统级恢复及故障矩阵不放行。**

P0 按修复方案的验收范围交付实际源码位置、接口决策、能力边界和运行时待证项，不以所有 cell 成功恢复为完成条件。本次没有实施完整 RewardTxn 适配、P1 runner 或正式 FT1/FT2。早期可行性检查只完成下述后端验证，完整 RecoverHandler 和 F4 运行验证仍开放。

## 交付

- [接口审计与决策](P0_INTERFACE_AUDIT.md)：恢复调用链、保存完成语义、六类故障的数据/PID 边界、源码及版本证据。
- [能力表](capability_matrix.csv)：A/A+R、C/C+R、B/B+R × 六类 cell，共36个唯一条目；区分静态接口、实际运行证据和正式矩阵范围。
- [独立 oracle 规范](oracle_spec.md)：最终保留状态链、完整状态、三维判定、RTO、证据隔离及屏障 release/abort/watchdog 合同。此为规范，尚未实现 oracle。
- [运行验收凭证](p0_evidence/verification.json)、[CPU 命令与环境](p0_evidence/execution_summary.json)、[源码来源](p0_evidence/source_provenance.json)。

## 已执行验证

| 检查 | 结果 | 证据边界 |
|---|---|---|
| 官方 test_recover、test_grouped_rollout_workflow、test_async_reward_wrapper | 62 passed，40.27秒，exit 0 | 包含 Mock、合成 workflow 和真实评分子进程；不能代替真实训练恢复 |
| RecoverInfo 真实文件读写 | 往返一致；损坏 JSON、缺失 cursor 文件被拒绝 | 只验证元数据，不证明多文件原子性 |
| ByteCheckpoint 实际 API | `framework="megatron"` 被 ValueError 拒绝 | 证明当前入口不直接支持；不证明无法开发转换适配 |
| Megatron GPU 保存端 | 2次更新→同步 DCP 保存→第3次更新；exit 0 | 单 rank、固定合成 token batch |
| Megatron GPU 恢复端 | 新容器进程加载→相同第3批更新；exit 0 | 保存状态、下一 RNG、下一步状态均精确一致 |

GPU 条件：Qwen2.5-0.5B-Instruct，494,032,768参数；1张 H100 PCIe，UUID `GPU-e3a46342-07b6-3c91-ce7f-833f0201a336`；`megatron:d1p1t1`、non-LoRA、batch2、sequence16、lr1e-6、linear scheduler、dropout关闭、`async_save=False`、`with_optim=True`、`use_checkpoint_opt_param_scheduler=True`。实际 engine seed为42，初始化日志证明其覆盖了探针前置seed；不是正式seed211实验。

保存状态及下一步状态均逐项比较 model、optimizer 持久 tensor、scheduler、RNG与LR；每个状态包含783个tensor/array全内容hash，Python/NumPy/Torch CPU/CUDA下一随机值一致。MCore明确不保存的 LocalNonpersistentObject 单列观察，不用它们声称checkpoint恢复成功。两个容器内部PID均为7是PID命名空间现象；它们由两个独立 `docker run --rm` 顺序启动，保存端先退出0，不是在同一进程中重置。

实际镜像内容 ID `sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`，RepoDigests为空，未编造registry digest。Torch 2.9.1+cu129、MCore 0.17.0、mbridge 0.15.1。GPU探针均隔离网络、只读挂载来源，单次timeout600秒；启动前重新检查至少4卡空闲及所选GPU空闲，仅申请1卡。收尾检查无本次容器残留，GPU0恢复4MiB占用。

成功保存/加载的精确 Docker argv、时间戳、设备快照和脚本hash见 [baseline命令](p0_evidence/gpu_probe_r3/baseline_command.json)、[resume命令](p0_evidence/gpu_probe_r3/resume_command.json)。结果见 [baseline](p0_evidence/gpu_probe_r3/baseline.json)、[resume](p0_evidence/gpu_probe_r3/resume.json)，两端原始日志和退出码在同目录。模型内容见 [model_manifest](p0_evidence/model_manifest.json)。

## 失败尝试完整保留

1. 首次CPU测试因容器UID没有passwd记录，collection失败；设置USER/LOGNAME后通过。不是被测方法失败。
2. GPU r1训练两步成功，但探针未创建checkpoint目录，保存调用报FileNotFoundError；补一行建目录后改用新r2目录。
3. GPU r2已写出真实checkpoint，摘要函数不认识LocalNonpersistentObject而退出；依据实际MCore定义分开非持久对象后，改用新r3目录。未修改第三方方法实现或放宽状态比较。

r1/r2日志、退出码及对应脚本快照保留；r2/r3各保留一份约6.918GB checkpoint，总约13.8GB，不覆盖失败产物。r3 baseline/resume才构成通过的验证对，不拼接失败版本结果。不从这些包含初始化、hash与调试的时间估计方法RTO或正式GPU预算。

## P0 决策及后续边界

1. **A优先使用v1＋Megatron。** 同步单rank后端已验证，完整RecoverHandler、数据游标/pending工作、四卡actor/rollout布局及故障恢复仍须集成验证。不得将本次engine PASS复制为F2或A+R PASS。
2. **异步保存单独验证。** 静态链显示RecoverInfo可能早于异步finalize；正式前必须验证完成语义。同步探针不构成正式禁用异步能力的决定，也不偷偷给基线补R提交协议。
3. **A选定内存dispatcher路径的F3预先N/A。** 没有持久未ACK集合，不能造接口；仅限审计路径。C的应用集成尚未定义，因此C/F3标条件阻塞，不提前判所有实现N/A。
4. **F4及X1/X2不虚报适用。** F4尚未实测全K生成/评分K/2切点；先自然事件观察，若需两阶段工作流则先修订设计。X1/X2回调通知不携reward值，需继续映射真实结果路径，不直接改内部字典。原F4主检验与样本规则未变。
5. **C为条件分支。** 先解决Slime Megatron optimizer到ByteCheckpoint的真实状态适配；不冒充原生支持。B官方artifact仍未确认，检索范围与局限见 [搜索记录](p0_evidence/robustrl_search.json)，不能据此断言不存在公开代码。

可以推进P1的隔离runner、屏障退出及CPU合同；P2/P3的后端相关实现须兑现上述接口与集成验证。没有启动正式故障矩阵，不重开准确率、正常吞吐或5%开销复测。
