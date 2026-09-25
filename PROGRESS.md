# RewardTxn 项目进展报告

## 当前状态：2026-09-25

**最小 FT 正式矩阵（AReaL 分支，冻结 2026-09-24 13:14）13/13 对已完成并通过验收。** 冻结见[FORMAL_FREEZE_20260924.md](docs/experiments/rewardtxn-ft-20260916/FORMAL_FREEZE_20260924.md)，逐对记录在 `docs/experiments/rewardtxn-ft-20260916/minimal_evidence/formal-*-pair.json`。

| 场景 | 对数 | A（AReaL 原生恢复） | A+R | 备注 |
|---|---:|---|---|---|
| F2 | 10 | 10/10 `safe_discard`，目标 0/32 复用 | 10/10 `correct_recovered`，目标 32/32 复用 | 无 `invalid_commit`/`safe_stop`/timeout |
| 无故障 | 3 | 3/3 `no_fault_verified` | 3/3 `no_fault_verified` | 仅描述正常路径成本 |

- 按冻结判定规则（10 对均安全可判定）：不一致对 10:0，双侧精确 McNemar p≈0.002 < 0.05；精确 95% 区间 A 0/10 为 [0, 0.31]，R 10/10 为 [0.69, 1]。以上为我据逐对记录复算，正式统计报告尚未单独产出。
- F2 故障后生成 token：A 约 12.1–12.8 万，R 约 9.8–10.5 万。单 run 墙钟中位（cost.json，训练至验收）：F2 A≈785 s、R≈1094 s；无故障 A≈530 s、R≈847 s。R 正常路径成本明显更高，不得宣称 R 整体更快；按冻结规则不以无故障耗时直接相减推断收益。
- 第 2 对原尝试 R 在故障信号前权重同步 CUDA OOM → `technical_invalid`，按冻结规则做唯一一次同 seed/同顺序完整重做（`-redo1`）通过，见[重做记录](docs/experiments/rewardtxn-ft-20260916/FORMAL_P02_TECHNICAL_INVALID_REDO_20260924.md)；原件保留。GPU 分配改为任意四张同时空闲 H100，见[修订](docs/experiments/rewardtxn-ft-20260916/FORMAL_GPU_ASSIGNMENT_AMENDMENT_20260924.md)。
- 存储：第 6 对（s938）两臂全量保留，其余通过 run 已删大分片。峰值增量 A≈7 GB、R F2≈34.7 GB / 无故障≈21 GB。
- 未完成：正式结果报告与统计脚本化；F1/F4 及 C/B 分支仍推迟；本日改动（大量 tracked 修改与新文件）尚未提交。

## 2026-09-20 状态（历史）

**P0接口审计交付完成。** 官方CPU测试62项通过；单H100同步Megatron保存→新进程加载→下一步状态连续性通过。见 [P0报告](docs/experiments/rewardtxn-ft-20260916/P0_REPORT.md)。完整R适配、系统级故障恢复、F4实际切点和正式冻结仍未完成；本次仅运行P0验证探针，未启动正式矩阵。后续可推进P1隔离runner/CPU合同。

### 9月16日主线决定（继续有效；当时进度由上文更新）

**研究主线已切换为已发表方法的容错优势比较。** 已交付[RewardTxn 已发表方法容错优势补充实验方案](RewardTxn%20已发表方法容错优势补充实验方案.md)（FT-v1），本轮只完成规划与文档同步，新适配器、故障注入矩阵及GPU实验尚未实现/启动。

**用户验收决定：准确率验证目前视为通过，性能无明显回退，短期内不规划重做。** clean R/O增seed、LITE重跑、24单元统一版本确认、O/R×异步/同步、正常路径吞吐与5%公共开销复测均不在近期计划，也不作为新容错实验前置。保留历史分数、统计区间和工程门禁结果；项目验收决定不改写历史为统计等价已通过。

| 工作项 | 最新状态与执行边界 |
|---|---|
| 准确率与正常性能验证 | 按用户决定视为通过、无明显回退；短期关闭复测主线 |
| 最后三单元 | R163=76/100、R179=71/100、LITE179=74/100，均500步技术完成；旧“未启动/暂缓”不再适用 |
| LITE−R结果检查 | 12对平均−0.83pp，95%区间[−4.99,+3.32]；跨版本探索性，未复现稳定改善，不安排重跑 |
| 官方已发表系统主对照 | AReaL原生恢复开启，对比同版本AReaL＋完整R；计划70对/140次30步故障运行，未实施 |
| checkpoint组件补充 | 官方ByteCheckpoint集成前后加R，40次正式故障run；不能代表RobustRL完整系统，未实施 |
| 专用RL容错强对照 | RobustRL（OSDI 2026）官方artifact未确认可获取；保留条件性复现分支，不能用旧B5替代 |
| 下一步 | FT0版本/能力/恢复接口审计与适配，FT1功能预检；不先重做准确率或正常性能 |
| 预算 | 可获取分支含预检、主矩阵、组件对照及重复故障共最多210次，暂估209–352 GPU·小时；资源未预留 |

结果依据：[最后三单元检查](docs/experiments/rewardtxn-last-three-review-20260916/REPORT.md)。执行入口：[HANDOFF](HANDOFF.md)。新方案明确区分官方实现、官方组件集成和机制模拟；测量正确恢复、错误提交、RTO与重复计算，不将容错覆盖范围之外的功能缺失描述为原论文错误。

## 历史记录：以下均不代表当前待办

以下按原文保留早期进展、失败与当时决定。“最新执行范围”“尚未启动”“仍阻断”“下一步”等表述只对记录当时有效；当前任务、验收决定和补跑完成状态以上节及HANDOFF为准。原异步/同步准确率方案不进入短期执行队列。

**最新执行范围：先清理，再补最后3单元（取代全24重跑）。** 已清理完成的6个post-#2238 pilot共108个中间权重目录，释放约100GiB；终点权重/评估哈希通过，日志/诊断数据保留，当前确认及全部失败产物未动。补跑固定R/163→R/179→LITE/179，各从基座500步，新目录/新冻结，旧21结果单列；合并仅作跨版本探索性观察，不称统一版本独立确认。新候选独立复审/性能及运行前门禁仍阻断，尚未启动，无自动重试。见[清理清单](runs/SPACE_CLEANUP_20260913-215432.json)与[3单元准备清单](docs/experiments/rewardtxn-ablation-20260911/implementation/LAST_THREE_PREPARATION.json)。

**重启范围已明确：** 用户要求用同一最终修复版本从头完整重跑24个确认单元（原12seed × R/LITE、原顺序）；旧21个成功及失败产物单列保留，不补跑拼接。新冻结/目录及原运行门禁仍适用，尚未启动。见[重启准备](docs/experiments/rewardtxn-ablation-20260911/implementation/RESTART_PREPARATION.md)。

**2026-09-13 退出确认修复候选已完成：** 新isolated_v2_driver_exit保留R6/R7修复，并以真实父进程waitpid证据和独立确认线程替代Ray CLI收尾等待。87项CPU功能全通过(无跳过，含576payload)，13场景17单元真实CPU Ray控制链通过，真实Ray CLI探针退出0/ACK约0.18秒，30秒正常离线评估通过；未放宽10秒期限。尚待独立复审，新候选性能未测，原5%失败门禁仍阻断，不启动smoke/训练，不混旧freeze。见[候选结果](docs/experiments/rewardtxn-ablation-20260911/implementation/isolated_v2_driver_exit/CPU_RESULTS.md)和[重启准备](docs/experiments/rewardtxn-ablation-20260911/implementation/RESTART_PREPARATION.md)。

**2026-09-13 driver退出超时排查：** 原日志显示Ray核心关闭后，job supervisor/CLI和退出凭证仍需多层传递；凭证mtime已到10秒预算的第9.130秒，未及时收到driver确认。实际冻结wait_container每秒轮询且串行执行查询，4个隔离CPU时序对照通过，同一晚到凭证可随轮询时点通过或超时；机制已复现，原现场具体调用延迟仍未测得。见[排查报告](docs/experiments/rewardtxn-ablation-20260911/implementation/driver_exit_diagnosis_20260913/REPORT.md)。未修改源码、门禁或启动训练；R163仍失败，完整确认仍未完成。

**2026-09-13 15:31 confirm_recovery_03失败，未自动重试：** index20 LITE/seed163已成功；index21 R/seed163记录500步(0–499)并留下终点checkpoint，但driver关闭许可后10.022秒触发EXIT_TIMEOUT，未进入评估、无成功凭证。Ray job退出凭证为0，训练容器实际退出143/OOMKilled=false；恢复包装器最终OS退出码未持久记录，不能以代码return1或通知exit0代替。driver退出凭证mtime约为许可后9.130秒，结合1秒轮询提示关闭确认时序竞争，具体延迟原因未证实。现21个成功单元、1个失败、2个未启动；不得将完整训练或退出0追认为成功，不扩大超时、不补评估绕门禁。此前“恢复运行中”为历史。见[现场及核验目录](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/incidents/recovery03-20260913T073151Z)。

**2026-09-13 10:50 恢复再次失败，未再重试：** confirm_recovery_02在index20 LITE/seed163因固定GPU1/2/3上外部VLLM进程各占79436MiB触发foreign_gpu_process fatal，本次无training_step记录。训练容器实际退出143、OOMKilled=false；恢复包装器最终OS退出码未持久记录，不能将代码return1或通知投递exit0冒充实测退出码。前20个成功单元证据未变，后3单元未启动；首失败attempt已归档499文件、本次失败444文件哈希均通过。原420源码/402overlay与freeze/确认计划未变。10:48曾执行一次恢复，以下“未恢复”表述为此前历史；当前保持失败暂停，所有门禁及isolated_v1不放行结论不变。见[核验记录](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/incidents/recovery02-20260913T025047Z/executor_verification.json)。

**争用源首次取得直接归属证据：`dongjingkun`（uid 1008），宿主机会话，不在任何容器内。** 本次失败当场趁进程存活读取 /proc/PID/{cgroup,status,cmdline}，四个VLLM::Worker各占79436MiB，占用宿主机GPU0/1/2/3，与冻结设备重叠于1/2/3。早前推测容器`jyc-vllm-dev`为争用源属错误，该容器内无vllm进程。01:50与02:14两次争用未取得直接归属，不作等同认定。见[归属证据](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/contention_attribution_20260913.json)。

**2026-09-13 10:14 只读复查：外部GPU争用已再次出现，恢复窗口关闭；未启动任何训练。** 冻结设备2/3/4连同GPU0被新一批VLLM::Worker占用（pid 923144/923146/923148/923151，各约34.8GiB），与事故时的860188系列不是同一组进程；GPU1/5/6空闲，GPU7仍为无关外部进程。磁盘不变：根盘372GiB、/public 202GiB。10:05曾观察到短暂无争用（[快照](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/resource_recheck_20260913-1005.json)），该窗口已失效，不得据此认为门禁可过；共享卡反复被占正是恢复前必须实测而非沿用旧快照的原因。批次状态未变，未重跑、未换卡换seed、未修改冻结，失败attempt的499个文件原位保留。最新快照见[10:14资源复查](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/resource_recheck_20260913-1014.json)。

**2026-09-13 09:50 事故记录（资源状态以上方最新快照为准）：** 已核验20个成功凭证；第21单元LITE/seed163在step23后因外部GPU争用fatal，确认启动器退出1；余3单元未启动。事故后首次快照中固定GPU2/3/4被占用；10:05一度无争用，10:14又被新一批VLLM占用。未重试、未换seed/卡、未修改冻结，所有产物保留。见[失败证据与待审恢复计划](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/RECOVERY_PLAN.md)。以下确认启动描述为此前历史，不表示仍在运行。

**2026-09-13 实施验收阻断：R6/P1、R7/P2已由执行者独立复现。** ABORTED组重排失败及关闭计时记录MemoryError可被吞而不锁存fatal，原attempt终态审计不能补足此缺口。不能宣称完整满足v4 fatal合同；尚未证实本批数值受损。保留当前冻结及全部产物，不停训、不热改、不自动重跑。见[核查及处置方案](docs/experiments/rewardtxn-ablation-20260911/implementation/R6_R7_DISPOSITION.md)。此前通过记录仅代表当时已执行的检查，不覆盖新发现路径。
**隔离候选 v1 独立复审完成：** R6/R7候选功能修复通过独立复审；完整实施验收仍开放，CPU不放行、不能进入七组smoke。扩展总开销O16.4389%/R13.5869%超过预定5%，不能改阈值或将总开销全归于本轮修复。主审仅复算保存计时，未重跑性能。 原RM为12通过/1跳过，主审另补验576-payload通过。见[审查状态更正](docs/experiments/rewardtxn-ablation-20260911/implementation/isolated_v1_review_status.json)及[REVIEW.md](docs/experiments/rewardtxn-ablation-20260911/REVIEW.md)。统计依赖、有效argv、受评checkpoint运行时内容哈希增强仍未交付。


**执行更新：原批次七组 smoke 和初筛 21/21 单元通过当时正常路径检查（不代表完整实施验收）；LITE 是唯一符合原规则的独立确认候选。** R−O 为 −5/−4/+1pp（均值 −2.67pp）；LITE−R 为 +5/+6/+2pp（均值 +4.33pp）。初筛不能确认根因，LITE 是整组简化，不能归因单独 Seal/CAS/I/O。

当前批次：`/home/caiyiwen-rewardtxn-results/v4-0912-c`。初筛与确认冻结均通过独立复核；已冻结12个新seed的R/LITE配对顺序，共24run，此前包装器及运行前门禁通过，确认由runner PID2756403启动；现确认启动器已退出1，20个单元有成功凭证、1个失败、3个未启动，恢复待审且尚未执行。主判据为独立12对均值≥2pp且双侧95%配对t区间下界>0。不得追加seed、按质量剔除结果或自动重试。完整目标尚未完成。

结果：[RESULTS.md](docs/experiments/rewardtxn-ablation-20260911/RESULTS.md)；[初筛原始报告副本](docs/experiments/rewardtxn-ablation-20260911/implementation/screen_report.json)；[独立复核](docs/experiments/rewardtxn-ablation-20260911/implementation/screen_independent_review.json)；[确认冻结](docs/experiments/rewardtxn-ablation-20260911/implementation/confirmation_plan.json)；[执行笔记](docs/experiments/rewardtxn-ablation-20260911/implementation/NOTES.md)。

第10单元首次启动因外部GPU争用在训练前失败，415个文件完整归档；已按原顺序恢复且全部完成，失败未覆盖。计划/训练源/模型/输入/设备角色保持原冻结。确认所有产物写根目录NVMe；456GiB为启动前整阶段预算。10:05快照根盘剩余约372GiB，剩余4单元启动公式要求至少296GiB；/public约202GiB，恢复前须重测并持续满足200GiB门禁。保留全部checkpoint和失败证据，不修改正式E7协议或使用预留test500。终态通知仅在完成/失败时唤醒。

## 历史阶段记录

以下保留各阶段当时的状态；其中“未运行”“运行中”和资源数值不代表当前状态，当前状态以上表及链接结果为准。

## 2026-09-11 Slime官方队列修复已移植，实验未重启

按用户要求移植官方PR #2238的限量取队列、非阻塞完成队列及生成端背压，并保留本地ABORTED重置/超时。官方4项+本地5项CPU测试、现有11项restart测试通过；6组取4组后剩2组可供后续消费。未改历史pilot或启动新训练。详见[修复记录](/public/home/caiyiwen/rewardtxn/docs/slime_2238_backport_20260911.md)。下方“尚未修复”为修复前排查记录，不代表当前源码状态；修复后精度尚未验证。

## 2026-09-11 Pilot 误差来源排查

六次0.5B pilot均完成500步；Oracle验证准确率73/72/73%，RewardTxn为68/70/72%，质量门槛通过但等效功效门槛未过。已完成原始3,000批/96,000条样本核验及700条评估重评分。

发现当前Slime基线fully-async完成队列排空后只返回4组、超额组不回收的缺陷；最小复现及运行日志证实六次都有大量丢组，同seed两组实际消费轨迹不一致。未发现奖励绑定错配；不能把−2.667pp单独归因于RewardTxn或该队列缺陷。详见[排查报告](/public/home/caiyiwen/rewardtxn/runs/pilot-error-audit-20260911/REPORT.md)。本轮未修复训练代码或启动新实验；建议先修队列并受控复测，暂不直接进入正式实验。下方“pilot尚未运行”为历史阶段记录。

## 2026-09-11 配置落实

最新：用户确认0.5B，新版隔离数据入口、固定终点真实评估/质量门禁、pilot+输入冻结检查和显式manifest配对统计已实现，见 [重启流程](docs/e7_restart_0.5B_20260911.md)。旧启动器已委托新版入口，下段“数据隔离待适配”已由此次代码解决；但真实pilot、工作区版本整理和成功冻结仍待执行。正式入口缺证据时拒绝运行，未启动训练。

后续E7共用启动函数默认lr1e-6并开启`--use-rollout-logprobs`，覆盖直接入口和start_e7_clean包装入口；新meta记录概率参照开关。诊断对照与通用Day2默认不改。此为Slime已有参数的配置适配，不是底层loss bug修复。模拟参数传递测试通过，未启动新实验；旧入口数据隔离与正式冻结仍须另行处理。见 [配置说明](docs/e7_training_config_20260911.md)。

**更新时间**：2026-09-10 21:30（Asia/Shanghai）
**当前状态**：退化诊断完成。Oracle500为73/100、RewardTxn500为71/100，均无截断；两组审计与全部评估通过。原E7论文等价性仍未完成。
**交接入口**：[HANDOFF.md](HANDOFF.md)；详细过程见 [诊断记录](runs/diagnosis-20260910/STATUS.md)、[诊断方案](runs/diagnosis-20260910/PLAN.md)。

## 阶段总览

| 阶段 | 当前结果与边界 |
|---|---|
| Phase 1–3 | 已归档：GO 7/7、PASS 12/12、PASS 9/9 |
| E2 trace | 30,000 次、0 failure；真实代码切点与协议模型 fixture 分层报告 |
| E3/E6/E7 pilot | 18/18 短运行完成，不代表学习等价性功效门禁通过 |
| 原 E7 clean | 0.5B × seeds 29/42/73/101 × Oracle/RewardTxn，8/8 完成 500 步，但最终全部 0/500 |
| D0 数据与初始转换 | 完成：新划分隔离、初始权重往返一致 |
| D1 故障证据 | 完成：旧 8 run 前 100 步指标、两个 s29 iter449 完整文本评估 |
| D2 受控诊断 | A/B/C、固定输入 high/low/clipped、在线 D/E 全部完成 |
| D3 长程稳定性 | 两组均通过：Oracle73/100、RewardTxn71/100，均无截断 |
| 最终诊断与论文重启 | 诊断报告和完成审计已交付；论文级完整矩阵、配对统计及冻结仍未完成 |

基础证据：[基础交付](REWARDTXN_EXPERIMENT_DELIVERY.md)、[trace](runs/TRACE_REPORT_PAPER.json)、[pilot](runs/PILOT_RESULTS_20260831.md)。

## 已完成的退化诊断

### 数据与管道校准

- 原源文件 7,473 题包含旧 eval500；旧日志不足以重建实际训练消费交集，不能声称全部 500 题都参与过更新。
- 新划分为 train **6,373** / validation **100** / reserved test **500**，另排除旧 eval500；按原始索引与完整 prompt 两两无交集。6,973 条文件仅为排除旧 eval 后的中间产物，不是实际新训练集。
- 新验证集基座 **69/100、截断 0%**；固定前 20 题为 15/20。reserved test500 未用于本轮调参，但不声称旧模型未见过它。
- 初始 Megatron→HF 往返 BF16 参数完全一致，固定输入 logprob 差为 0、greedy64 一致；初始权重推送检查通过。这不证明后续每次权重更新均无同步问题。
- 两个旧 s29 iter449 均 **0/20、100% 截断**，完整输出已保存；旧 8 run 的指标确认退化在前 10–20 步已经出现，不是第 390–400 步才开始。

证据：[数据审计](runs/diagnosis-20260910/audit.json)、[往返比较](runs/diagnosis-20260910/roundtrip_comparison.json)、[基座验证](runs/diagnosis-20260910/base_validation.json)、[Oracle449](runs/diagnosis-20260910/oracle449.json)、[RewardTxn449](runs/diagnosis-20260910/rewardtxn449.json)。

### 50 步在线对照

各组均从基座开始，seed29、同一隔离数据。表内数字为正确题数；中间点各 20 题，终点 100 题。

| 组 | 模式 / 学习率 / 概率参照 | step5 | step10 | step20 | step30 | step50 | 终点截断 |
|---|---|---:|---:|---:|---:|---:|---:|
| A | fully-async / 5e-5 / 默认 | 8 | 0 | 0 | 0 | 0 | 99% |
| B | 真同步 train.py / 5e-5 / 默认 | 13 | 0 | 0 | 0 | 0 | 99% |
| C | fully-async / 1e-5 / 默认 | 12 | 12 | 9 | 7 | 24 | 1% |
| D | fully-async / 1e-5 / rollout logprobs | 16 | 15 | 8 | 6 | 44 | 1% |
| E | fully-async / 1e-6 / rollout logprobs | 13 | 15 | 15 | 13 | **64** | **0%** |

全部训练、计划评估及消费审计完成：每组 50 批 × 32 条，划分交集为 0，组大小与 token/logprob 长度正确、记录的数值有限。E 的所有评估点均无截断。

最终只读复核：基座、A–E、Oracle/RewardTxn共1,320条保存回答重跑verifier v1，标签、顺序、哈希、正确数和截断率汇总一致，见 [结果复核](runs/diagnosis-20260910/saved_results_audit.json)。这不是重新生成或独立人工判分。

预定诊断门槛为 validation100 **至少 59 题正确、截断不超过 10%**，并通过执行与消费/有限值审计。仅 E 通过短程门槛；此标准不是旧预注册的 1pp 等价检验，也不证明优于基座。

### 机制证据及限制

- B 真同步仍退化，因此异步样本滞后不是唯一解释；降低学习率缓解退化。
- 固定 A 首批 32 条输入重放 10 步：初始非零更新的奖励加权 logprob 方向正确，但默认参照下后期正确答案概率也下降。不能据此认定全局奖励符号反了。
- 默认配置缺少旧策略概率时，以当前 logprob.detach() 为参照，当前前向的比率为 1；固定输入实验中 clipfrac 始终为 0。加入 `--use-rollout-logprobs` 后，low 第 10 步正确答案平均 token logprob 变化从 -0.637 改善至 +0.094，在线终点 C→D 从 24/100 改善至 44/100。
- 证据支持“更新幅度与约束不足”是优先机制解释，但固定输入重复训练不等于原在线流程，不能宣布已证明唯一根因。E 同时需要更低学习率，本次两组500步复验均通过；单seed结论不外推为普遍稳定。

## 最终配对结果与下一步

| 组 | step50/100/250（各20题正确数） | step500（100题） | 评估截断 | 训练审计 |
|---|---|---:|---:|---|
| Oracle | 10 / 14 / 14 | 73 | 全部0% | 500批/16,000条，通过 |
| RewardTxn | 12 / 11 / 14 | 71 | 全部0% | 500批/16,000条，通过 |

两组训练及全部评估均exit0，固定终点iter499，不选择好中间点替代。实际CLI除run路径和奖励函数外一致，均0.5B、seed29、fully-async、lr1e-6与rollout logprobs参照。执行链已结束，没有诊断容器仍在运行。

交付：[诊断报告](runs/diagnosis-20260910/REPORT.md)、[逐项完成审计](runs/diagnosis-20260910/COMPLETION_AUDIT.md)、[论文重启前置条件](runs/diagnosis-20260910/PAPER_RESTART_PREREQUISITES.md)。建议后续以该组合为已验证起点，先完成论文方案/统计/冻结前置条件；本轮不自动恢复多seed/faulted全矩阵。73与71的差异不支持统计优劣或1pp等价声明。

## 原 E7 与论文结论边界

旧 8 个最终模型均 0/500，旧基座 286/500（57.2%）。原计划为 1.5B × 5 seeds × 5 groups，实际缩为 0.5B × 4 seeds × 2 clean groups；不能称完整 E7 已完成。旧 metadata 的 lr1e-4 与实际 5e-5 不符，旧冻结状态未闭合；统计脚本为非配对 Welch 且未处理零方差。不得用零分一致宣称学习等价，不能改写旧预注册或用中间 checkpoint 替换固定 step500 终点。

## 清理、资源与版本

9 月 9 日经用户确认永久清理约 95.5 GiB，包括两份 LLaDA 模型；路径和保留清单见 [交接清理记录](HANDOFF.md#清理与产物保留)。本轮诊断保留新证据，未再次删除模型或 checkpoint。

21:30 文件系统可用约 **397 GiB（95% 已用）**。诊断证据继续保留；共享资源不视为预留，未来启动须重新门禁。

诊断期间新增专用脚本，并在训练入口增加学习率/同步入口等参数传递及元数据修正；未修改底层 RL loss，未回写旧结果。本轮诊断执行已完成并归档；没有提交Git，没有执行论文重启或再次清理。

**文档版本**：6.0（2026-09-16：容错对照主线；此前内容保留为历史）
