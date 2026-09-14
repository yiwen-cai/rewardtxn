# RewardTxn 当前交接

**2026-09-14：先整理并提交当前结论；SQLite完整验证仍未完成。**

最新结论与可随Git保存的证据见[SQLite阶段结论](docs/experiments/rewardtxn-mechanism-20260913/conclusions_20260914/CONCLUSIONS.md)。固定回答CPU RM成本下降21.9%的结论保持不变；在线R、DBM已各完成500步和终点评估，R重复仍缺失，不能宣称准确率改善或根因成立。

原R重复在step217后因GPU1争用失败。用户已授权只重跑最后一轮；恢复查询限定GPU1–4后资源检查通过，但NVIDIA容器运行时仍报GPU requires reset，恢复训练零步。GPU5单卡reset返回Not Supported（用户终端输出），重启及设备恢复尚未验证。前两轮、原失败轮、恢复失败目录和claim保留；恢复前需准备保留这些现场的新入口，不能直接重跑旧recover_last.py。

本次优先保存结论；设备恢复后核验GPU/容器，再接续最后一轮、完成三轮比较和独立审计。原CPU结论、冻结源码和既有门禁限制不变。以下为此前阶段历史与暂缓事项。

## 暂缓的原三单元补跑安排

以下为SQLite优先级调整前的安排，暂不执行；不构成当前SQLite任务的前置顺序。

当前执行安排：清理不需要的数据后，仅补跑最后3个确认单元。**清理已完成；补跑尚未启动，运行门禁仍阻断。** 新完成回退机理的离线诊断与CPU对照：已找到优先验证的SQLite成本因素，尚未证实准确率回退的因果链。此摘要为当前执行依据，下方记录中的旧PID、“运行中”和“全24重跑”均按其历史时间理解。

| 项目 | 当前状态 |
|---|---|
| 旧确认批次 | `v4-0912-c`：21个单元有成功凭证；R/163关闭超时失败；R/179、LITE/179未启动 |
| 最新补跑范围 | `R/163 → R/179 → LITE/179`，各从基座训练500步；不续用失败轨迹，不自动重试 |
| 数据清理 | 已删除6个已完成post-#2238 pilot的108个中间权重目录，释放约100GiB；终点/评估SHA验证通过，日志和诊断数据保留 |
| 资源快照 | 清理后`/public`约401GiB、结果所在根盘约330GiB；这是历史快照，实际启动前重新检查 |
| 退出修复 | 新隔离候选`isolated_v2_driver_exit`：87项CPU测试通过、无跳过；真实Ray CLI与30秒离线评估通过，未放宽10秒门禁 |
| 当前阻断 | 新候选待独立复审、未完成性能测量；既有5%开销门禁失败仍适用，同版本smoke及运行证据增强尚未完成 |
| 结果解释 | 新3单元使用独立目录/新冻结，旧21单元单列；合并仅为跨版本探索性观察，不能称统一版本24单元独立确认，seed163配对跨版本 |
| 回退机理调查 | 离线与CPU验证已完成：DBM在固定回答下使RM耗时下降21.9%，奖励及逻辑记录一致；SQLite开销是优先验证因素，尚不能归因为准确率回退根因 |

下一步：先完成候选复审及性能/运行证据验收，再进行同版本smoke和资源预检；通过后按已授权顺序一次性启动3单元。没有新增的跳门禁或自动重试授权。原确认成功/失败产物、旧freeze与isolated_v1/v2已交付包保持不变，不热改或补签失败单元成功凭证。

接续执行与验收顺序：

1. 完成`isolated_v2_driver_exit`独立复审、最终版本固定完整性能测量及运行证据增强（统计依赖、有效argv、受评checkpoint内容哈希）。验收：功能/控制/语义证据绑定最终源码，原5%性能门禁通过并有复审结论。
2. 在同一最终版本上完成七组smoke，重新实测资源、GPU UUID、模型/数据与空间门禁。验收通过后冻结新目录的3单元清单与一次性包装器；现有准备清单不等于最终运行冻结。
3. 按`R/163 → R/179 → LITE/179`各从基座训练500步，固定`iter_0000499_hf`、validation100终点评估，不使用reserved test500。验收以完整成功凭证为准，失败即停、不自动重试；结果逐行标注版本，旧21单元单列。

**状态来源与防误读：** 本次读取原批次[pipeline_status.json](/home/caiyiwen-rewardtxn-results/v4-0912-c/pipeline_status.json)，仍为`failed / confirm_recovery_03 / R163 / EXIT_TIMEOUT`。[LAST_THREE_PREPARATION.json](docs/experiments/rewardtxn-ablation-20260911/implementation/LAST_THREE_PREPARATION.json)记录`started=false`且门禁阻断，作为最新补跑范围依据。[engineering_status.json](docs/experiments/rewardtxn-ablation-20260911/implementation/engineering_status.json)混有历史字段：顶层`cpu_release=true`和旧四单元`pending`不代表当前放行或范围，应结合`current_cpu_release=false`、`driver_exit_fix_candidate`及`requested_restart_scope`读取。机理交付验收见[acceptance.json](docs/experiments/rewardtxn-mechanism-20260913/artifacts/acceptance.json)，其通过不解除训练门禁。

入口：[3单元准备清单](docs/experiments/rewardtxn-ablation-20260911/implementation/LAST_THREE_PREPARATION.json)、[重启准备与门禁](docs/experiments/rewardtxn-ablation-20260911/implementation/RESTART_PREPARATION.md)、[退出修复CPU结果](docs/experiments/rewardtxn-ablation-20260911/implementation/isolated_v2_driver_exit/CPU_RESULTS.md)、[清理凭证](runs/SPACE_CLEANUP_20260913-215432.json)。独立审查任务：`01a090c3-8b2f-7cb3-b106-2a3cfea661a8`，已收到候选与最新3单元范围投递；投递成功不代表审查已通过。

## 2026-09-13 回退机理：离线诊断与CPU验证已完成

**当前结论：有候选原因，不是“没有找到方向”，也不是“已找到准确率根因”。** SQLite持久化开销可能通过异步完成/消费时序影响训练。已证实的是操作成本差及运行间消费轨迹分歧；“SQLite开销→轨迹变化→准确率下降”后两段因果连接仍未验证，不能把该因素认定为唯一原因。

- **离线证据：** 原七组初筛21个单元全部分析；原确认成功21个单元仅复查成本和消费轨迹，10个完整R/LITE配对，LITE/163单列。672,000条消费样本与debug、诊断摘要、奖励记录及原题目/标签交叉核对通过。初筛3/3、确认10/10配对中R的平均RM耗时均更高，十个50步窗口的130/130配对窗口同向；窗口不是独立训练重复。同题频次重合约99%，同一步题目重合约6.5%–10.8%，完整回答几乎不重合，版本滞后差异方向不一致。不同回答的工作量及同方法重复的轨迹噪声仍是解释限制。
- **固定回答CPU对照：** 输入固定为初筛R/LITE × seed11/23/37 × step0/249/499，共18个batch、576条完整回答、72个八回答组。七变体剖析后按预定规则在DBM/LOGM/PAYLOAD中选中DBM（SQLite内存模式）。正式比较原R、DBM、恢复原实现的R，21个随机区块、前三个预热、保留18个，种子20260913；只执行一次正式计时。DBM−R耗时差中位数−38.878 ms，95%区块bootstrap区间[−39.631, −38.029] ms，相对降幅21.9%；恢复R−原R中位数+0.174 ms，区间[−0.530, +0.568] ms，恢复波动检查正常。逐项奖励、输入对象、返回位置、规范化逻辑记录与实际加载源码绑定均通过，原始归档独立复算一致。此降幅是固定回答的CPU RM成本，不是训练吞吐或准确率改善。
- **尾部证据限制：** 旧`queue_complete`在实际put前，不能作为成功入队凭证；`quiesce`在等待剩余任务结束前记录队列长度。加入其后完成的入队前观测，42/42单元的评分尾部均可在总量上对齐；单条最终去向仍不能逐一证明，不认定为丢失。
- **验收与边界：** 25项解析/统计测试、七变体真实CPU语义验证通过；21,800个原始输入文件前后SHA一致，五张静态图由CSV重建哈希一致。所有终点评估文件仅校验字节哈希，未解析确认准确率，也未用中途准确率挑因素。未新增训练、未占GPU、未生成回答或复制模型/checkpoint；交付约172 MiB，已校验归档并清理临时副本。原R6/R7证据限制继续保留，**本轮结果不替代退出修复候选原有5%性能门禁，不改变最后3单元补跑安排**。

**后续机理实验建议（未实施，不作为新增训练启动授权）：** 在既有候选验收与运行门禁满足后，另行预注册R vs DBM配对训练，只切换SQLite内存模式，其余源码、基座、seed、数据、保存频率和资源配置固定；纳入R同方法重复估计轨迹噪声。预期RM成本下降，若此因素解释质量回退则终点准确率应相对R改善，并检查轨迹差异是否超出重复噪声。成本不降、成本下降但轨迹无一致变化、或轨迹变化而准确率未恢复，均削弱对应因果链；不得根据结果追加有利seed或追测到通过。

复查入口：[中文报告](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-mechanism-20260913/artifacts/report/REPORT.md)、[CLI与复现说明](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-mechanism-20260913/README.md)、[输入与源码清单](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-mechanism-20260913/artifacts/snapshot/manifest.json)、[逐seed配对](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-mechanism-20260913/artifacts/analyze/pairs.csv)、[CPU协议](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-mechanism-20260913/artifacts/cpu/formal_protocol.json)、[独立复算](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-mechanism-20260913/artifacts/report/independent_cpu_recomputation.json)、[交付文件SHA清单](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-mechanism-20260913/artifacts/delivery_manifest.json)。工具位于独立目录，不加入`scripts/ablation_*.py`冻结扫描集合；`snapshot/analyze/cpu/report`均显式接收批次和输出目录，拒绝覆盖阶段结果。

## 阶段记录（保留追溯）

**最新执行范围：先清理，再补最后3单元（取代全24重跑）。** 已清理完成的6个post-#2238 pilot共108个中间权重目录，释放约100GiB；终点权重/评估哈希通过，日志/诊断数据保留，当前确认及全部失败产物未动。补跑固定R/163→R/179→LITE/179，各从基座500步，新目录/新冻结，旧21结果单列；合并仅作跨版本探索性观察，不称统一版本独立确认。新候选独立复审/性能及运行前门禁仍阻断，尚未启动，无自动重试。见[清理清单](runs/SPACE_CLEANUP_20260913-215432.json)与[3单元准备清单](docs/experiments/rewardtxn-ablation-20260911/implementation/LAST_THREE_PREPARATION.json)。

**已被最新3单元补跑范围取代的历史方案：** 用户此前要求用同一最终修复版本从头完整重跑24个确认单元（原12seed × R/LITE、原顺序）；旧21个成功及失败产物单列保留，不补跑拼接。新冻结/目录及原运行门禁仍适用，尚未启动。见[重启准备](docs/experiments/rewardtxn-ablation-20260911/implementation/RESTART_PREPARATION.md)。

**2026-09-13 退出确认修复候选已完成：** 新isolated_v2_driver_exit保留R6/R7修复，并以真实父进程waitpid证据和独立确认线程替代Ray CLI收尾等待。87项CPU功能全通过(无跳过，含576payload)，13场景17单元真实CPU Ray控制链通过，真实Ray CLI探针退出0/ACK约0.18秒，30秒正常离线评估通过；未放宽10秒期限。尚待独立复审，新候选性能未测，原5%失败门禁仍阻断，不启动smoke/训练，不混旧freeze。见[候选结果](docs/experiments/rewardtxn-ablation-20260911/implementation/isolated_v2_driver_exit/CPU_RESULTS.md)和[重启准备](docs/experiments/rewardtxn-ablation-20260911/implementation/RESTART_PREPARATION.md)。

**2026-09-13 driver退出超时排查：** 原日志显示Ray核心关闭后，job supervisor/CLI和退出凭证仍需多层传递；凭证mtime已到10秒预算的第9.130秒，未及时收到driver确认。实际冻结wait_container每秒轮询且串行执行查询，4个隔离CPU时序对照通过，同一晚到凭证可随轮询时点通过或超时；机制已复现，原现场具体调用延迟仍未测得。见[排查报告](docs/experiments/rewardtxn-ablation-20260911/implementation/driver_exit_diagnosis_20260913/REPORT.md)。未修改源码、门禁或启动训练；R163仍失败，完整确认仍未完成。

**2026-09-13 15:31 confirm_recovery_03失败，未自动重试：** index20 LITE/seed163已成功；index21 R/seed163记录500步(0–499)并留下终点checkpoint，但driver关闭许可后10.022秒触发EXIT_TIMEOUT，未进入评估、无成功凭证。Ray job退出凭证为0，训练容器实际退出143/OOMKilled=false；恢复包装器最终OS退出码未持久记录，不能以代码return1或通知exit0代替。driver退出凭证mtime约为许可后9.130秒，结合1秒轮询提示关闭确认时序竞争，具体延迟原因未证实。现21个成功单元、1个失败、2个未启动；不得将完整训练或退出0追认为成功，不扩大超时、不补评估绕门禁。此前“恢复运行中”为历史。见[现场及核验目录](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/incidents/recovery03-20260913T073151Z)。

**2026-09-13 14:34 第三次恢复已启动，运行中，未完成：** 按`recovery_plan_03.json`执行index20-23，主进程PID1724931、阶段`confirm_recovery_03`、首单元LITE/seed163，容器rtxa-5c3483efb686已起并在初始化权重。启动前`verify(BATCH)`通过、freeze与confirmation_plan的SHA一致、冻结设备1/2/3/4无外来compute进程、根盘370GiB（剩余4单元要求≥296GiB）、/public 301GiB。index20第三次从头跑满500步，不续用任何受争用轨迹；前20单元不重跑。**启动依据为用户在会话中的直接授权，RECOVERY_PLAN第1条的独立审查者复核仍未取得。** 预计约2-3小时，不自动重试；失败即停并由包装器当场抓取现场。运行中不代表任何质量结论；R6/R7实施验收缺口与isolated_v1不放行结论均不变。见[启动记录](/home/caiyiwen-rewardtxn-results/v4-0912-c/recovery_03_launch.json)。

**失败attempt累计三份，全部原位保留：** `failed_attempts/screen-09-LITE-s23-attempt1`（415文件，初筛阶段）、`confirm-20-LITE-s163-attempt1`（499文件，曾到step23）、`confirm-20-LITE-s163-attempt2`（444文件，无training_step）。第三份归档前逐文件核验444项SHA全部匹配incident快照，归档后原路径腾空供重跑，不覆盖既有归档。

**2026-09-13 14:30 空间清理：`/public`释放100.0 GiB（200.7→300.7 GiB）。** 经用户批准，仅删除pre-#2238那批6个pilot（后缀`20260910-222540`，见pre2238归档清单）的108个非终点checkpoint目录，每个run保留`iter_0000499`/`iter_0000499_hf`及全部日志、`rewards.jsonl`、`seals.jsonl`、`meta.json`、`restart_eval`、`diagnosis_summary.json`；6个run各由18.5GiB降至2.1GiB。post-#2238批次及其他run未触及，无run目录被删除。pilot既有准确率数字的证据链完整，但这些中间步权重已不可再评估。逐条路径与字节数见[清理记录](runs/SPACE_CLEANUP_20260913-143047.json)。

**2026-09-13 10:50 恢复再次失败，未再重试：** confirm_recovery_02在index20 LITE/seed163因固定GPU1/2/3上外部VLLM进程各占79436MiB触发foreign_gpu_process fatal，本次无training_step记录。训练容器实际退出143、OOMKilled=false；恢复包装器最终OS退出码未持久记录，不能将代码return1或通知投递exit0冒充实测退出码。前20个成功单元证据未变，后3单元未启动；首失败attempt已归档499文件、本次失败444文件哈希均通过。原420源码/402overlay与freeze/确认计划未变。10:48曾执行一次恢复，以下“未恢复”表述为此前历史；该次失败后曾保持暂停，现已由14:34第三次恢复取代（见顶部），isolated_v1不放行结论不变。见[核验记录](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/incidents/recovery02-20260913T025047Z/executor_verification.json)。

**争用源首次取得直接归属证据：`dongjingkun`（uid 1008），宿主机会话，不在任何容器内。** 本次失败当场趁进程存活读取 /proc/PID/{cgroup,status,cmdline}，四个VLLM::Worker（pid 1083151/1083152/1083158/1083162）cgroup均为`user.slice/user-1008.slice/session-81106.scope`，各占79436MiB，占用宿主机GPU0/1/2/3，与冻结设备重叠于1/2/3。本会话早前曾推测容器`jyc-vllm-dev`（jianyuchen1）为争用源，该推测错误：该容器内只有sleep infinity与vscode-server、无vllm进程，不得据此处置。01:50与02:14两次争用的进程在检查前已退出，仅有进程名/TP形态/占卡顺序相似，未取得直接归属，不作等同认定。GPU4/5/6在三次事件中均未被占。见[归属证据](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/contention_attribution_20260913.json)。

**2026-09-13 10:14 只读复查：外部GPU争用已再次出现，恢复窗口关闭；未启动任何训练。** 冻结设备2/3/4连同GPU0被新一批VLLM::Worker占用（pid 923144/923146/923148/923151，各约34.8GiB），与事故时的860188系列不是同一组进程；GPU1/5/6空闲，GPU7仍为无关外部进程。磁盘不变：根盘372GiB（剩余4单元公式要求≥296GiB），/public 202GiB，仅高于200GiB门禁2GiB。批次状态与事故时一致：confirm-00~19共20个成功单元、confirm-20-LITE-s163失败目录499个文件原位保留、index21–23未创建，`pipeline_status.json`仍为failed/confirm/exit1，批次近2小时无新增文件。10:05曾观察到短暂无争用（[快照](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/resource_recheck_20260913-1005.json)），该窗口已失效；恢复仍须按RECOVERY_PLAN第1/3条先经审查者复核并另建一次性恢复执行记录，启动前实测资源、重做设备UUID绑定与源码/overlay/模型/数据/确认计划完整校验，不得沿用任何历史快照。本次未重跑、未换卡换seed、未修改冻结。最新快照见[10:14资源复查](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/resource_recheck_20260913-1014.json)。

**2026-09-13 09:50 事故记录（资源状态以上方最新快照为准）：** 已核验20个成功凭证；第21单元LITE/seed163在step23后因外部GPU争用fatal，确认启动器退出1；余3单元未启动。事故后首次快照中固定GPU2/3/4被占用；10:05一度无争用，10:14又被新一批VLLM占用。未重试、未换seed/卡、未修改冻结，所有产物保留。见[失败证据与待审恢复计划](docs/experiments/rewardtxn-ablation-20260911/implementation/confirm_failure_20260913/RECOVERY_PLAN.md)。以下确认启动描述为此前历史，不表示仍在运行。

**2026-09-13 实施验收阻断：R6/P1、R7/P2已由执行者独立复现。** ABORTED组重排失败及关闭计时记录MemoryError可被吞而不锁存fatal，原attempt终态审计不能补足此缺口。不能宣称完整满足v4 fatal合同；尚未证实本批数值受损。保留当前冻结及全部产物，不停训、不热改、不自动重跑。见[核查及处置方案](docs/experiments/rewardtxn-ablation-20260911/implementation/R6_R7_DISPOSITION.md)。此前通过记录仅代表当时已执行的检查，不覆盖新发现路径。
**隔离候选 v1 独立复审完成：** R6/R7候选功能修复通过独立复审；完整实施验收仍开放，CPU不放行、不能进入七组smoke。扩展总开销O16.4389%/R13.5869%超过预定5%，不能改阈值或将总开销全归于本轮修复。主审仅复算保存计时，未重跑性能。 原RM为12通过/1跳过，主审另补验576-payload通过。见[审查状态更正](docs/experiments/rewardtxn-ablation-20260911/implementation/isolated_v1_review_status.json)及[REVIEW.md](docs/experiments/rewardtxn-ablation-20260911/REVIEW.md)。统计依赖、有效argv、受评checkpoint运行时内容哈希增强仍未交付。


**执行更新：原批次七组 smoke 和初筛 21/21 单元通过当时正常路径检查（不代表完整实施验收）；LITE 是唯一符合原规则的独立确认候选。** R−O 为 −5/−4/+1pp（均值 −2.67pp）；LITE−R 为 +5/+6/+2pp（均值 +4.33pp）。初筛不能确认根因，LITE 是整组简化，不能归因单独 Seal/CAS/I/O。

当前批次：`/home/caiyiwen-rewardtxn-results/v4-0912-c`。初筛与确认冻结均通过独立复核；已冻结12个新seed的R/LITE配对顺序，共24run，此前包装器及运行前门禁通过，确认由runner PID2756403启动；现确认启动器已退出1，20个单元有成功凭证、1个失败、3个未启动，恢复待审且尚未执行。主判据为独立12对均值≥2pp且双侧95%配对t区间下界>0。不得追加seed、按质量剔除结果或自动重试。完整目标尚未完成。

结果：[RESULTS.md](docs/experiments/rewardtxn-ablation-20260911/RESULTS.md)；[初筛原始报告副本](docs/experiments/rewardtxn-ablation-20260911/implementation/screen_report.json)；[独立复核](docs/experiments/rewardtxn-ablation-20260911/implementation/screen_independent_review.json)；[确认冻结](docs/experiments/rewardtxn-ablation-20260911/implementation/confirmation_plan.json)；[执行笔记](docs/experiments/rewardtxn-ablation-20260911/implementation/NOTES.md)。

第10单元首次启动因外部GPU争用在训练前失败，415个文件完整归档；已按原顺序恢复且全部完成，失败未覆盖。计划/训练源/模型/输入/设备角色保持原冻结。确认所有产物写根目录NVMe；456GiB为启动前整阶段预算。10:05快照根盘剩余约372GiB，剩余4单元启动公式要求至少296GiB；/public约202GiB，恢复前须重测并持续满足200GiB门禁。保留全部checkpoint和失败证据，不修改正式E7协议或使用预留test500。终态通知仅在完成/失败时唤醒。

## 历史执行记录

以下为当时的启动与诊断记录，不是待执行任务。旧PID仅用于追溯；已完成目录不得复用。

## 2026-09-11 16:30：启动两份 Oracle 同seed观察实验

该次用户指定改为同一种方法两次独立启动，替代此前的B6→Oracle复跑方案。现两次Oracle已完成，结果见上方当前状态。选择GPU1–4，Oracle/group_rm、seed11、0.5B、各500步，同一组卡顺序执行。启动前资源和数据门禁通过。后台PID2703273，日志 `logs/oracle_repeat_20260911-083022.log`，启动记录 `logs/oracle_repeat_launch.json`。

独立实验清单在 `runs/oracle-repeat-s11-20260911-083022/manifest.json`，两个run为 `e7restart-oracle-repeat-r1-s11-20260911-083022` 和 `e7restart-oracle-repeat-r2-s11-20260911-083022`。每次从基座启动，均执行原100题终点评估，全部成功后自动写同目录 `comparison.json`，比较准确率、逐题胜负和消费轨迹。源文件SHA在启动时记录、每次开始前复核。原pilot清单未修改。没有持续监控或自动重试；后续追溯以本清单和日志为准，不再启动旧的B6→Oracle方案。

## 2026-09-11 16:25：修复后 pilot 因果排查

本轮CPU审计和3次固定GPU回放完成。96,000条消费奖励绑定正常；576条原始回答经两条真实RM（含独立CAS/Seal I/O）重评分同序逆序均一致。固定同batch、seed11、GPU1的 Oracle/B6/Oracle 重复对照：实际奖励、优势、returns、mask、logprob等张量完全相同；首个非零更新中跨方法权重差异L2=5.225e-6，同方法重复差异L2=5.195e-6，未发现RewardTxn特有的单步偏差，但存在同方法数值波动，不能据此归因500步端点差距。生成权重平均滞后两组均约22次更新，没有明显B6特有滞后。

报告与边界：[因果排查报告](runs/pilot-causal-audit-20260911/REPORT.md)。计划与结果JSON均在同目录。旧pilot清单与结果未修改。

**已被替代的方案：B6→Oracle复跑。** 该方案在16:24因GPU不足未启动，随后用户指定改为两份Oracle，并已执行完成。`scripts/e7_causal_repeat.sh`保留为历史准备入口，不再按其旧授权自动启动；当前下一步是上方链接的七组消融方案。

## 2026-09-11 11:27：按用户要求重启修复后 pilot

使用已核对 SHA 的 #2238 修复，启动原定 seed 11/23/37 × group_rm/b6、0.5B、各500步，GPU0–3。新批次后缀 `20260911-032711`，后台矩阵 PID `1802655`；六项 preflight 和启动资源门禁通过。旧清单完整归档为 `runs/e7_restart_0.5B_20260911_pilot_pre2238_archived_20260911-032711.json`，旧 run 保留。当前清单 `runs/e7_restart_0.5B_20260911_pilot.json` 指向新六run；启动日志 `logs/e7_pilot_post2238_20260911-032711.log`，源码SHA及启动记录 `logs/e7_pilot_post2238_launch.json`。后台流程依次训练、评估并在全部成功后聚合；未启动正式实验，不作完成或质量声明。按用户要求仅确认启动，不持续跟踪或新增监控任务。

## 2026-09-11 最新：已移植官方 #2238，不重启实验

共同队列缺陷已按官方补丁修复，包含超额组保留、完成回调非阻塞与生成背压；原本地ABORTED重置和超时保留。9项队列CPU测试+11项restart测试通过。生产文件修复后SHA为`b9ab67e7a3a1f8b2772560898128079d26c38c7b8c0e582f7a21c9c1a55fcea7`。详见[移植及验证记录](/public/home/caiyiwen/rewardtxn/docs/slime_2238_backport_20260911.md)。

用户明确要求先不要重启实验，本轮未启动pilot/正式训练或监控。不要把旧pilot解释为修复后结果；后续训练需要用户确认、新run与源码版本记录，仍无精度恢复结论。下面的未修复描述仅为历史排查记录。

## 2026-09-11 最新：Pilot 误差排查

6次pilot已完成，原始数据与评估均复核。发现项目使用的Slime fully-async基线会静默丢弃超额已完成组；已用实际函数CPU复现，并从六run日志得到合计21,666组丢弃下界。同seed共享消费组仅38.25%–41.50%，不能将当前差距归因于奖励语义损失。排查与未证实的因果边界见[报告](/public/home/caiyiwen/rewardtxn/runs/pilot-error-audit-20260911/REPORT.md)。

本轮只新增`e7_pilot_*audit.py`审计脚本、证据和文档，没有修训练实现/改评分/启动训练。后续应先修队列守恒与测试，再另建受控诊断；旧pilot不可冒充修复后结果。权限问题已补救、历史exit_code保留；验证准确率仍为Oracle73/72/73与RTX68/70/72，RTX seed23截断2%，其余0%。正式冻结及Faulted B5仍未完成。下方未运行pilot描述仅为历史记录。

## 2026-09-11 后续脚本调整

最新入口已升级：`scripts/e7_formal_launch.sh --stage pilot --check` 只读检查6个独立pilot计划；默认formal检查须真实pilot与有效冻结。数据隔离、真实评估和配对统计已接入，新协议为 `prereg/restarts/0.5B-20260911/protocol.json`。真实pilot尚未运行，工作区尚未整理为可冻结状态，Faulted B5仍未实现。详见 [新版流程与未完成项](docs/e7_restart_0.5B_20260911.md)；下段共用函数信息为上一版，不再作为执行入口。

E7共用`run_formal_e7`现默认lr1e-6并保证开启rollout logprobs，`start_e7_clean.sh`导入同函数，Phase2新meta记录开关。通用Day2和A–E诊断对照保持不变；未启动实验或修改旧prereg。注意旧入口的源数据默认与路径预检、正式冻结/评估门禁仍待下一轮适配，不能直接当作正式重启授权。原理、验证与范围见 [配置说明](docs/e7_training_config_20260911.md)。

**更新时间**：2026-09-10 21:30（Asia/Shanghai）
**状态**：退化诊断完成；不是原E7论文等价性通过。
**入口**：[PROGRESS.md](PROGRESS.md)、[诊断报告](runs/diagnosis-20260910/REPORT.md)、[逐项完成审计](runs/diagnosis-20260910/COMPLETION_AUDIT.md)。

## 最终结果

| 模型/组 | validation100正确数 | 截断 | 证据状态 |
|---|---:|---:|---|
| 未训练基座 | 69 | 0% | 基座对照 |
| E短程50步 | 64 | 0% | 通过候选门槛 |
| Oracle500 | 73 | 0% | 训练、500批审计及全部评估通过 |
| RewardTxn500 | 71 | 0% | 训练、500批审计及全部评估通过 |

预定门槛为至少59/100、截断≤10%、记录数值有限、消费审计通过，不是1pp等价。两组各500批/16,000条实际送训样本；step50/100/250各20题正确数分别Oracle10/14/14、RewardTxn12/11/14，全部无截断。两组均从基座开始，不是恢复优化器的续训。

## 运行已结束：不要重启旧链

执行链会话49623已真实exit0；原PID3388352和训练日志捕获PID不再作为续接入口。Oracle与RewardTxn训练及四个长程评估容器均已退出成功，当前没有诊断容器运行。脚本拒绝覆盖评估结果，不要重新执行整个finish_chain。

最终run：
- `/public/home/caiyiwen/rewardtxn/runs/diagnosis-E7-D3-oracle-s29-20260910`
- `/public/home/caiyiwen/rewardtxn/runs/diagnosis-E7-D3-rewardtxn-s29-20260910`

各目录的`diagnostic_eval/step500_n100.json`引用固定`checkpoints/iter_0000499_hf`；原始日志、终态、消费摘要与debug批次保留。没有用中间点替换终点。

## 已完成证据与解释边界

- D0：实际train6373与源行匹配；validation100/reserved test500/old eval500按索引与完整prompt隔离；初始往返参数/logprob差0、greedy64一致。旧实际消费交集无法重建，限制保留。
- D1：旧8-run前100步指标及两个s29 iter449完整评估完成；旧退化在前10–20步，449均0/20且100%截断。
- D2：A/B/C/D/E终点分别0/0/24/44/64（各100题）；全部训练审计与计划评估完成。真同步也退化；降低LR和rollout概率参照改善表现。
- 固定输入high/low/clipped均完成；证据支持更新幅度/约束问题，但不是唯一根因证明。正优势回答概率变化不等于所有正确样本平均。
- D3：两组匹配实际训练argv（仅run路径/RM函数归一化），lr1e-6、use-rollout-logprobs、seed29、0.5B、fully-async、GPU0/5/6/7。配置匹配不是消费轨迹逐样本相同。
- [结果复核](runs/diagnosis-20260910/saved_results_audit.json)：1,320条保存回答重跑verifier v1、数据标签/顺序/哈希/汇总检查通过。这不是独立人工判分；原始批次审计另有证据。
- 单seed、反复使用validation选配置、未做1e-6默认参照对照；不报告多seed稳定、泛化显著提升或1pp等价。保留reserved test，不用于追加调参。

详细失败启动、脚本变更与阶段记录见 [STATUS.md](runs/diagnosis-20260910/STATUS.md)；其中“运行中”是历史，当前状态以本交接为准。

## 后续建议（不自动执行）

按 [D4前置条件](runs/diagnosis-20260910/PAPER_RESTART_PREREQUISITES.md) 决定下一版模型/矩阵，验证配对TOST和零方差处理，完成pilot功效/精度及冻结，保护最终测试。旧E7实际0.5B×4seeds×2clean、全部0/500，不等于预注册1.5B×5×5完整完成；不得回写旧结果或事后放宽margin。

诊断脚本在`scripts/e7_diagnosis_*`。复核保存结果可在Slime只读CPU容器运行`scripts/e7_diagnosis_verify_results.py`；它不重新生成、不占GPU。需要新实验时另记run与方案，重新资源门禁，不重启旧链、不自动删除证据。

## 清理与产物保留

2026-09-09 经用户确认完成两轮永久删除，按当时 `df` 可用空间增量合计约 **95.5 GiB**。共享磁盘上的其他活动会影响后续可用空间。

第一轮约 54.4 GiB，包括缓存（C++ 索引、DGL Reddit、3 个 VS Code staging、旧 Codex 目录）和以下 checkpoint 路径：

```text
runs/smoke-K8-s42-20260828-232940/checkpoints/
runs/verify-e7-patch2c-0.5B-4gpu-s29-20260903-192605/checkpoints/
runs/verify-e7-patch3-0.5B-4gpu-s29-20260903-193233/checkpoints/iter_0000039/
runs/verify-e7-patch3-0.5B-4gpu-s29-20260903-193233/checkpoints/iter_0000049/
runs/formal-E7-clean-oracle-1.5B-4gpu-s29-/checkpoints/
```

第二轮约 41.1 GiB，删除：

```text
/public/home/caiyiwen/model/LLaDA-MoE-7B-A1B-Instruct/
/public/home/caiyiwen/model/LLaDA-MoE-7B-A1B-Instruct-fused/
```

未建立备份；这些目录不能再作为现存复现入口。删除只针对上述模型/checkpoint 和缓存，运行日志、meta、结果与源码保留。不要把这次失败 1.5B 路径清理扩大解释为“所有 seed 17 产物已清理”。

2026-09-10 核对保留：

- 8 个正式 0.5B `iter_0000499`、8 个对应 HF 目录、8 份 eval JSON。
- 8 个 `iter_0000449`、两个 s29 `iter_0000449_hf`。
- 5 个 `iter_0000399` 目录：s101 Oracle 和 s29/s42/s73/s101 RewardTxn。这里只验证目录存在，尤其 s42 的分片完整性应在转换前检查。
- `runs/verify-e7-patch3-0.5B-4gpu-s29-20260903-193233/checkpoints/iter_0000059/`。
- 较早失败 0.5B s29 Oracle（后缀 `20260903-165500`）的 iter 99/149 仍在，但不能混入最终成功矩阵。

## 资源与版本

最终核查磁盘约397 GiB可用、95%已用；新诊断checkpoint全部保留。没有新一轮清理，没有Git提交。工作区仍含既有未提交脚本、文档和运行资产，不进行reset或批量权限修改。

**文档版本**：8.1
