# RewardTxn 当前交接

更新：2026-09-20。本次按用户要求停止派发新实验，整理交接并推送；完整 FT-v1 实现/实验目标尚未完成，正式 GPU 样本数 **0**。

## 接手入口与当前状态

- 工作目录：`/public/home/caiyiwen/rewardtxn`；远程：`https://github.com/yiwen-cai/rewardtxn.git`；交接分支：`codex/ft-handoff-20260920`。
- 总体状态：[EXECUTION_STATUS](docs/experiments/rewardtxn-ft-20260916/EXECUTION_STATUS.md)；原范围：[FT-v1方案](RewardTxn%20已发表方法容错优势补充实验方案.md)；执行门槛：[SCRIPT_REPAIR_PLAN](docs/experiments/rewardtxn-ft-20260916/SCRIPT_REPAIR_PLAN.md)。不要用 CPU fixture、P0 或手动重启代替完整恢复验收。
- 当前无本任务运行容器；最后 batch-bridge-r1 容器已独立确认移除。双故障探针只完成源码/语法检查，**尚未运行隔离验收**。交接时不应重启任何旧 GPU job。
- economy-dev 在本根会话开启，root=`01a0bd89-a9c3-7d00-b16d-4a0a766de676`；用户选择 Astra medium。独立新会话不自动继承技能开关。优先读项目与第三方适用 AGENTS。

## 已验证结果（分层，不能相互替代）

| 单元 | 当前证据 | 限制 |
|---|---|---|
| P0 | 官方 CPU 62项；单H100同步完整状态保存/新进程加载/下一步一致 | 不证明完整RL pending/异步/多rank恢复 |
| 原4卡pilot | 3步RLVR保存；手动新进程RecoverHandler加载后下一步成功 | 不是故障自动重启；有预取pending缺口 |
| native trainer r4 | 完整step0后第二次optimizer完成，精确单trainer SIGKILL；收尾通过 | 未进入原生retry，方法timeout。900秒为总run窗口，故障后观察610.408秒；不是恢复RTO |
| P2 | 同版本408 CPU合同通过；96次真实SIGKILL、51次owner竞争；独立审计408全解析、384核心签名 | 8组F2实际状态等价，仅规范marker区分；另8限制、184待补；不是600完成 |
| controller-probe-r1 | 两种真实控制器崩溃窗口、新进程拒绝重复注入、正常对照及拒绝负例；7项通过 | CPU控制协议；训练/oracle not_evaluated，184待补不变 |
| P3 draw | 9机制测试及1真实loader测试通过 | 无consumed authority，恢复时全部durable draw重投 |
| P3评分缓存 | r2 11通过/1测试路径错误；仅改测试后r3定向1项通过 | 支持合法0/1缓存；内部fallback零分仍不可辨识；没有一次12项全绿重跑 |
| P3 batch-bridge-r1 | 1真实库集成测试通过，30.849秒；真实异步executor/Grouped/PPO/split到train_batch入口；8组/64样本，发布32样本intent | 生成/前向是合成fixture；状态prepared_not_applied；没有optimizer、consumed、commit或GPU |

native r4失败原因的源码与现场证据见 [P1_NATIVE_TIMEOUT_DIAGNOSIS](docs/experiments/rewardtxn-ft-20260916/P1_NATIVE_TIMEOUT_DIAGNOSIS.md)：torchrun已退出，孤儿进程仍持tee管道写端，shell job未结束，原生retry未触发。不能删失败、改N/A或由controller替方法重启；也不要求所有baseline恢复成功才记录有效实验结果。当前原生A+R故障闭环仍未验证。

## 最优先接续

1. 验收新双故障CPU探针：`tests/ft/p2_second_fault_probe.py`、`tests/ft/test_p2_second_fault_probe.py`。先用 `--make-config /tmp/rtx-second-fault-next.json` 生成配置，再通过原 `scripts.ft.container_run` 的CPU profile运行；不要宿主直接运行guardian/worker。资源2CPU/2GiB/128PID，总run150秒，无GPU/网络。运行后用 `--evidence <新run目录>` 验证6项并独立确认容器删除。不得覆盖历史run目录。
2. 继续P3真实训练适配：批次身份桥已到train_batch入口，下一步必须绑定实际optimizer/scheduler、异步staging/finalize、不可变完整generation与commit，再实现基于retained chain的consumed loader overlay和跨epoch replay/adoption；不能把`cpu_contract`改名当GPU fencing。
3. i06清单操作顺序与实际业务切点不一致，i07第二目标/多fault oracle还未映射。见 [P2_CONTROLLER_PENDING_PLAN](docs/experiments/rewardtxn-ft-20260916/P2_CONTROLLER_PENDING_PLAN.md)。探针通过不自动减少184项；修订manifest前归档旧版本并记录语义变化。
4. C/ByteCheckpoint完整适配、B官方artifact条件分支、正式freeze/功能预检/配对矩阵/独立统计仍未完成。保留原FT-v1范围，不补准确率/正常吞吐实验，不改async为sync冒充正式通过。

固定已有镜像：`sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469`；容器Python `/opt/.venv/bin/python`；宿主验证器 `.venv-tq/bin/python3.11`。新机器需要恢复对应依赖与模型，仓库不包含镜像、venv或权重。AReaL固定 `b83d1f40196e5bd7d9f83092563443561870d550`，ByteCheckpoint固定 `6f00167c153f3e65a67240aaa5ef4850a1c740fd`；两者交接前保持干净。Slime已有本地修改必须保留。

## 待用户决定

正式评分合同尚待答复：保留官方函数返回语义并披露内部异常/超时也可能返回0的限制，还是增加两臂共用状态接口并重新审查/冻结评分器。当前CPU合同仅证明`official_call_returned`，不能称内部验证成功。不要替用户选择或提前冻结正式评分口径。

## Git 与证据边界

本次提交包含FT源码、测试、计划/报告和小型验证摘要。约33.48GiB原始证据/检查点保留在本机 `docs/experiments/rewardtxn-ft-20260916/*_evidence/`；模型、第三方checkout、虚拟环境与历史无关未提交实验也保留本机。见 `LOCAL_EVIDENCE_INDEX.json` 获取证据目录及摘要引用。摘要不是原始证据替代品；依赖历史实存样本的审计测试必须先恢复这些目录，不能在纯clone中把缺材料写成通过。既有tracked仓库历史随分支完整推送，本次不将无关历史untracked实验混入提交。

---

## 先前交接记录（历史快照，以以上状态为准）


更新：2026-09-20

**最新限定进展：** P1 native trainer 探针实际命中step0保存后的第二次更新，但900秒内未原生重试，方法超时；原生shell/tee管道仍由孤儿子进程持有的证据已保存，隔离容器和网络已清理。见 `docs/experiments/rewardtxn-ft-20260916/P1_NATIVE_TRAINER_REPORT.md`，不能称恢复成功。P2同版本408格CPU合同通过（96次真实SIGKILL、51次owner竞争），独立审计408格全解析/384核心签名，8组F2 marker等价；8限制与184待补仍保留。P3a draw机制9项和真实loader1项通过，仅证明持久draw重投，无训练consumed authority。

**早期RLVR探针已完成限定验证。** 真实4卡3步保存＋新进程原生RecoverHandler加载后下一步成功，最终完整文件hash一致；预取游标28对训练12组的pending缺口已实测。见 `docs/experiments/rewardtxn-ft-20260916/PILOT_REPORT.md`。P2 state 21项、独立oracle 25项CPU合同通过（含旧祖先load伪保留链修复），真实评分器CPU入口及版本拒绝检查通过；真实故障工程探针结果见上；正式矩阵尚未启动。

**P1控制器已部分实现。** 直接子进程的pidfd注入、屏障释放/中止、失败保留和后续schedule经过11项隔离容器CPU测试；证据见 `docs/experiments/rewardtxn-ft-20260916/p1_evidence/`。新增多层worker内核凭据/pidfd合同8项及独立容器生命周期4项通过，7个测试容器移除已核验；官方AsyncRewardWrapper真实CPU评分池被杀后的原生重建重试已通过；真实训练角色/epoch和native launcher恢复接入仍未完成，不能据此启动正式矩阵。当前继续核实早期RLVR恢复与F4可达性，完整目标见 `EXECUTION_STATUS.md`。

**P0接口审计已交付。** 62项官方CPU测试通过，单H100同步Megatron checkpoint在新进程加载后的model/optimizer/scheduler/RNG及下一步状态一致。见 [P0报告](docs/experiments/rewardtxn-ft-20260916/P0_REPORT.md)。完整RecoverHandler/RL恢复、F4切点及A+R适配仍开放；可推进P1隔离runner/CPU合同，不能启动正式矩阵。以下FT0总体待办仍保留，其中接口审计与上述限定测试已由本次完成。

**当前主线：补充RewardTxn相对已发表、可获取实现的方法的容错优势证据。** 已交付[容错优势补充实验方案](RewardTxn%20已发表方法容错优势补充实验方案.md)（FT-v1）；新适配器与对照实验尚未实现、未启动。

**用户决定：准确率验证目前视为通过，性能无明显回退，短期内不规划重做。** 不再安排clean R/O增seed、LITE确认重跑、O/R×异步/同步、正常吞吐或5%公共观测开销复测，也不将它们作为新容错实验的前置。该决定更新项目验收与执行范围；历史测量、统计区间及失败门禁记录保留，不改写为新的统计通过证明。

下一步按新方案执行FT0来源/能力审计与适配设计，再完成FT1功能预检：

- 主对照为官方AReaL启用RecoverHandler、完整状态checkpoint和原生组保护，对比同版本AReaL＋完整RewardTxn；目标是同一系统上的增量容错收益。
- 补充官方ByteCheckpoint集成对照；它是checkpoint组件，不能代表完整RobustRL。RobustRL（OSDI 2026）另列强容错分支，官方artifact目前未确认可获取，缺失时不宣称已完成直接比较。B4/B5不能冒充论文官方复现。
- 核验Selective Replay真正回到训练消费、提交与完整训练状态的绑定、故障命中及独立oracle；同等正确恢复下比较真实RTO、重复计算及GPU成本。不新增终点准确率评估，不重开正常性能验收。
- A主矩阵预定70对/140次30步故障run；连同预检、ByteCheckpoint分支及重复故障检查，可获取分支最多210次、暂估209–352 GPU·小时。RobustRL条件分支另计。当前只形成方案，资源未预留。

已完成状态更正：最后三单元R163、R179、LITE179补跑均技术完成，validation100为76/71/74；全部12对LITE−R平均−0.83pp，95%区间[−4.99,+3.32]，仅作跨版本探索性结果。见[检查报告](docs/experiments/rewardtxn-last-three-review-20260916/REPORT.md)。不再按旧文档安排“补最后三轮”或全24轮重跑。

已有容错证据包括混版本修正、checkpoint后真实kill-9恢复和30k分层trace；尚未完成与已发表专用容错系统的同条件比较。旧准确率排查、SQLite单seed研究和[异步/同步方案](docs/experiments/rewardtxn-async-sync-plan-20260914/PLAN.md)作为历史背景保留，当前优先级以本文和新FT方案为准。旧实验的全部成功/失败产物保留。[此前证据汇总](docs/experiments/rewardtxn-reproducibility-20260914/REPORT.md)。


## 执行快照 2026-09-16 09:54
- FT0 启动：`docs/experiments/rewardtxn-ft-20260916/FT0_STATUS.md`；AReaL `b83d1f4`；ByteCheckpoint `6f00167` 已 checkout，未做运行时验收。
- 空间：已清中间 checkpoint（记录 `runs/SPACE_CLEANUP_20260916-ft0.json`）；`/public` 约 596G 空；根盘约 1.1T 空；`v4-0912-c` 约 103G（保留终点）。
- 资源：GPU0/1/2 被 wutong ray 占用；GPU7 xiaoxunpeng；**未启 FT1/FT2 GPU**。


## 门禁放宽 2026-09-16 15:52
用户决定：启跑不再锁定 device=1,2,3,4，**任意 ≥4 张空闲 GPU** 即可；选用当时空闲卡并写入冻结/日志。routine h100-gpu 已同步。探索性 LAST_THREE 禁止重复启跑。FT1/FT2 正式矩阵仍须 FT0（官方 recover 短例、A+R/C+R、新 freeze）就绪后才启；资源齐而工程未齐时只推进 FT0，不盲启正式故障矩阵。
