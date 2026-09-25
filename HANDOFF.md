<!-- CURRENT_SNAPSHOT_START -->
# RewardTxn HANDOFF — 当前快照

> **2026-09-25 更新（覆盖下方 09-23 快照的"正式矩阵样本数 0"等表述）**：已转入最小方案（每 run 10 次更新；30 步 R 变慢问题本身未在此验证），正式冻结 2026-09-24 13:14 后 13/13 对全部 `formal_pair_verified`：F2 10 对 R 全部 `correct_recovered`（32/32）、A 全部 `safe_discard`（0/32）；无故障 3 对均通过；p02 一次技术无效已按规则重做。汇总与数字见 [PROGRESS.md](PROGRESS.md) 顶部。结果报告见 `docs/experiments/rewardtxn-ft-20260916/FORMAL_RESULTS_20260925.md`。下一步待用户选择：F1/F4 扩展或 R 开销优化。

> 2026-09-24 更新：用户要求删除旧 FT1 原始试跑数据并以最小空间执行新方案。五个旧 `*_evidence/` 目录中的未跟踪原始文件已清理，仅保留已跟踪的小型摘要；见 [清理记录](docs/experiments/rewardtxn-ft-20260916/FT1_RAW_EVIDENCE_CLEANUP_20260924.json)。旧报告与判定仍是历史记录，原检查点已无法在本机重新加载复验。新执行范围及逐 run 存储规则见 [最小方案](docs/experiments/rewardtxn-ft-20260916/FT_MINIMAL_AREAL_PROPOSAL_20260923.md)；下列 2026-09-23 快照中的旧 FT1 原始证据保留要求不再适用，其他历史判定不变。

- **时间**：2026-09-23 晚（本机 H100，`/public/home/caiyiwen/rewardtxn`）
- **分支**：`handoff/20260923-ft1`，HEAD `9884bbf`；**本日全部改动尚未提交**（见文末"Git 状态"）
- **正式矩阵样本数：0**。以下都是工程试跑或 pilot（`formal_sample=false`），不得宣称正式通过。

## 一句话现状
FT1 配对审计已完成，F1 s421 为工程 Go。随后发现 **R 臂每步线性变慢**，30 步 run 会超过 FT-v1 §6 的 45 分钟上限。第一轮修复（v2）已实现并通过 CPU 验证，但 **30 步 F2 门控不通过**：R 用了 2739.5 s，超上限 40 s，残余增长 1.57 s/步。按规则已停下，等用户决定第二轮修复。

## 今日完成
1. **R-r2 验收补齐**：F1 s421 R-r2 的 load 重跑通过（上次失败是 `draw-copy` 目录残留导致的工具缺陷），FV = `correct_recovered`。记录：`p3_evidence/ft1-f1-s421-r-r2/acceptance-load-rerun-20260923-090847.json`。原 `acceptance-status.json` 仍写着 `blocked_at_load`，这是历史记录，**以重跑记录为准**。
2. **验收脚本可重入**：`tests/ft/run_ft1_acceptance.py` 在输出目录已存在时自动改用 `-rerun-<时间戳>` 新目录；每个验收容器 900 s 超时 kill。
3. **F1 s421 配对审计**：工程 Go，但不是同一冻结版本的配对（A-r1 的 observer 是修复前版本），不能进正式矩阵。报告：`docs/experiments/rewardtxn-ft-20260916/ft1_f1_s421_pair_audit_20260923.json`。
4. **R 变慢根因**：`state._head()` 每次都对全部历史代的 checkpoint（每代 6.9 GB）重做全量 SHA-256，每步 3k+3 次，恢复路径同理。方案 v1 → 独立审计（有条件通过）→ v2：`R_SLOWDOWN_FIX_PLAN.md`、`R_SLOWDOWN_FIX_PLAN_AUDIT.md`、`R_SLOWDOWN_FIX_PLAN_v1.md`。
5. **用户批准并已实现**（记录：`R_SLOWDOWN_FIX_IMPLEMENTATION.md`）：
   - L1：历史代只做元数据加 stat 检查；恢复时只对要加载的 head 做内容哈希。
   - L2：commit 的第二次全量哈希改为完整 stat 身份比较。
   - L3：运行中剪掉链上 k−2 及更早代的 `native/*.distcp`；marker 内容确定、先写后删；pin 恢复加载代与恢复后首次提交。
   - C1a/C1b：R 专用并行 `native_snapshot_r`（与原函数输出逐字节一致），与 prehash 并发执行。
   - 活性修复：`prepare` 的 parent 改用 token 链派生的 head。
   - P2 `*.i09` 改为篡改 head；审计工具支持 marker、pin 和步数参数化。
   - FT-v1 方案追加"修订 R1"（§9 证据保留：验收通过的 `correct_recovered` run 删除大分片；失败 run 与固定抽样的 10% 配对全量保留；两臂同样适用）。
6. **CPU 验证**：
   - 宿主 state 系 39/39 通过；
   - 容器全部 `tests/ft` 与 HEAD 基线逐项对照一致；
   - P2 60 个 i09 用例前后都是 51 通过、9 个未执行；
   - 旧证据回归：`ft1-f1-s421-{r-r2,a-r1}` 与原判定一致。
7. **30 步 F2 门控**（1 对，seed 431，GPU 4–7，第 12 次更新后 kill trainer）：报告 `R_SLOWDOWN_GATE_REPORT.md`，判定 `r-slowdown-gate-check.json`。

| 判据 | 结果 |
|---|---|
| R 墙钟 ≤ 2700 s | ❌ 2739.5 s（在 2700 s 被判 timeout，30 次提交已完成，收尾未完成） |
| 每步斜率 < 1 s/步 | ❌ 1.57（修复前约 20） |
| 恢复只哈希 1 代 | ✅（8.0 s） |
| 两臂验收 | ❌ A `functional_verification_written`（`safe_discard`）；R `blocked_missing_final_native_state` |
| 磁盘峰值 ≤ 45 GiB | ✅ 34.7 GB |

A 臂墙钟 1149 s。

## 残余问题（下一步的核心）
R 每步逐段拆分：prepare→opt、opt→scheduled、scheduled→finalized、finalized→commit 都恒定；**只有 committed→下一 update_prepared 在涨（13.9 → 39 s）**，主体是 R 独有的 `batch_taken→train_batch`（7 → 28 s）。

最可能的来源（**未剖析确认**）：批次身份桥和 RetainedLoader 每步、每行对线性增长的 `control.json` 做 `_locked` 读和带 fsync 的整文件重写。相关位置：`batch_identity.py:68,80,114`、`training_replay.py:155`、`validate_rows` 的 `_current`、`authorize_attempt`/`accept_result`。

## 后续方案（按顺序，每步需要的用户决定已标出）
1. **【待用户批准】C1c 剖析**：CPU 上用真实大小的 `control.json`（30 步约 1.4 MB、约 1000 个 attempts）重放 bridge/loader 路径，给出每个调用点随 k 的耗时。不占 GPU。
2. **第二轮修复方案 → 独立审计 → 用户批准 → 实现**。候选方向：进程内缓存 control（锁内 mtime/ino 校验）、按批合并多行读写、attempts/accepted 拆分为追加式日志。不得改变提交与恢复语义，也不根据 RTO 挑选变体。
3. **【需再批 GPU 预算】重跑 30 步 F2 门控**（新 seed，判据不变）。FT1 预检上限 16 次已超出，本轮已用 2 次。
4. 门控通过后：
   - 写 `run_pair.py`（全自动配对：GPU 空闲检查 → 两臂 → 自动验收 → 分类 → 追加 `runs.csv`，含超时 kill 和按规则补一次）和 `audit_pair.py`（把今天人工核对的 9 项脚本化）；
   - 生成**正式 freeze**（镜像、源码、配置、seed 表、配对 schedule、FT-v1 修订 R1、10% 抽样名单；补齐 `OVERLAY_PIN.json` 的 commit）。
5. 用户已同意的简化范围：本轮只做 **AReaL 分支 F4 20 对＋F1、F2 各 10 对，共 80 次 30 步 run**；C/B 分支、X1/X2、FT4 推迟，184 项 CPU 合同与正式实验解耦。主检验不变：F4 A+R 对 A 的 RTO 中位节省 ≥ 20% 且区间下界 > 0。
6. **统一两臂 RTO 口径**：现在 A 用"最终 checkpoint 上界"，R 用"commit 观测"，不可比。正式冻结前必须统一为"受影响工作首次正确持久提交"。

## 另立的已知问题
- 容器内 uid 1028 没有 passwd 条目：跑测试需设 `USER`/`LOGNAME`。
- `tests.ft` 包名被 `third_party/areal` 的 `tests` 遮蔽：子进程式 state 测试需在宿主跑。
- `check_training_fault.py`、`check_training_integration.py` 对应的旧证据（`training-*-gpu-r*`）本机不存在，只做了编译检查。
- 磁盘：`/public` 约 185–193 GB 可用（98%），`/` 约 377 GB。正式矩阵依赖修订 R1 的清理策略。

## 操作红线（不变）
- 不宣称正式通过，不宣称 R 恢复更快，不启动正式矩阵。
- 不覆盖、不删除任何既有证据（包括失败和 timeout 的目录）；重跑一律使用新目录。
- 开 GPU run 前确认用户已批准预算；每个 run 都要有超时 kill。
- 实现修改结束当前版本批次，修改前后的结果不合并。

## Git 状态
- 已修改（tracked）：`scripts/ft/{state,training_adapter,areal_ft1,ft1_fault_hooks}.py`，`tests/ft/{check_ft1_chain,check_ft1_fault,check_ft1_input_audit,check_ft1_load,check_training_fault,check_training_integration,p2_schedule_driver,run_training_fault}.py`，FT-v1 方案 md。
- 新增（untracked）：`tests/ft/{run_ft1_acceptance,offline_generation,run_ft1_gate,check_ft1_gate,test_state_efficiency,test_native_snapshot_r}.py`，`docs/experiments/rewardtxn-ft-20260916/R_SLOWDOWN_*.md`、`r-slowdown-gate-check.json`、`ft1_f1_s421_pair_audit_20260923.json`。
- 备份：被改文件旁的 `*.bak-20260923-100804`；`HANDOFF.md.bak-20260923`。
- 大证据目录（`p3_evidence/ft1-gate-*` 等）只留在本机，不入库。

<!-- CURRENT_SNAPSHOT_END -->

## 历史快照 2026-09-23 08:55（已被上方新快照取代）

- **时间戳**: 2026-09-23 08:55:00 +0800
- **权威工作树**: `/public/home/caiyiwen/rewardtxn`（对齐 GitHub `yiwen-cai/rewardtxn`）
- **基线 commit（写快照前 master）**: `41dc0bb`
- **本分支**: `handoff/20260923-ft1`
- **本分支 tip**: 见 push 后 `git rev-parse HEAD`（本修复提交会更新 tip）
- **禁止宣称正式通过**；正式矩阵样本数 **0**（`formal_sample=false` 仅工程试跑）

## FT1 工程状态（权威摘要）

| 项 | 状态 | 证据 / 备注 |
|---|---|---|
| **F2-A s419-r4** | **工程 Go** | `docs/experiments/rewardtxn-ft-20260916/p3_evidence/ft1-f2-s419-a-r4`：exit0；FV `safe_discard`；`formal_sample=false`；acceptance=`functional_verification_written` |
| **F1 s421 A-r1** | **工程 Go** | `.../p3_evidence/ft1-f1-s421-a-r1`：exit0；`f1-claimed`；FV `correct_recovered`；`formal_sample=false` |
| **F1 s421 R-r1** | **technical_invalid** | `.../p3_evidence/ft1-f1-s421-r-r1`：exit2；无 claim；acceptance=`fault_not_valid_hit`；**保留失败终态，不覆盖** |
| **F1 s421 R-r2** | **injection OK；acceptance 未完** | `.../p3_evidence/ft1-f1-s421-r-r2`：exit0；`f1-claimed` / descendant / signal 有；fault 单点已过（`valid_hit` / `correct_recovered` 中间产物）；acceptance 现为 `blocked_at_load`；**load/FV 全链 backfill 因 Mac↔H100 断连未完成**；同 seed R 技术补做额度已用，禁止再开 GPU 训练重跑除非新裁定 |
| **F1 s421 pair** | **仍 No-Go** | 过审前不宣称配对；不启 F2/其他场景 |
| **正式矩阵样本** | **0** | 上述均为工程试跑 |

## 代码侧已落盘（本 handoff 分支）

- R 臂 injection 竞态修复：`scripts/ft/ft1_scheduler_observer.py`（`run_batch` 前后 claim；允许 scheduler 内 sibling finished+unfinished 武装）。说明：`docs/experiments/rewardtxn-ft-20260916/F1R_INJECTION_RACE_FIX.md`
- 主机 Python 3.8 兼容：`tests/ft/check_ft1_fault.py`、`tests/ft/check_ft1_load.py`（去掉 `str.removeprefix`）
- 相关：`scripts/ft/ft1_fault_hooks.py`、`tests/ft/run_training_fault.py`（F2 async DCP 等工程补丁，未宣称正式冻结）

## 未决（下一步）

1. **acceptance backfill**：对 `ft1-f1-s421-r-r2` 只补验收链（fault→chain→input→load→finalize），**不动** `events.jsonl` / `exitcode`；写出完整 FV/load 后再交结果审计。
2. **pair 重审**：R-r2 验收过审后，由结果审计审 F1 s421 pair；过审前仍 No-Go。
3. **freeze hash**：正式矩阵前再锁脚本/配置哈希；当前禁止用工程 Go 冒充正式通过。

## 操作红线

- 不启训练 / 不开 GPU 训练重跑（除非科研助手新裁定）
- 不覆盖 A-r1 / R-r1 / R-r2 失败或既有终态证据
- 不宣称正式通过、不宣称 pair Go、不启 F2



---

# RewardTxn 当前交接

2026-09-21最新补充：用户接受首批单训练rank＋3推理实例，并批准公共评分状态补丁。现已应用并通过116项CPU测试（28严格评分/缓存/重放＋88原有相关测试），当前6373条标准答案解析/自比较检查全通过。A/A+R共用严格评分：正常0/1可复用，回答无表达式仍有效0；gold无解析结果为数据错误，异常/超时不转有效0；默认每wrapper单并发、15秒执行等待、最多2次尝试，超时独立进程回收。终态失败收齐同组任务并停止批次，不静默换样本；R envelope schema2/status=scored，verifier指纹已更新。见 [评分补丁报告](docs/experiments/rewardtxn-ft-20260916/SCORING_STATUS_PATCH_REPORT.md)。7个CPU容器清理完成，未占GPU。下文official_call_returned/异常零分不可辨识是旧版本状态；当前依赖公共评分执行修改，下一步仍需FT1配对预检及F4故障进程重新映射、正式冻结，不能沿用旧GPU结果作新版本正式验收。预检两次尝试尚非正式参数冻结，正式GPU样本仍0。

2026-09-21最新补充：checkpoint写入中故障自动恢复已通过CPU/GPU独立验收。真实DCP文件已写272,437,394字节、剩703个tensor时SIGKILL trainer；公共job层清理writer，原生自动重试，适配器依据绑定旧job的完整清理凭据解除pending阻塞，弃用半成品并精确加载step0。最终96唯一样本/3提交，未提交32样本全部重放；故障到下一optimizer178.49秒、下一commit244.40秒。容器4项CPU与宿主state21项通过；GPU仅一次尝试。详见 [写入中恢复报告](docs/experiments/rewardtxn-ft-20260916/P3_MIDWRITE_RECOVERY_REPORT.md)。容器/网络及4卡已释放。下文“pending总是拒绝接管/写入中未覆盖”是历史快照；当前只允许同boot/PID namespace、受信任专属subreaper的完整清理凭据接管，缺证据仍拒绝。多rank、连续GPU故障及正式配对矩阵仍待完成，正式样本0与184待补不变。

2026-09-21最新补充：用户批准公共launcher生命周期修订后，真实GPU自动恢复闭环已验证。单rank、第二optimizer成功/保存前唯一SIGKILL；原生local_main自动run0→run1，精确加载保留状态；4次物理更新、3次有效提交、96唯一样本，未提交32样本均重放消费。故障到下一optimizer173.20秒、下一commit233.61秒；6项CPU及独立GPU验收通过，容器/网络和4卡已释放。见 [自动恢复报告](docs/experiments/rewardtxn-ft-20260916/P3_AUTOMATIC_RECOVERY_REPORT.md)。旧未经修订部署r4超时保留；此结果不覆盖写入中故障或多rank，正式GPU样本0、184待补不变。

2026-09-21最新补充：真实训练接入工程验收已完成。3次真实optimizer/scheduler、3代异步完整checkpoint、96个唯一consumed；64个跨进程复用样本实际训练并提交，新进程完整状态精确加载通过。最终同版本CPU21项及宿主state/fork/exit23项通过（部分重叠）。见 [P3接入报告](docs/experiments/rewardtxn-ft-20260916/P3_TRAINING_INTEGRATION_REPORT.md)。修复评分fork继承mutation锁导致的阻塞，所有本轮容器/网络已清理。以下历史快照的“仅train_batch入口/尚无optimizer接入”已过时；单rank、同PID namespace正常保存/新进程加载的验收不能替代GPU故障自动恢复，P2的184项待补与正式GPU样本0不变。

2026-09-21补充：双故障CPU隔离探针已执行，独立6项验收全通过，无跳过；容器删除已独立确认。见 [验收报告](docs/experiments/rewardtxn-ft-20260916/P2_SECOND_FAULT_REPORT.md)。以下9月20日快照中“双故障尚未运行”及接续第1项已完成；184项待补、训练/oracle未验证与正式GPU样本0均不变。

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
