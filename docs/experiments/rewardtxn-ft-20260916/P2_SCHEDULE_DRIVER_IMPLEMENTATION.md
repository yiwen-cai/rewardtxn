# P2固定12格CPU驱动

日期：2026-09-20。新增 `tests/ft/p2_schedule_driver.py`、`tests/ft/test_p2_schedule_driver.py`。不修改600清单、生产state/oracle、P1或第三方；没有GPU或真实训练。驱动不解释任意manifest操作、不批跑408格，只显式实现 [P2_SCHEDULE_PLAN](P2_SCHEDULE_PLAN.md) 第8节的12个ID。

## 使用与分账

使用Python3.11的现有解释器，直接脚本避免环境中其他tests包冲突：

```sh
.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py \
  --case F2.b09.i05 --output /absolute/new-single-case-run

.venv-tq/bin/python3.11 tests/ft/p2_schedule_driver.py \
  --representative --output /absolute/new-twelve-case-run \
  --case-timeout 30 --total-timeout 240

.venv-tq/bin/python3.11 -m unittest discover -s tests/ft \
  -p 'test_p2_schedule_driver.py' -v
```

`--case`可重复，但不可重复同一ID；`--representative`与它二选一。每次总输出目录必须新建，各case目录及result均不覆盖。选择清单内其他ID会产生明确not_executed，不创建state-run/子进程；接口pending格不能用这个入口计通过。选择清单外ID直接拒绝。默认单case30秒、总240秒；worker协议读与进程等待受截止时间约束，固定小型本地oracle计算前后也检查截止时间。超时是harness_failure，不作方法RTO。清理仅遍历本次Popen持有的直接子进程，不按名称/PGID杀共享进程，不使用P1登记接口。

顶层inventory记录静态600分类、请求ID、当前支持列表、源码/manifest全hash、Python版本、文件系统及deadline。coverage区分requested/executed/passed/failed/harness_failure/not_executed，并列各线实际启动、SIGKILL与owner竞争数量。case `passed` 是**所声明CPU合同测试通过**，不是方法correct_recovered，更不是600通过；184待补接口、8限制验证的静态账本保持原样。驱动没有将未执行格转成通过。

每case保留：冻结case.json、各worker的操作JSONL/stderr、parent-receipts、完整launch argv及退出状态、真实state-run、方法fixture输入、独立oracle原始图/spec/report、result。失败不清空这些文件；只清理仍活的自有进程。result的must_reach引用实际事件序号/文件/进程身份；任何必达项没有观测即不能通过。

## S/P真实执行

S使用当前7个真实state API。所有生成样本均有实际CPU JSON文件；每组K=8，组件文件每rank各11类、每文件32KiB。两个rank是**规范中的组件集合**，由CPU writer实际写文件，不是两个Megatron rank。optimizer start/end为明确标注cpu_fixture的驱动事件，没有假装调用真实优化器。state负责检查消费/授权/收据及文件持久合同；独立oracle负责reward语义反例。

每个case建立真实已提交bootstrap父代，目标只消费自己逻辑组；probe接受后显式记录pending，目标生成若实际prepare则在data snapshot中包含probe待重新生成义务。缺样本不编造成功receipt，wrong verifier真正送到accept_result并检查指定拒绝原因。状态拒绝必须命中具体语义检查，不接受任意StateError充通过。

| case | 实际S/P执行及边界 |
|---|---|
| F1.b00.i01 | bootstrap样本逆序生成；目标只生成/接受1条，真实prepare拒绝不完整K |
| F1.b09.i04 | probe新授权后旧结果抢先拒绝；目标生成8条、只接受7条，预期清单发现从未产生reward的第8项，prepare拒绝 |
| F2.b04.i00 | 真实intent持久化，记录CPU optimizer完成但不写成功收据；精确SIGKILL，新进程回父代且保留rollback_intent |
| F2.b09.i05 | 两真实进程争owner锁，winner保持锁等loser退出，再在同run继续；全组件/finalize完成，manifest持久后、token link前SIGKILL；新进程充分证据补提交 |
| F3.b00.i00 | 目标只保存actor:0集合，commit真实缺rank拒绝后SIGKILL；新进程回父并保留未提交义务。只验证存储切点，无ACK |
| F3.b01.i09 | 真实token hard-link后、目录fsync前SIGKILL；新进程验证完整可见token；先独立无损O对照，再中段修改父checkpoint，另一新进程select拒绝损坏 |
| F4.b00.i00 | 写8份真实CPU回答，记录4个**规范评分start事件**及第4个in-flight边界；不声称真实评分并发、GPU自然可达或真实故障命中 |
| F4.b06.i03 | probe先接new后拒绝old；目标第8项声明v2，真实accept拒绝混verifier |
| X1.b01.i00 | 真正计算exact=1及1−exact=0，将v2/0送入真实accept被拒；之后prepare仍拒绝缺项 |
| X1.b09.i08 | 合法版本但错误reward=0真实进入accept/prepare，证明state不代替语义评分；独立O查出invalid_commit，再删除bootstrap optimizer_start验证明确缺证据 |
| X2.b02.i04 | probe与目标分别执行新授权后旧结果先到；真实拒绝old后接收new；O另给错误保留old的独立负例 |
| X2.b09.i05 | 两真实owner竞争，winner继续同run；关闭该owner后调用commit真实拒绝。这里验证失效owner对象，不冒称已经覆盖另一GPU owner接管后的所有执行者fencing |

owner竞争通过两条独立管道ready/start放行；winner在继续目标前保持owner锁，loser必须实际拒绝并退出。不能两个进程先后独立成功后宣称互斥通过。winner身份、bootstrap和目标事件一致。

SIGKILL切点只在测试子进程中窄包装现有state函数：manifest切点为原 `_write` 完成后，token切点为真实 `os.link` 返回后。子进程发精确cut收据并阻塞；父进程核对其真实PID/start-time/boot-id后，通过所持Popen精确kill并确认-9，再启动独立恢复子进程。恢复由真实acquire_owner检查旧登记进程退出；不传布尔退出证明。进程崩溃后token可见不是断电持久性试验。

## O独立规范图

复用当前 `test_oracle.py` 的手写Graph构造，显式扩为K=8、两rank组件覆盖、bootstrap与target两个真实规范更新；不读取state产物生成oracle图或expected。每个actual training row包含tokens/mask/logprobs/reward及payload hash。图中的时间为固定模拟controller时间，不能当本次CPU耗时或方法RTO；S/P返回也不会决定O应通过还是拒绝。

两线在启动前各自读取同一个 `external-fixture.json`：bootstrap保留manifest的 `Return one.`，target保留 `Return the integer one.` 聊天列表，绝不统一替换。manifest未展开的bootstrap/probe及未生成样本，明确冻结CPU展开值tokens=[11,21+i]、mask=[0,1]、logprobs=[0,-0.25]；这些是预定输入，不是已经观测到的生成。O不读取S/state产物。external输入与dataset均进入oracle冻结inputs全hash。父进程逐份核对实际S payload及收据hash与冻结输入相同；i01从winner实际生成收据提取bootstrap顺序，必须为7到0。

所有格先跑独立完整健康对照。每格primary和counterexample分别保留，primary遵循清单语义：

| 范围 | O primary实际图 | O负例 |
|---|---|---|
| 两个F1 | 最终只保留bootstrap，目标无optimizer更新；work_dropped及safe_stop，必须safely_dropped/safety pass | 保留不完整1或7条K组，必须invalid_commit |
| 两个F2、F3.b00 | physical-target实际完成后被丢弃，checkpoint_loaded加载bootstrap进入epoch1，再physical-recomputed；最终保留重算且rolled_back_updates恰为physical-target | 强行把丢失的physical-target加入最终状态，必须invalid_commit |
| F3.b01 | target已持久；故障后实际加载target，再追加独立其他组更新，保留祖先义务 | 破坏保留组K，必须invalid_commit |
| F4.b00 | normative生成8/评分启动4/无结束事件，之后完整目标更新，仅correct_recovered_model范围 | 不完整保留K，必须invalid_commit |
| F4.b06、X1、X2 | 分别错误保留未授权verifier、伪reward、旧attempt或旧epoch | primary本身即独立invalid_commit反例，另有完整健康对照 |

各图冻结故障角色、目标及case边界，generation/reward进度为明确模拟事件；它们不验证真实自然到达。F2.b09的S验证充分证据补提交，而O依清单另验证合法回滚重算；这是不同的两个规范场景，不声称O从真实S恢复过程派生。F3无ACK恢复证明。

i08先审计包含错误reward的完整反例图（必须invalid_commit且独立reward不符），再删除bootstrap optimizer_start并重新封装观察证据，必须unverifiable且缺口精确指向该事件。i09先健康图通过，再修改一份>16KiB独立observer组件的中间字节、保持旧seal，必须fullhash mismatch；方法checkpoint中段篡改另经真实state验证。不同probe分别保留报告，不把缺证据报告当“方法安全通过”。

目前不存在通用A-model仿真器、真实RLVR进度采集、ACK、P1注入journal、多故障聚合或实际GPU奖励变换。F4评分start/in-flight只能作为规范模型标注。case通过不能覆盖这些缺口，静态清单也未为本次实现修改。

## 本轮必要验证（分修订记录）

以下两项为共享输入与O分场景修正前的历史定向结果，不能代替当前修订验收：

- 实执行 `F2.b09.i05` 通过，输出保留 `/tmp/rtx-p2-driver-target-r1`。
- 实执行 `F3.b01.i09` 通过，输出保留 `/tmp/rtx-p2-driver-target-r2`。
- 两项必要driver测试通过：不完整K真实API拒绝收据；错误reward反例与删除证据分开检查（0.230秒）。

上述执行是少量实现定向检查；最终完整8项driver测试及12格代表验收交主任务运行，不提前标全通过。没有运行408或600批次，没有启动GPU、AReaL或P1服务。

当前修订定向检查：F1真实API拒绝+实际逆序+共享输入检查通过（0.276秒）；F1 safe-drop/F2真实load回滚重算模型测试通过（0.072秒）；F3.b01 token-load模型及负例、fullhash篡改probe通过。未重跑完整12格，由主任务验收。

## 主任务验收 r1（当前有效凭证）

主任务完整验证：driver **8项通过，1.161秒**；`representative-r1` **12/12 passed**，failed=0、harness_failure=0；实际确认 **4次SIGKILL、2组owner争锁**，全部must_reach具有实证。汇总及源码hash见 [driver-verification-r1.json](p2_evidence/driver-verification-r1.json)，测试日志见 [driver-tests-r1.log](p2_evidence/driver-tests-r1.log)，逐格结果见 [representative-r1](p2_evidence/representative-r1)。这是主任务执行结果，本轮文档更新没有重跑。

全局账本仍为600：**12 passed、396静态可执行但未跑、8限制验证、184待补接口**。本次requested的not_executed=0只描述请求的12格，不代表600都已执行。`p2_evidence/implementation-*` 是先前/tmp定向结果的归档，旧O图不足，仅为开发历史，不作最终验收凭证。后续扩展见 [P2_SCHEDULE_EXPANSION_PLAN](P2_SCHEDULE_EXPANSION_PLAN.md)。

## 后续首段扩展（待主任务完整验收）

当前driver已按 [首段实现说明](P2_SCHEDULE_STAGE_ONE_IMPLEMENTATION.md) 增加F1全80格与X1.b00的8格；与旧12并集98格，`--representative`仍固定旧12。上文“仅12格”的范围描述对应r1历史修订。v1清单完整归档，v2只改变X1.b00的10个清单项；全局分类及184接口缺口不变。本轮仅定向检查，不能把新支持数量当完成数量。

## 第二段storage扩展（待主任务验收）

首段已由主任务验收98个唯一case。当前进一步支持F2全部80格与F3可执行16格，支持并集190；切点处由父进程在SIGKILL前实读文件、收据及进度，再核对新进程恢复义务。详见 [第二段实现说明](P2_SCHEDULE_STAGE_TWO_IMPLEMENTATION.md)。新增支持不提前计通过，原manifest/静态分类未变。

## 第三段X1扩展（待主任务验收）

第二段已由主任务验收190个唯一case。当前进一步支持X1可执行72格，支持并集252；包含实际错reward来源、声明版本、原文/seal及未授权policy负例，并新增同组合法policy差异正例。详见 [第三段实现说明](P2_SCHEDULE_STAGE_THREE_IMPLEMENTATION.md)。b05及其他184接口缺口未扩展，支持不提前计通过。
