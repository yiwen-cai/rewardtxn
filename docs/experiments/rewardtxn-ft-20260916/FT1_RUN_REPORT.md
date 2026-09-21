# FT1 执行报告（等待 GPU）

2026-09-21更新；精确资源采集时间见ft1-gpu-block-snapshot.json。已完成7个GPU run：4个无故障、2个F4、1个F2 A+R；当前外部任务占用GPU，后续运行受阻。完整 FT1 尚未通过，FT2 未放行，正式样本仍为 0。

## 已完成：seed 401，A → A+R

| 验收 | A | A+R |
|---|---:|---:|
| 成功 optimizer 更新 | 10 | 10 |
| 真实训练输入独立重评分 | 320/320 | 320/320 |
| 完整 K=8 训练组 | 40 | 40 |
| 新进程 native model / optimizer / scheduler / RNG 加载 | 全部精确相等 | 全部精确相等 |
| 独立 checkpoint 内容哈希 | 6,917,854,855 bytes | 69,180,862,022 bytes / 10 代 |
| 实际数据游标恢复 | native dataloader round-trip | DrawLoader 自有 WAL 副本恢复 |
| 训练 run wall seconds | 433.514 | 1,915.896 |
| 训练 run allocated GPU·h | 0.481683 | 2.128773 |

A 最终 checkpoint 对应 step 9；R 的 10 代父链、intent/manifest/token 与实际训练输入一致，320 唯一样本，完整消费前缀和 pending 差集成立。两臂 RecoverInfo.next 为 step 10。以上核验只覆盖本次单 training rank＋3 推理实例、单 epoch 无故障运行，不外推多 rank 或性能收益。A+R 每代校验已保留历史 checkpoint，I/O 随历史增长；成本均保留，不在运行中优化或改配置。

主证据在 `p3_evidence/ft1-smoke-s401-{a,r}-r1`；独立重评分为 `ft1-input-s401-a-r2` / `ft1-input-s401-r-r1`；新进程加载为 `ft1-load-s401-a-r1` / `ft1-load-s401-r-r2`。首次 A 输入审计因 checker 未处理 PPO loss_mask 左移而失败，保留 `ft1-input-s401-a-r1`；修正与实际 actor.py 规则一致后重验。R load-r1 被 GPU 4 外部占用的空闲门禁拦在创建容器之前，后改空闲 GPU 3，未干预外部进程。

## 已完成：seed 409，A+R → A

两臂均 10 次成功更新、320 条真实输入独立重评分一致、40 个完整组，新进程 model/optimizer/scheduler/RNG 全部精确相等且 next step=10。A 最终 checkpoint 的 6,917,854,920 bytes 和 R 十代 69,180,862,859 bytes 独立重哈希通过；实际消费与保留链对应，无重复消费。A wall=391.100s / 0.434555 GPU·h；R wall=1,818.516s / 2.020573 GPU·h。两臂训练容器、网络及后续 CPU/单 GPU 加载容器均已清理。

4 个无故障 run 合计 40 更新、1,280 条重评分输入。逐项机器摘要见 `ft1-progress.json`；新证据为 `ft1-smoke-s409-{a,r}-r1`、`ft1-input-s409-{a,r}-r1`、`ft1-load-s409-{a,r}-r1`。

## 正在执行及故障准备

- 两对无故障已结束，随后安装故障 hooks。每对同四 GPU UUID；每臂启动前重新核实空闲，配对中途不改源码。
- F4/F2 公共故障 hooks 的 CPU 六项合同通过：实际评分 child SIGKILL＋原生同输入重试；未全生成、重复样本、跨组、时序竞争不注入；F2 optimizer 为 CPU stub，仅验证成功更新计数与一次性控制，不冒充 GPU 优化器/原生 launcher 验证。证据 `ft1-fault-contracts-r3`。
- F1 12 项 CPU scheduler 合同及 trainer marker 测试通过；真实 6,373 prompt 完整 tokenization 唯一性通过；只支持组级 worker 映射。用户已批准公共 scheduler hook，已在两对无故障结束后安装并冻结；安装后 18 项隔离 CPU 合同＋6 项入口/config/状态观察检查通过。详见 `FT1_F1_PATCH_REVIEW.md`。
- F4 seed417 A→R 已启动；后续为 F2 seed419 R→A、F1 seed421 A→R，均10步、共同 async DCP、同一原生 launcher retry1；属于 pilot，不进入正式 seeds 或分母。冻结见 `ft1-fault-freeze.json`，每 run 另存完整 AReaL Python 包与 hook/checker 源码副本、有效输入 hash。注入后 900 秒恢复判定窗与整 run 2400 秒分别记录，不混用。

## F4 进展

A 臂已有效命中，完成10次更新并通过独立验收：第四个实际评分 child 在全 K 生成后被 SIGKILL；strict retry 对相同 input SHA 成功；320 输入重评分、40组消费/训练映射、最终 checkpoint 全哈希通过；source5518已保留。原生 launcher 正常完成后也会消耗剩余 retry，随后新 trainer 真实加载 step9，model/optimizer/scheduler/RNG 与保存前和终态精确相等，无新增 optimizer 更新。这次完成后重启不计为 F4 引起的 trainer 故障恢复。证据 `ft1-f4-s417-a-r1/functional-verification.json`，加载交叉验证为 `ft1-native-load-f4-s417-a-r1/load-verification.json`。

A 的 237.583s 是注入到最终 checkpoint 观察的**上界**，不是首次持久提交的精确 RTO；`formal_rto_eligible=false`。A+R 臂也已完整验收为correct_recovered：10次更新、320条独立重评分、40完整组、十代checkpoint全哈希及320唯一样本消费链通过；新trainer实际加载step9的四类状态精确相等，零额外更新。目标5518进入第二代提交，故障到该提交观察为179.845s；首代不含目标，未被误用为恢复终点。R run wall=2322.353s / 2.580392 GPU·h，未触发2400s上限。两臂容器及网络全部清理。F2 seed419 R臂已结束，A臂随后被空闲门禁拦在创建容器之前。

## 尚未关闭的门禁

F4两臂和F2 A+R已关闭本轮验收；F2 A臂、F1两臂仍待验证。F3 沿用已审 v1 无 ACK 入口的预先 N/A；X1/X2 实际结果入口适用性尚未最终冻结，不以通知 callback 或内部状态篡改冒充结果故障。CPU 合同仍为408已验证、184待补、8限制，不用本次补充 CPU 测试抵扣。完整 freeze 与 functional_acceptance 仍未完成，不据无故障结果宣告整个 FT1 或正式实验就绪。

## F2 判读规则（运行前补充）

F4 A的真实时序显示：首次save返回时仅有common.pt；第二次optimizer成功在10:14:02.907594 UTC，而上一代DCP到10:14:19.811才在下一次save入口finalize。因此F2严格在第二次optimizer返回后、save前注入时，先前checkpoint可能尚不完整。保持原切点，不额外drain。原生恢复先恢复数据游标再加载模型，部分FileNotFoundError会被捕获；RecoverInfo读取成功不能代替模型load成功。

分类依据实际结果：明确恢复失败、全部原生生命周期已结束且无后续持久更新才可记safe_stop；缺加载或保留链关联为unverifiable；只有最终持久保存与独立证实的状态/消费不一致同时存在才记invalid_commit。运行超时单独保留，不被目标恢复成功覆盖。该补充解释既有独立oracle口径，不修改运行配置、切点或源码冻结。

## F2 A+R 已完成；A 臂等待 GPU

A+R 已准确命中第二次optimizer成功后、该次save前的SIGKILL，原生launcher自动重启。新trainer PID2225实际加载step0，独立观察的model/optimizer/lr_scheduler/RNG与故障前首代保存精确相等。恢复后第一批32条tokens/versions/mask/reward与被杀批次完全一致，重放结果在故障后271.460s进入step1提交。A+R已正常结束：11次物理更新、10代提交；352条物理输入独立重评分一致，十代69,180,862,148 bytes/120文件重哈希通过，最终320唯一样本/40组消费。未保存的32条输入重放没有重复计入保留链。wall=2173.767s / 2.415297 GPU·h。随后GPU3短暂空闲，已完成独立新进程step9加载：model/optimizer/lr_scheduler/RNG与保存前及终态均精确相等，DrawLoader WAL副本恢复sequence14、next step10；现完整验收correct_recovered。证据为ft1-f2-s419-r-r1/functional-verification.json、f2-restore-verification.json及ft1-load-f2-s419-r-r1/load-verification.json。此前interim-verification.json保留为历史中间状态。

## 当前资源阻塞及续跑点

F2 R退出并清理后，A-r1的启动前空闲检查发现GPU2/3已由外部进程占用；随后现场查询GPU0/1也被外部任务使用。A-r1只有gpu-idle.json和preflight-block.json，没有创建容器、训练或注入；不算方法失败，也不算一次GPU故障样本。已完成的不需GPU的F2 R审计，停止等待未启动A臂的本任务watcher；没有干预外部进程。

单卡F2 R最终加载已经补齐。最新现场仅GPU3空闲，GPU0/1/2仍为外部任务；同四张冻结GPU全部空闲后，在新证据目录续跑F2 A seed419，再执行F1 seed421 A→R。保留A-r1启动阻塞记录，不能覆盖；不因本次资源阻塞重跑已完成R臂。源文件冻结保持不变。F1公共hook已获批准且安装/CPU测试通过，真实生成worker SIGKILL仍未执行。
