# FT 实验执行状态

> 2026-09-21 最新补充：实现已提交（24b7285、6719426）；FT1 无故障配对入口及 4 项 CPU 预检通过，GPU 预检尚未运行，现场无四张空闲卡。完整 FT1 仍有恢复链 oracle、各 cell 映射及冻结验收未完成，不能放行正式实验。详见 [FT1_PREFLIGHT_REPORT.md](FT1_PREFLIGHT_REPORT.md)。下文较早阶段状态保留历史上下文。

2026-09-21最新补充：公共严格评分补丁已经用户授权并应用；116项CPU测试和6373条标准答案检查通过，详见 [报告](SCORING_STATUS_PATCH_REPORT.md) 与 [验收摘要](scoring-status-verification-r3.json)。有效0/1与异常/超时/gold错误分离；两臂同一有界评分进程与最多2次尝试；R仅缓存schema2/scored有效结果；终态错误停止实际Grouped/dispatcher批次并回收兄弟任务，不补样本。7个CPU容器已清理，本轮无GPU。用户已选择单训练rank＋3推理实例；这及重试参数仍需正式freeze。历史异常fallback零分和评分池证据不能当新版本正式结论；下一步FT1成对故障预检、独立分类与正式冻结，F4需按新实际执行进程重新映射，正式矩阵0/184待补不变。

2026-09-21最新补充：P3单rank checkpoint写入中SIGKILL自动恢复验收通过，见 [报告](P3_MIDWRITE_RECOVERY_REPORT.md) 与 [摘要](training-midwrite-verification-r1.json)。原生writer已有272,437,394字节/余703tensor，trainer被杀后writer被公共job清理；绑定旧owner/job的清理凭据允许同namespace pending接管，半成品弃用，保留完整状态精确恢复。96唯一样本/3提交、32未提交样本重放；到下一optimizer178.49秒、commit244.40秒。CPU容器4项和宿主21项通过，唯一GPU尝试及独立文件/消费/进程/资源核验通过。容器、网络、4卡已清理释放。下文未覆盖写入中的描述为历史状态；多rank与连续GPU故障仍未覆盖，正式矩阵0/184待补不变。

2026-09-21最新补充：公共生命周期修订已经用户明确批准并应用，P3单rank post-optimizer/pre-save唯一SIGKILL自动恢复验收通过。原生重试1次、精确加载、4次物理optimizer/3次有效commit、96唯一样本及32个未提交样本重放均独立核验；到下一optimizer173.20秒、下一commit233.61秒，仅本工程样本。见 [报告](P3_AUTOMATIC_RECOVERY_REPORT.md) 与 [验收摘要](training-fault-verification-r1.json)。本轮容器/网络及GPU已释放；旧部署超时不改判，正式矩阵0与184待补不变，写入中接管/多rank仍未覆盖。

2026-09-21最新补充：P3真实训练接入工程验收通过，见 [P3_TRAINING_INTEGRATION_REPORT](P3_TRAINING_INTEGRATION_REPORT.md) 和 [独立验收摘要](training-integration-verification-r6.json)。真实3步optimizer/scheduler→async finalize→完整checkpoint→消费提交链；96个唯一样本、64个跨进程复用并提交；新进程完整状态精确加载通过，20.75GB重新哈希核验。最终同版本CPU21项通过，无跳过；宿主23项含重叠案例。所有本轮容器/网络已清理。剩余工作为真实故障重启/写入窗口安全接管、多rank及正式实验门槛；本结果不计正式GPU样本，不减少184项待补。

2026-09-21补充：双故障CPU隔离探针6项验收全通过，覆盖两次真实SIGKILL、实际fixture加载与保留链、错误触发/不完整候选/旧owner负例；容器删除已独立确认。见 [P2_SECOND_FAULT_REPORT](P2_SECOND_FAULT_REPORT.md)。不减少184项待补，不计为训练oracle或正式GPU样本。

更新：2026-09-20。当前目标：继续执行 FT-v1 及其脚本修复方案，直到对应可实施分支的实现、验证、冻结、实验和结果审计完成。不得用 P0、CPU fixture 或单 actor 探针代替完整交付。

模式：economy-dev 开启，root `01a0bd89-a9c3-7d00-b16d-4a0a766de676`；用户最近明确选择 Astra medium。当前无推理强度覆盖。实现由 Astra 承担，主任务验证及集成；不声称运行中的主管模型已切换。

## 已核实状态与下一步

| 工作 | 当前证据 | 剩余验收 |
|---|---|---|
| P0 来源/接口 | P0_REPORT、P0_INTERFACE_AUDIT、36行能力表、oracle_spec | 已完成接口审计；运行时未证项保留 |
| 基础恢复 | 官方CPU62项；单H100状态连续性；4卡真实RLVR保存/新进程RecoverHandler加载后更新PASS | RL内存逐组件独立连续性、pending、异步完成、多rank、实际故障仍待验证 |
| P1 runner/屏障 | 直接子进程11项；后代凭据/pidfd 8项；独立容器生命周期4项通过，7个测试容器移除已核验，见p1_evidence | CPU后代/失联收尾及官方AsyncRewardWrapper真实worker被杀后的原生重建重试已验证；新增trainer探针CPU合同7项、profile mock 2项、后代8项、生命周期4项及协议4项通过（native-cpu-verification-r1.json）；真实torchrun CPU重启/环境传播1项通过，RecoverInfo/身份关联负例4项与真实接口7项复验通过；GPU r4完成step0保存及第二次成功更新后唯一SIGKILL，但900秒内未原生retry，method timeout；清理已独立核实，恢复正例门禁未过。四轮本探针1.10370 GPU小时，正式0 |
| P2 完整R状态与oracle | state.py 21项（r3）、独立oracle 25项（r2）CPU合同通过；已修复旧祖先load伪保留链；真实评分入口通过；v2清单14项静态/driver25项通过；同版本完整408格CPU合同通过，96次真实SIGKILL、51次owner竞争；独立审计12项通过、408格全解析、384核心签名，F2.b01–b04的8组实际状态等价/仅规范marker不同；8限制/184待补仍保留，见full-408-verification-r1.json与signature-verification-408-r1.json | 新增控制器两个真实SIGKILL窗口及新进程不重复注入7项通过（controller-probe-r1）；i06清单顺序/i07第二目标仍待真实映射，完整generation/消费绑定/fencing/独立保留链仍开放 |
| P3 A/A+R | P3a draw机制9项及真实loader1项通过，无consumed authority；历史正分缓存8项通过但不可训练；新结构化返回缓存支持合法0/1，r2为11通过/1测试路径错误；仅修测试后该项r3真实库定向复验通过，内部fallback零分不可辨识仍待正式口径决定；早期RLVR入口和观察hook完成，4项真实库CPU合同通过；nofault-r4完成3步RLVR/32回答每批/最终6.92GB全hash校验；resume-r1实际加载后下一步更新/保存与最终fullhash验证通过 | 原生完整恢复、CPU真实executor→Grouped→PPO→train_batch入口桥1项通过（30.849秒），仅prepared_not_applied；真实replay进入optimizer并持久保存、两臂公共配置 |
| P4 C/C+R | 入口拒绝megatron已实测 | 官方组件的完整状态适配及共同底座验证；不能冒充原生支持 |
| FT1/P5 | 未启动 | 官方最小恢复、CPU合同与GPU短故障；技术门槛不要求各方法都恢复成功 |
| FT2/P6 | 未启动 | 固定配对矩阵、实际故障/RTO/成本；失败完整保留 |
| FT3 | 未启动 | C正式组件对照；B仅官方artifact可获取并适配通过后执行 |
| FT4 | 未启动 | 3对100步重复故障；后续schedule、保留链与收尾有效 |
| 最终报告 | 未生成 | runs/pairs/RESULTS可从原始证据复算；完整分母、预定统计与覆盖边界 |

## 设计约束与未决接口

- A选定v1内存dispatcher的F3预先N/A；不转移预算或合成ACK。其余A原矩阵/seed/F4主终点保持FT-v1。
- F4先检查真实自然事件；如须改为两阶段workflow，应在正式结果前完成明确的设计修订，不能暗改baseline或挑收益大的cell。
- X1/X2必须走实际reward/结果入口；通知callback不携reward，不能直接改内部pending字典冒充消息故障。
- C未映射的F3仍为条件阻塞；ByteCheckpoint不是RobustRL替代。B官方artifact未确认不推断其不存在，最终报告保留直接比较缺口。
- 正确恢复、训练继续、安全性分开；有效方法失败进入结果，技术无效保留原始证据。同seed技术补做最多一次。
- 准确率与正常吞吐/5%开销复测保持关闭；只核验本轮恢复正确性及成本。
- GPU资源每次启动重新读取，允许任意至少4张空闲卡；单actor探针只占用1张。不得按旧快照预留卡，不清理或覆盖历史失败产物。
- 所有源码/配置/镜像/输入/有效argv及oracle在正式运行前冻结；不同实现版本不合并成正式配对。

## 完成审计所需产物

正式放行前必须有 BASELINES 来源能力记录、适用cell清单、design、固定配对schedule、完整freeze、适配器diff、官方测试证据及functional_acceptance。每run保存实际硬件/argv/故障命中/worker退出/状态清单/恢复链/oracle结果/成本；每分支输出runs.csv、pairs.csv和RESULTS.md。当前P0文档不充当这些尚未交付的产物。

本轮起点现场没有rtx容器在运行，GPU0–6空闲、GPU7外部占用；这些读数仅是检查时事实。目标保持active，尚无完成或阻塞判定。
