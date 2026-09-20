# FT 实验执行状态

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
