# FT-v1 独立 oracle 规范（P0）

版本：P0-20260920；规范未实现，不能把本文或合同 fixture 当作真实恢复通过。接口依据见 P0_INTERFACE_AUDIT.md 的 A1–A6/C1–C2/S1–S2。主矩阵、seeds、观察窗、F4 主比较沿用 FT-v1；F3 在所审 A 路径预先 N/A，F4 仍待真实映射，本文不授权修改两阶段 workflow。

## 输入和隔离

每个 run 冻结 run_nonce、method/version、有效 argv/config、源码及补丁 hash、镜像 ID/digest（没有 registry digest 就明确空）、模型/数据全内容 hash、拓扑、设备 UUID、故障 schedule 和 verifier 定义。所有原始事件包含事件类型、source role/rank、PID/start-time/boot-id/cgroup、单调时钟及控制器接收时间、event_nonce、logical group/sample、attempt、recovery epoch、policy/verifier version，不能用 task_id 代替上述全部身份。

observer 只追加保存原始 prompt/label、回答/tokens/mask/logprobs 的证据、reward callable/config 和训练入口实际 tensor 摘要。独立 authority 离线重算 reward，检查实际值、版本和回答对应关系；不要信任方法自报 authoritative_reward。使用可复算全内容 hash，不能采文件首尾。observer/authority 原始副本不挂载给方法读取，不通过控制消息发送恢复数据；公共 hook 只观察和控制屏障。方法缺失的回答必须自行重新生成，R 在线保存/hash/I/O 均计成本。

事件至少覆盖 generation_done、reward_start/done、group_admitted、optimizer_start/end（update_successful）、checkpoint_schedule/finalize、metadata_written、checkpoint_loaded、role_ready，以及该部署确有的 commit/ACK。基线没有 R token 时使用其真实保存/加载及状态证据重建，不要求基线制造 token。没有 ACK 接口就不生成 ACK 事件。

## 完整状态与最终保留链

1. 恢复状态 generation 绑定父状态、实际更新集合、模型、optimizer moment/step/master state、scheduler 计数及下一 LR、各参与 rank Python/NumPy/Torch CPU/device RNG/tracker、data cursor/epoch/shuffle/pending 工作、policy version、配置和全量文件 hash。异步 prefetch 已推进 cursor 时，必须解释尚未消费工作。外层 global_step 不等于一次 optimizer 更新。
2. checkpoint_schedule、目录出现、API async 返回、step_info 或 tracker 之一单独不足以证明完整状态。先确认选定后端所有 rank 保存 finalize，再关联相同切点 metadata；固定路径覆写必须保留足够证据分清前后代。读写快照不一致或 hash 缺失是不能证明，不是默认 PASS。
3. 从实际重启加载的 generation 起，逐个验证到最终状态的父子关系。物理执行后被回滚的更新不在保留链上，允许重算、计入浪费。只有相同逻辑更新在最终保留链保留两次才算 duplicate commit。step 编号重用、同权重 hash 或两条 apply 日志均不能单独证明重复。
4. 每个保留更新反查真实训练输入，确认完整 K、sample/attempt 身份、policy staleness 合法、reward 与外部 authority 一致、无旧 epoch 越权。对旧 attempt 不能只看 logical ID 去重；旧结果先抢占也必须被识别。
5. R generation 提交要求：完整合法组→消费集合→实际 optimizer/scheduler 成功→相同切点完整状态保存→全 rank 完成及内容验证→原子发布不可变 token→真实 ACK（若有）。保存完成但 token 前失败，只有预先持久映射足以独立验证才可补提交，否则回退完整父代；token 指向缺文件或内容错必须拒绝。恢复 owner fencing 只允许一个有效 epoch。
6. baseline 不因缺少 R 协议自动判错：判其实际保留状态。反之 observer 不替 baseline 修复数据、补 ACK、持久化或重投。无法追溯训练 tensor 到 optimizer/保存加载，输出 unverifiable。

固定小模型连续执行 vs 保存→新进程加载→同下一批用来检验 state continuity；需报告 model、optimizer、scheduler、RNG 后续值和下一数据批。单 rank probe 不证明多 rank RNG；仅 engine probe 不证明 RecoverHandler/data cursor。精确 hash 不同需查数值及非确定性来源，不能凭 hash 不同直接宣布 invalid_commit。在线异步两臂不要求逐字节相同随机轨迹。

## 三维判定与 RTO

每个 run 独立输出：

- **safety**：pass / invalid_commit / unverifiable。完整组、正确奖励、无旧 epoch 污染、保留链无重复/遗漏一致性问题。安全丢弃不等于 invalid_commit。
- **training_continuation**：continued / safe_stop / timeout / unverifiable。证明故障后仍产生真实有效更新，不能只看进程启动或 reward 返回。
- **affected_work_recovery**：recovered / safely_dropped / unresolved / unverifiable。必须是故障目标 prompt/组或预先冻结的同 prompt、同 K 合法替代；原生 drop 后换新 prompt 只能证明训练继续。

`correct_recovered` 要求角色可服务、目标工作首次 oracle-valid 持久提交，且最终无安全性违规；首次恢复时间与完成目标工作量分别报告。RTO 起点是控制器证实实际注入的时刻，终点是上述两个条件同时成立时刻。其他组提交不能充当终点；第30个保留更新不能截断未解决目标组。跨进程使用经校验的控制器时间线，不能相减无关单调时钟。

保留单故障900秒、30步总45分钟、多故障100步总120分钟、计划内原生重试最多3次，分层重试次数/成本另列。未正确恢复的真实 RTO 为空/右删失；仅统计惩罚分数使用900秒，不能写“实际RTO=900”。同时记首次角色恢复、首次有效更新、目标工作恢复和最终工作量完成。

## 故障屏障合同

- 冻结 event_id、目标 group/sample/attempt、条件和目标角色。worker 在真实边界发送 ready 并等待；控制器核对 run nonce、PID start-time/boot-id/cgroup、角色、rank、真实状态证据及 event_nonce。信号前重查身份。不能广泛 pkill、关闭共享 Ray 或杀 observer/证据盘。
- F1 需要至少一个真实回答完成且同组尚有生成在途；F2 需要实际 optimizer 成功且该更新未完整保存；F3 需要真实持久未ACK集合，缺接口直接预先 N/A；F4 必须同时证明 K 回答完成及指定评分计数，执行中/已完成含义事先消歧；X1 值来自真实不同 verifier；X2 必须走实际公开结果接口，不改内部 pending 字典。
- 控制器追加 armed→fired→observed，注入后以退出/信号或真实消息行为证据确认命中。重复 ready/ack 必须幂等，event_nonce 不匹配拒绝；恢复保留后续 event，不清空整个 schedule。
- 注入成功并确认 `observed` 后，控制器负责向**所有仍存活的本事件等待方**发送幂等 `release(event_nonce)` 并收齐收据，包括非目标rank/同组workflow等待方；被杀进程不等待release ACK。发送失败交watchdog明确abort，不能让存活waiter永久阻塞，也不能把漏release造成的停顿算方法RTO。
- `release` 仅解除本屏障，不携带回答、reward 或 replay 建议。**F4 不能以 release 顺序隐式改造成两阶段算法**；需要先暂停所有 reward 才可达时，标设计修订待定/原定义未验证。
- 未匹配、已过窗口或尚未 armed 时 `abort`：停止注入、释放仍健康的等待方进入明确退出路径，保留证据，不能假称命中。collective rank 的暂停须一致，不能单 rank 永久阻塞其他 rank。
- watchdog 使用预先冻结的握手期限（独立于900秒方法恢复窗）。控制通道丢失、观察日志损坏、fired 状态不明：中止本 run，不盲目再杀；只对身份仍匹配的本 run 进程清理。屏障自身阻塞不能计为方法恢复超时。必须记录 release/abort/watchdog 的原因和收据。

## 结果分类、成本和审计负例

有效命中后方法不能恢复、自身资源耗尽、安全停止、错误提交或超时都是 method outcome，保留分母；不要因收益不理想改 N/A。未命中、身份不符、外部抢卡、observer/注入控制器故障才是 technical_invalid；保留原始记录，同 seed 最多补一次，不覆盖、不择优。unverifiable 明确缺哪段证据，不算通过。R 实现缺陷修复后重新冻结版本，修前后不混为配对。

结果状态至少包含 correct_recovered、safe_stop、invalid_commit、timeout、technical_invalid、not_applicable、unverifiable；同时保留三维字段，避免一个 success 掩盖安全丢弃。N/A 记录源码/部署范围、决定时间和原因，不能因训练恢复失败倒推“不支持”。

成本累计包括回滚更新、重复生成 token、丢弃回答、verifier 调用/CPU秒、I/O/checkpoint、分配GPU秒（包括保留/备用卡）和整run成本。不用历史每步常数估算新收益。observer 观测成本两臂同口径，R 在线持久化成本属于R。

未来实现必须通过以下独立反例：合法回滚重算不判重复；保留链重复才判错；完整性/版本错误 reward、旧 attempt 抢占、并发恢复、缺一 rank、部分 checkpoint、token 有而内容错、ACK 丢失（仅适用接口）、cursor跳过pending、伪造success、缺optimizer映射、observer数据反哺、PID重用、controller重启重复注入、多故障后续schedule丢失。模型fixture通过不升级任何真实 cell 能力。
