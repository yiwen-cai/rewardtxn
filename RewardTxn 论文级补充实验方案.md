---
名称: RewardTxn 论文级补充实验方案
导入时间: 2026-08-30T01:31:00.000Z
文档类型: 实验计划
---

> 文档状态：待执行实验设计，不代表已经取得实验结果。
>
> 修订记录（2026-08-30）：按审查意见补全为自包含分阶段方案——新增 §15 分阶段执行细则与附录 A–E（执行映射、术语表与 fault schedule schema、新故障注入与 oracle 规格、风险登记、阶段 tag 纪律）；正文修正 E3/E6/E7 的重复矩阵、指标口径与统计冻结规则；第 13 节相邻论文已逐条核实存在。
>
> 目标：把 RewardTxn 从“研究原型有效”提升到“足以支撑 MLSys / EuroSys 完整论文”的证据强度；NSDI 作为增加生产痕迹、跨节点规模和长期运行后的 stretch 目标。
>
> 制定日期：2026-08-30；目标硬件上限：单节点 8× NVIDIA H100 PCIe。
关联材料：
- [研究方案](https://app.notion.com/p/71168ec396634205a44a25f70ef5b6d7)
- [专项审计](https://app.notion.com/p/4d475b27b4404830959f9df9b05ec8fd)
- [H100 自包含实验指导](https://app.notion.com/p/3c64bd03a040810a8edfd74efbecaf20)
- [全实验交付说明](https://app.notion.com/p/3cb4bd03a04081a8ad0ccc14dcfac8bc)
---
## 1. 投稿主张与实验总原则
论文不应把 Group Seal、幂等、checkpoint 或 exactly-once 单独包装成创新，而应验证下面这条跨层主张：
> RewardTxn 将 verifier revision 一致的 Reward Group 与可持久恢复的 Optimizer Step 绑定，使奖励组要么恰好一次进入 durable optimizer history，要么对该历史完全不可见；崩溃、重试和 ACK 丢失不会形成静默混版本、部分提交或重复更新。
实验必须同时回答四类审稿问题：
1. **问题是否真实且具有普遍性**：不是只在 Slime 的单个 hook 中人为构造；
2. **新保证是否必要**：已有 group validation、dedup、lifecycle 和 per-step checkpoint 为什么仍不够；
3. **机制是否真的有效**：完整系统在真实进程崩溃和全切点回归中保持安全与可恢复；
4. **代价是否值得**：正常路径开销、恢复时间和重算量相对强 baseline 有明确收益。
所有实验遵循以下原则：
- 完成实验前冻结假设、指标、样本量和停止规则；
- 真实系统结果与协议模拟结果分表报告，不互相代替；
- 论文原系统与“论文机制等价实现”明确区分，禁止把后者写成完整复现；
- correctness 使用零容忍门禁，性能使用置信区间和效果量；
- checkpoint 本身的开销与 RewardTxn 元数据开销分开报告；
- 负结果、替代模型和资源限制必须进入最终论文的 Limitations。
---
## 2. 当前证据与论文缺口
<table fit-page-width="true" header-row="true">
<tr>
<td>证据维度</td>
<td>当前已有</td>
<td>投稿前必须补齐</td>
</tr>
<tr>
<td>问题真实性</td>
<td>Slime R1/R2/R3、TransferQueue Q0、Learner kill-9</td>
<td>第二个完整训练栈中的核心 invariant 复现；每类故障重复注入</td>
</tr>
<tr>
<td>Baseline</td>
<td>B0–B5 机制梯度和协议模拟矩阵</td>
<td>真实系统 baseline、统一配置、公平对比及置信区间</td>
</tr>
<tr>
<td>组件贡献</td>
<td>2A/2B/2C 分阶段门禁、AUTO_FIX on/off 配对</td>
<td>完整 leave-one-component-out 消融矩阵</td>
</tr>
<tr>
<td>正确性</td>
<td>20/20 混版本检出、真实自动恢复、9/9 回归</td>
<td>大样本 crash consistency 测试和失败率置信上界</td>
</tr>
<tr>
<td>性能</td>
<td>0.39% 元数据开销、2.78% 观测吞吐差、约 92% replay 节省</td>
<td>2/4/8 卡、多个 K/U/并发度、p50/p95 和配对统计</td>
</tr>
<tr>
<td>训练影响</td>
<td>0.5B 100 步、1.5B 20 步协议复验</td>
<td>至少 5 seeds、较长训练、clean/fault/oracle 三方对照</td>
</tr>
<tr>
<td>泛化性</td>
<td>数学 verifier，0.5B/1.5B/3B</td>
<td>数学 + 代码 sandbox 两类 reward；第二训练栈；7B 为可选增强</td>
</tr>
</table>
---
## 3. Research Questions、假设与论文门禁
<table fit-page-width="true" header-row="true">
<tr>
<td>RQ</td>
<td>可检验假设</td>
<td>主要指标</td>
<td>通过门槛</td>
</tr>
<tr>
<td>RQ1 问题普遍性</td>
<td>完整组检查和普通 dedup 之后，仍存在 reward revision 或 consume-to-commit 静默窗口</td>
<td>静默错误类型、受影响栈、optimizer delta</td>
<td>至少 2 类真实错误，覆盖 2 个独立栈或 1 个训练栈 + 1 个独立 data plane</td>
</tr>
<tr>
<td>RQ2 安全性</td>
<td>RewardTxn 在所有允许故障下保持零 invalid committed step</td>
<td>invalid commit、duplicate/missing step、reward mismatch</td>
<td>真实注入为 0；30,000 次确定性 trace 注入为 0（12 切点每切点 ≥1,500 次，分层随机化），使 95% 失败率上界约为 10\^-4（聚合口径；单切点按约 2,500 次计上界约 1.2×10\^-3，两个口径均如实报告）</td>
</tr>
<tr>
<td>RQ3 相对强 baseline</td>
<td>单独 validation、dedup、lifecycle 或 checkpoint 无法同时达到相同安全性与恢复成本</td>
<td>正确性、恢复成功率、重算 tokens、GPU/CPU seconds</td>
<td>RewardTxn 安全性不弱于最强 baseline，且昂贵重算中位节省至少 30%</td>
</tr>
<tr>
<td>RQ4 组件必要性</td>
<td>每个核心模块对应一种不可被其他模块替代的失败模式</td>
<td>leave-one-out 后的 invariant violation 或成本退化</td>
<td>每个核心模块至少有一个预注册反例；纯性能模块需有显著效果量</td>
</tr>
<tr>
<td>RQ5 效率与规模</td>
<td>RewardTxn 元数据路径不会抵消异步训练收益</td>
<td>step time、throughput、p95 latency、storage、scale efficiency</td>
<td>不含 checkpoint 的正常路径开销 95% CI 上界小于 5%</td>
</tr>
<tr>
<td>RQ6 学习语义</td>
<td>clean 路径与 oracle 等价；故障下 RewardTxn 比默认栈更接近 oracle</td>
<td>loss/reward 曲线、评测准确率、AUC、梯度差异</td>
<td>clean 等价性通过预注册 TOST；故障条件下不劣于 oracle margin，或如实降级为系统正确性论文</td>
</tr>
</table>
---
## 4. 统一实验设置
### 4.1 软件栈与版本
<table fit-page-width="true" header-row="true">
<tr>
<td>对象</td>
<td>固定版本</td>
<td>实验角色</td>
</tr>
<tr>
<td>THUDM/slime</td>
<td>v0.3.1，a6272da0d4f3d0a08520c99a2f3b4f6c887960dc</td>
<td>主训练栈、完整 RewardTxn 实现、主要 baseline</td>
</tr>
<tr>
<td>AReaL</td>
<td>b83d1f40196e5bd7d9f83092563443561870d550</td>
<td>第二训练栈；复现 incomplete-group 与 revision/commit 边界</td>
</tr>
<tr>
<td>TransferQueue</td>
<td>release/v0.1.10，8497a52a5c4347c4d67c97f7ce500e544426a08a</td>
<td>独立 data plane 的 consume/crash/reclaim 微基准</td>
</tr>
<tr>
<td>RewardTxn</td>
<td>从 phase3-final 创建冻结的 paper-eval tag（即 §15.1 的 paper-e0；后续每阶段 tag 命名见附录 E）</td>
<td>所有论文实验必须记录 commit、镜像 digest 与配置 hash</td>
</tr>
</table>
### 4.2 模型、任务与硬件
- **高重复故障扫描**：Qwen2.5-0.5B-Instruct，2/4 卡；
- **论文主结果**：Qwen2.5-1.5B-Instruct，4 卡；
- **规模泛化**：Qwen2.5-3B-Instruct，4/8 卡；
- **可选增强**：Qwen2.5-7B-Instruct，只有 pinned-memory 问题解决且不挤压核心重复数时执行；
- **数学 reward**：DAPO-Math-17k + GSM8K；
- **代码 reward**：HumanEval 或 MBPP，使用确定性本地 CPU sandbox；
- **group size K**：4、8、16；
- **每步组数 U**：1、4、8；
- **训练随机种子**：17、29、42、73、101；
- **硬件规模**：2、4、8 H100；所有对比固定 GPU 型号、卡号布局和 CPU/IO 配额。卡号按拓扑固定：2 卡用 GPU 1,2（NV12 直连）、4 卡用 GPU 0,1,2,5 或 3,4,6,7（NUMA 内环）、8 卡用全部（H100 指导 §3.3）。若 8 卡窗口受外部占用不可得，核心结论以 4 卡为准，8 卡只在偶发窗口做单次确认并明确标注（Figure 3 降级口径见 §11）。
### 4.3 公平性控制
同一张对比表中的变体必须保持模型、数据顺序、seed、global batch、K、U、训练步数、GPU 数、checkpoint 频率和 verifier 版本完全一致。运行顺序采用随机区组设计，以日期/GPU 组为 block；每个变体交替执行，避免把机器负载变化误判为系统收益。
---
## 5. Baseline 设计
### 5.1 可运行系统 baseline
<table fit-page-width="true" header-row="true">
<tr>
<td>编号</td>
<td>实现</td>
<td>覆盖能力</td>
<td>论文表述</td>
</tr>
<tr>
<td>S0</td>
<td>Slime v0.3.1 默认 fully-async 路径</td>
<td>默认吞吐、R1/R2/R3/L0/L2 行为</td>
<td>真实系统 baseline</td>
</tr>
<tr>
<td>S1</td>
<td>AReaL 固定提交，开启 group ID 与 drop_incomplete_group</td>
<td>组完整性、policy staleness、whole-group retry</td>
<td>真实系统 baseline；不能只用模拟替代</td>
</tr>
<tr>
<td>S2</td>
<td>TransferQueue 固定版本</td>
<td>producer/consumer、dedup、consume-to-trainer crash window</td>
<td>独立 data-plane baseline，不声称是完整训练系统</td>
</tr>
<tr>
<td>S3</td>
<td>RewardTxn on Slime</td>
<td>完整端到端协议</td>
<td>目标系统</td>
</tr>
<tr>
<td>S4</td>
<td>RewardTxn 最小 invariant adapter on AReaL</td>
<td>Group Seal + StepToken + recovery 判定</td>
<td>跨栈泛化；若无法完成，论文必须降级泛化 claim。实施排入 P2：P1 仅完成 AReaL 上的 R2/R3 故障复现（E1 门禁），P2 再实施 adapter（映射见附录 A）</td>
</tr>
</table>
### 5.2 机制梯度 baseline
<table fit-page-width="true" header-row="true">
<tr>
<td>编号</td>
<td>机制</td>
<td>对应相邻工作</td>
<td>必须回答的问题</td>
</tr>
<tr>
<td>B0</td>
<td>默认开源栈</td>
<td>Slime</td>
<td>不加保护时故障是否静默</td>
</tr>
<tr>
<td>B1</td>
<td>explicit group ID + exact group-size validation</td>
<td>complete-group 系统的最低保护</td>
<td>完整但混版本的组是否仍穿透</td>
</tr>
<tr>
<td>B2</td>
<td>drop incomplete group + whole-group retry</td>
<td>AReaL-like</td>
<td>安全丢弃能否解决 revision 与 commit 问题</td>
</tr>
<tr>
<td>B3</td>
<td>logical ID + idempotent dedup</td>
<td>TransferQueue-like</td>
<td>普通去重能否处理旧 epoch、ACK ambiguity 和 nondeterminism</td>
</tr>
<tr>
<td>B4</td>
<td>Reserve/Occupy/Consume + timeout/reclaim</td>
<td>StaleFlow-like</td>
<td>生命周期管理能否跨越 optimizer durable boundary</td>
</tr>
<tr>
<td>B5</td>
<td>per-step checkpoint + full-step/full-group restart</td>
<td>RobustRL/Belayer-like recovery envelope</td>
<td>在达到同等安全性时，恢复成本是否过高</td>
</tr>
<tr>
<td>B6</td>
<td>Seal + fencing + checkpoint-bound StepToken + Reconciler + Selective Replay</td>
<td>RewardTxn</td>
<td>是否同时获得安全性、可判定恢复和低重算</td>
</tr>
</table>
论文中 B4/B5 必须写成 literature-inspired 或 mechanism-equivalent baseline，除非确实运行了论文官方实现；不得写成“复现 StaleFlow/RobustRL/Belayer”。
RolloutPipe、DistRS 的主要目标分别是 complete-group pipelining 和 Reward Service 资源调度，与 RewardTxn 的 crash-consistency 主指标不完全同构。它们应进入 Related Work 和边界实验，不应被强行包装成同任务 baseline。
---
## 6. 故障模型与统一注入矩阵
<table fit-page-width="true" header-row="true">
<tr>
<td>编号</td>
<td>注入动作</td>
<td>预期风险</td>
<td>主要 invariant / 指标</td>
</tr>
<tr>
<td>R1</td>
<td>reward worker 计算中异常或 kill-9</td>
<td>残缺组或静默丢样本</td>
<td>无 partial group commit；恢复成功率</td>
</tr>
<tr>
<td>R2</td>
<td>retry 返回旧 reward revision</td>
<td>stale result 混入当前组</td>
<td>committed reward revision mismatch = 0</td>
</tr>
<tr>
<td>R3</td>
<td>同组前后半使用不同 verifier image</td>
<td>完整组的静默混版本</td>
<td>混版本组进入训练器的样本数 = 0</td>
</tr>
<tr>
<td>R4</td>
<td>相同 logical ID/revision 返回不同 digest</td>
<td>nondeterministic conflict 被覆盖</td>
<td>冲突必须阻断，禁止 last-write-wins</td>
</tr>
<tr>
<td>R5</td>
<td>两个 recovery worker 并发写同一组</td>
<td>重复权威记录或旧 attempt 复活</td>
<td>唯一 CAS winner；旧 epoch 写入为 0</td>
</tr>
<tr>
<td>Q0</td>
<td>mark-consumed 后、get-data 前 kill consumer</td>
<td>队列数据永久不可重取</td>
<td>样本可 reclaim 或由 manifest 判定恢复</td>
</tr>
<tr>
<td>Q1</td>
<td>数据已取出、StepManifest 前 kill learner</td>
<td>数据悬空或重复投递</td>
<td>未提交数据最终恰好一次进入计划</td>
</tr>
<tr>
<td>L0</td>
<td>StepManifest 准备前 kill trainer</td>
<td>恢复入口丢失</td>
<td>恢复后无 committed-step 误判</td>
</tr>
<tr>
<td>L2</td>
<td>optimizer 执行中 kill rank/actor</td>
<td>部分 rank 更新或错误续跑</td>
<td>全 Trainer rollback；无 partial optimizer state</td>
</tr>
<tr>
<td>L3</td>
<td>optimizer 返回后、checkpoint 前 kill</td>
<td>内存更新被误认为 durable</td>
<td>仅信任 durable checkpoint/token</td>
</tr>
<tr>
<td>C1</td>
<td>checkpoint durable 后、commit pointer 前 kill</td>
<td>已写 checkpoint 但提交状态模糊</td>
<td>Reconciler 给出唯一判定</td>
</tr>
<tr>
<td>C2</td>
<td>commit pointer 发布后丢 ACK</td>
<td>恢复后重复 apply</td>
<td>duplicate optimizer effect = 0</td>
</tr>
</table>
每个注入事件记录 phase、step、group IDs、attempt/epoch、verifier digest、checkpoint hash、StepToken、进程退出码、恢复决策和最终 oracle verdict。

说明：本表在早期 R0–C2 基础上并入 R4/R5；省略 L1（forward/backward 完成、optimizer 执行前崩溃——其回滚语义已被 L2 的全 Trainer rollback 覆盖，不再单列切点，避免审稿人对照早期文档产生覆盖疑问）；R4/R5/Q1/L3 的注入机制与实现位置见附录 C.1。
---
## 7. 补充实验 E1–E8
### E1：跨栈问题复现与最小反例
**目的**：证明 RewardTxn 不是 Slime-specific bug fix。
**设计**：
1. 在 Slime 上重复 R2、R3、Q0、L2、C1、C2；
2. 在 AReaL 上至少重复 R2/R3 和 learner/checkpoint 边界；
3. 在 TransferQueue 上重复 Q0/Q1，并连接一个最小 trainer consumer；
4. 每个真实进程切点至少独立执行 10 次（关键切点 R3/Q0/L2/C1/C2 在 E2 中提高至每个 baseline/RewardTxn 组合至少 20 次）；确定性接口级反例每个执行 100 次；
5. 记录 default、group validation、dedup 后错误是否仍存在。
**门禁**：
- 至少两个独立系统边界出现可复现错误；
- 至少一类错误在“完整组 + 普通 dedup”后仍然存在；
- 错误必须可测地改变 reward、advantage、gradient 或 durable-step 判定；
- 如果第二训练栈无法复现，论文只保留 Slime + independent data plane claim，并在标题/摘要降级泛化表述。
### E2：Crash-consistency 全切点安全性
**目的**：为核心 invariant 提供统计上可解释的零错误证据。
**设计**：
- Trace Runner 对 R1–R5、Q0–Q1、L0/L2/L3、C1/C2 共 12 个切点执行不少于 30,000 次确定性随机化调度，每切点不少于 1,500 次（分层抽样，避免组合维度饿死某切点）；
- 随机化维度包括 kill 时间、ACK 丢失、attempt 到达顺序、revision、group 大小和 checkpoint 延迟；
- 对 R3、Q0、L2、C1、C2 五个关键切点做真实进程级实验，每个 baseline/RewardTxn 组合至少 20 次；R5 增加真实进程级并发恢复实验（复用 Phase 3C 多进程 CAS 底座，并发 worker ≥2，期望唯一 CAS winner）；
- oracle 判定规则（什么算 invalid committed step）见附录 C.2，由自动 oracle 脚本对每个事件给出 verdict；
- 对恢复后的模型/optimizer state 做内容 hash、step 序列和 oracle delta 对拍。
**主要指标**：
- invalid committed GroupManifest / StepManifest；
- duplicate 或 missing StepToken；
- checkpoint/token hash mismatch；
- 错误 reward 进入 committed gradient 的比例；
- liveness failure 和恢复超时。
**门禁**：
- RewardTxn 在确定性 trace 与真实进程实验中均为 0 invalid commit；
- 30,000 次零失败按 rule of three 给出约 10\^-4 的 95% 失败率上界；
- B1–B5 至少各有一个预注册反例或可量化成本退化。
### E3：强 baseline 公平对比
**目的**：证明 RewardTxn 不是只比默认栈更好。
**设计**：
- 在相同 1.5B、K=8、U=4、4 H100 配置下运行 B0–B6；
- 分别使用 R3、Q0、L2、C1/C2 代表 revision、queue、optimizer、commit 四类故障；
- 每个 cell 至少 10 次真实短训重复（每次 20–30 步；B0/B2/B5/B6 跑满 4 类故障，B1/B3/B4 减为 2 类故障 ×10 次）；性能结果采用配对运行，配对仅对 B0/B2/B6 执行；
- 统一 checkpoint 频率，另外给出“checkpoint 开销剥离”的结果；
- B2/B4/B5 的策略参数在预实验后冻结，不能为 RewardTxn 单独调优。
**输出**：
- Correctness matrix：每个 baseline 能否检测、阻断、恢复；
- Recovery work：重新生成 tokens、重算 reward、重跑 optimizer steps；
- Recovery time：p50/p95 与 95% CI；
- Failure-free overhead：step time 与 throughput；
- Storage/network：manifest、CAS、checkpoint bytes。
**门禁**：
- RewardTxn 是唯一覆盖全部 invariant 的变体，或明确承认与其他系统等价；
- 相比达到相同安全性的最强 baseline，昂贵重算中位节省至少 30%；
- 如果 B5 在同等安全下成本差异不足 20%，Selective Replay 不应作为主要性能贡献。
### E4：逐模块消融
<table fit-page-width="true" header-row="true">
<tr>
<td>变体</td>
<td>移除/替换内容</td>
<td>主故障</td>
<td>预注册预期</td>
</tr>
<tr>
<td>A0</td>
<td>完整 RewardTxn</td>
<td>全部</td>
<td>零 invalid commit，最低正确恢复成本</td>
</tr>
<tr>
<td>A1</td>
<td>移除 Group Seal/revision check</td>
<td>R2/R3</td>
<td>完整混版本组可进入训练</td>
</tr>
<tr>
<td>A2</td>
<td>移除 Reward CAS</td>
<td>R5/duplicate</td>
<td>并发写产生多份权威结果</td>
</tr>
<tr>
<td>A3</td>
<td>保留 CAS、移除 epoch/attempt fencing</td>
<td>R2/R5</td>
<td>迟到旧 attempt 有机会覆盖或复活状态</td>
</tr>
<tr>
<td>A4</td>
<td>移除 checkpoint-bound StepToken</td>
<td>C1/C2</td>
<td>恢复无法唯一判断 committed step</td>
</tr>
<tr>
<td>A5</td>
<td>移除 Reconciler，直接从 latest checkpoint 重启</td>
<td>Q0/L3/C1</td>
<td>出现重复、丢失或恢复空转</td>
</tr>
<tr>
<td>A6</td>
<td>Selective Replay 替换为 whole-step replay</td>
<td>R1/R3/Q0</td>
<td>正确性相同但重算成本显著升高</td>
</tr>
<tr>
<td>A7</td>
<td>group-RM 替换为 single-sample RM</td>
<td>R3</td>
<td>持久审计可修正，但已返回训练器的 reward 无法撤回</td>
</tr>
</table>
**执行要求**：
- 确定性层每个变体×故障至少 100 次；
- 真实系统仅选择与该模块直接相关的 1–2 个关键故障，每个 cell 至少 10 次；
- 只移除一个模块，其他配置完全相同；
- 正确性指标使用 Wilson 区间；性能使用配对 bootstrap 95% CI 和 Cliff’s delta；
- 多个消融同时检验时使用 Holm–Bonferroni 校正。
### E5：Selective Replay 恢复收益边界
**目的**：回答“什么时候 selective replay 才值得”。
**自变量**：
- K ∈ \{4,8,16\}；
- U ∈ \{1,4,8\}；
- 失效组占比 ∈ \{1/8, 1/4, 1/2, 1\}；
- reward 成本：轻量数学规则、CPU sandbox、人工注入的 100ms/1s/5s verifier；
- 故障点：R1、R3、Q0、C1；
- 模型：0.5B 与 1.5B。
**对比**：whole-group retry、whole-step replay、latest-checkpoint restart、RewardTxn selective replay。
**指标**：重复 rollout tokens、reward CPU/GPU seconds、sandbox seconds、重复 optimizer steps、RTO、总 GPU-hours。
**门禁**：
- 报告收益随失效比例和 reward 成本变化的曲线，而不是只给一个最佳数字；
- 核心 workload 中位节省至少 30%，95% CI 下界大于 20%；
- 明确 selective replay 在“整步全部失效”时收益趋近于零的边界。
### E6：正常路径开销与可扩展性
**矩阵**：
- GPU：2/4/8；
- 模型：0.5B/1.5B/3B；
- K：4/8/16；
- reward concurrency：16/32/64；
- 变体：Slime default、group-RM only、Seal+CAS、完整 RewardTxn。
**测量方法**：
- 每次运行先 warmup 20 steps，再测量至少 100 steps；
- 两级重复设计：核心 cell（1.5B、4 卡、K=8，完整 RewardTxn vs group-RM only）至少 5 次独立运行并配对；其余格点先做 1 次筛选扫描，只对接近门禁（开销 95% CI 上界 5%）的格点补足重复；
- 报告 throughput、step time p50/p95、GPU utilization、CPU、memory、IO、metadata bytes；
- 使用 CUDA Event 记录 GPU 段，wall clock 记录端到端段；
- checkpoint on/off 分开画图，避免把 checkpoint 代价算入协议元数据。
**门禁**：
- 核心 4 卡 1.5B 配置中，RewardTxn 相对同 group-RM baseline 的开销 95% CI 上界小于 5%；
- 2→4→8 卡时协议开销不能随卡数超线性增长；
- CAS/manifest 数据量随 committed groups 近似线性增长且 retention 后稳定。
### E7：训练语义与多 seed
**组别**：
1. Clean Oracle：无故障、权威 v1；
2. Clean RewardTxn：完整协议、无故障；
3. Faulted Default：默认栈，注入 revision/crash；
4. Faulted B5：per-step checkpoint + whole replay；
5. Faulted RewardTxn：相同故障序列、完整协议。
**核心配置**：
- Qwen2.5-1.5B、DAPO-Math/GSM8K、K=8、U=4、4 H100；
- seeds 17/29/42/73/101；
- 最少 500 optimizer steps；资源允许时延长到 1,000 steps；
- 使用预生成 fault schedule（schema 与生成规则见附录 B.2），使同 seed 各变体经历完全相同的故障；
- 每 50 steps 保存训练指标并跑固定评测集（评测集划分见附录 C.3）。
**统计方案**：
- 主要终点：最终评测准确率（代码 verifier 为 pass@1）；reward AUC 仅用于可选 learned judge 边界实验，不作为确定性 verifier 的主指标；
- 次要终点：loss MAE、梯度 cosine、有效样本率和 wall-clock-to-quality；
- Clean RewardTxn vs Clean Oracle 使用 TOST 等价性检验；
- 等价 margin 在 pilot 后、查看正式结果前冻结为单一值：首选最终准确率 ±1 个绝对百分点，同时报告标准化效应量；禁止在 ±1pp 与 0.2×seed 标准差之间事后二选一；loss MAE 仅作辅助指标，不作为等价性主判据；
- Faulted 组使用相同 seed 配对比较，报告均值、95% CI 和效果量。
**解释规则**：
- 如果 Faulted Default 没有显著学习退化，不能宣称 RewardTxn 提升模型质量；
- 此时仍可主张系统语义正确性、可恢复性和资源节省；
- 如果 Clean RewardTxn 未通过等价性，必须先定位协议是否改变数据顺序/样本分布，不能用更多 seed 掩盖。
### E8：工作负载泛化与长期 soak
**工作负载**：
- 数学 verifier：确定性答案匹配；
- 代码 verifier：HumanEval/MBPP + 确定性本地 CPU sandbox（选型与参数见附录 C.4）；
- 可选 nondeterministic judge：只作为边界实验，不作为主正确性 oracle。
**长期运行**：
- 4 H100、1.5B、至少 3 次×8 小时；
- 使用冻结的 Poisson fault schedule，平均每 30 分钟一次真实进程/ACK/queue 故障；
- 可选 24 小时单次 soak 作为 artifact 增强；
- 持续记录 CAS 大小、WAL/manifest、checkpoint retention、内存、磁盘、恢复次数和训练进度。
**门禁**：
- 零 invalid commit；
- 每次故障均在预设 RTO 内恢复，连续无进度不超过 10 分钟；
- retention 后磁盘与 metadata 增长有界；
- 数学和代码 reward 均通过相同 group-to-optimizer invariant。
---
## 8. 统计与报告规范
### 8.1 样本量与重复
- 确定性 correctness：按事件计数，核心结论使用不少于 30,000 次 trace 注入；
- 真实进程正确性：关键 cell 至少 20 次，普通 baseline/消融 cell 至少 10 次；
- 性能：每个核心配置至少 5 次独立运行，每次不少于 100 个稳定 steps；
- 训练效果：至少 5 个固定 seeds；
- 预实验只用于估计方差和冻结 margin，不进入正式显著性检验；
- 所有预注册内容（样本量、排除条件、单一等价 margin、停止规则）以 JSON 提交到仓库 prereg/ 目录，与实验代码同 commit 冻结；
### 8.2 统计方法
- 比例/失败率：Wilson 或 Clopper–Pearson 95% CI；
- 延迟和吞吐：paired bootstrap 10,000 次，报告 median、p95、CI；
- 非正态配对数据：Wilcoxon signed-rank，并同时报告 Cliff’s delta；
- 多消融/多故障比较：Holm–Bonferroni；
- 学习等价性：TOST，而不是把“不显著”误写成“等价”；
- 所有图展示原始点或 seed 曲线，不能只画均值柱状图。
### 8.3 停止、异常与复现规则
- 正式实验不得因结果好看而提前停止；
- 只有预注册的安全门禁、资源故障或序贯规则允许提前终止；
- 外部 GPU 抢占、磁盘故障、网络中断单独标记，不静默删点；
- 排除任何 run 必须保留原始日志并在 exclusion log 中给出原因；
- 所有分析脚本从 immutable raw JSON/JSONL 生成表格，禁止手改论文数字。
---
## 9. 执行顺序、资源门禁与优先级
<table fit-page-width="true" header-row="true">
<tr>
<td>阶段</td>
<td>内容</td>
<td>预计时长</td>
<td>进入下一阶段条件</td>
</tr>
<tr>
<td>P0</td>
<td>冻结 paper-eval 版本、schema、fault schedule、统计脚本</td>
<td>2 天</td>
<td>CPU regression、数据完整性和配置 hash 全通过</td>
</tr>
<tr>
<td>P1</td>
<td>E1 跨栈复现 + E2 trace 安全性</td>
<td>3–4 天</td>
<td>问题普遍性与零 invalid commit 门禁通过</td>
</tr>
<tr>
<td>P2</td>
<td>E3 baseline + E4 消融</td>
<td>4–5 天</td>
<td>每个核心模块有反例，最强 baseline 定义冻结</td>
</tr>
<tr>
<td>P3</td>
<td>E5 replay 边界 + E6 开销/规模</td>
<td>4–5 天</td>
<td>开销与恢复收益达到门槛</td>
</tr>
<tr>
<td>P4</td>
<td>E7 多 seed 训练 + E8 workload/soak</td>
<td>7–10 天</td>
<td>训练等价性、长期恢复和 artifact 完整</td>
</tr>
</table>

> 本表为概览；每阶段的详细任务分解、脚本入口、产出物、tag 与回退点见 §15，实验到脚本的映射见附录 A。

执行优先级：
1. **必须完成**：E1–E7、第二训练栈核心路径、5-seed 训练、完整消融；
2. **强烈建议**：代码 verifier、3×8 小时 soak、8 卡规模；
3. **可选增强**：7B、24 小时 soak、真实生产 failure trace；
4. **不得挤占核心重复数**：大模型单次展示、复杂 dashboard、跨数据中心复制。
GPU 预算不预先拍脑袋固定。P0 用 pilot 测得每个 cell 的 wall time 后，按“重复数 × GPU 数 × wall hours”生成预算；如果预算不足，先删除可选模型/规模点，不降低核心 baseline、消融和 seed 数。每阶段预算超支 20% 触发裁剪评审（附录 D 风险 3）。基于 Phase 1–3 实测 wall time（1.5B/4 卡 step median ≈9s、mean ≈51s）的粗算见 §15.6；各阶段详细任务、脚本入口、tag 与回退点见 §15 与附录 A/E。
---
## 10. 实验产物与数据契约
每个 run 必须具有唯一 exp_id，并保存：
- meta.json：代码提交、镜像、模型、数据、seed、K/U、GPU、fault schedule；
- config.json：所有命令行参数和环境变量；
- events.jsonl：group、reward、queue、learner、checkpoint、ACK 时间线；
- manifests/：GroupManifest、StepManifest、StepToken；
- metrics.json：correctness、performance、recovery、training 指标；
- resource.jsonl：GPU/CPU/memory/disk/network；
- verdict.json：自动门禁结果；
- stdout/stderr 与进程退出状态；
- artifact_manifest.json：所有证据文件的 SHA-256。
聚合报告必须保留从 figure/table cell 到 exp_id 的反向索引，使审稿人和 artifact evaluator 能从论文数字定位到原始日志。
---
## 11. 论文图表清单
<table fit-page-width="true" header-row="true">
<tr>
<td>编号</td>
<td>图表</td>
<td>回答的问题</td>
</tr>
<tr>
<td>Figure 1</td>
<td>Reward Group → Queue → Optimizer → Checkpoint 故障时间线</td>
<td>新 correctness boundary 在哪里</td>
</tr>
<tr>
<td>Table 1</td>
<td>系统、版本、模型、数据和硬件</td>
<td>实验是否可复现和公平</td>
</tr>
<tr>
<td>Table 2</td>
<td>B0–B6 × R/Q/L/C correctness matrix</td>
<td>已有机制为什么不够</td>
</tr>
<tr>
<td>Table 3</td>
<td>A0–A7 消融</td>
<td>每个模块是否必要</td>
</tr>
<tr>
<td>Figure 2</td>
<td>不同 fault fraction / reward cost 下的 replay work</td>
<td>Selective Replay 的收益边界</td>
</tr>
<tr>
<td>Figure 3</td>
<td>2/4/8 GPU throughput 与协议开销（8 卡窗口不可得时降级为 2/4 并明确标注，规模性结论限于 4 卡）</td>
<td>正常路径和规模性</td>
</tr>
<tr>
<td>Figure 4</td>
<td>恢复时间 p50/p95 与分项耗时</td>
<td>崩溃后恢复是否更快</td>
</tr>
<tr>
<td>Figure 5</td>
<td>5-seed clean/fault/oracle 训练曲线</td>
<td>是否保持或恢复学习语义</td>
</tr>
<tr>
<td>Table 4</td>
<td>Slime/AReaL/TransferQueue 与数学/代码 reward 泛化</td>
<td>是否超越单栈单 workload</td>
</tr>
<tr>
<td>Figure 6</td>
<td>8 小时 soak 的故障、恢复和存储时间线</td>
<td>长期运行是否有界且有进展</td>
</tr>
</table>
---
## 12. 最终投稿 Go / No-Go
### 12.1 MLSys / EuroSys 完整论文 Go
必须全部满足：
- 两个独立系统边界中存在真实问题证据；
- B0–B6 公平 baseline 完成，且 B4/B5 表述诚实；
- A0–A7 消融完成，每个 correctness 模块有独立反例；
- 30,000 次 trace 和关键真实故障中 RewardTxn 为零 invalid commit；
- 正常路径协议开销 95% CI 上界小于 5%；
- 相比同等安全 baseline，核心 workload 重算中位节省至少 30%；
- 至少 5 seeds 的 clean 等价性完成；
- 代码、配置、原始日志、分析脚本和 artifact manifest 可复现。
### 12.2 降级或 No-Go 条件
- AReaL/第二边界无法复现且只能证明 Slime-specific 行为；
- B5 达到同等安全性且恢复成本差异小于 20%；
- 去掉 StepToken/Reconciler 后没有可观测差异；
- clean RewardTxn 改变训练样本语义且无法解释；
- 主要结果依赖单 seed、单次 8 卡或协议模拟器；
- 新近工作公开了等价的 reward-revision-to-durable-optimizer commit。
出现上述情况时，应缩小 claim 为“Slime 的 crash-consistency extension / artifact”，或转为 workshop、短论文和上游工程贡献，不继续扩张“通用 RLVR 事务协议”表述。
---
## 13. 相邻论文与实验标尺
- [AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/33c00862bfa29ac72ecf630a41e19352-Abstract-Conference.html)：异步 RL、staleness 与第二训练栈标尺。
- [DistRS: Disaggregated Reward Service for RLVR with Batch-Level Constraint, NSDI 2026](https://www.usenix.org/conference/nsdi26/presentation/zhu-ruidong)：Reward Service workload、batch constraint 与真实任务标尺。
- [RobustRL: Role-Based Fault Tolerance System for RL Post-Training, OSDI 2026](https://www.usenix.org/conference/osdi26/presentation/chen-zhenqian)：角色级故障隔离、恢复效率和大规模 fault injection 标尺。
- [StaleFlow: Staleness-Aware Data Management for Fully Disaggregated RL Post-Training](https://arxiv.org/abs/2601.12784)：trajectory lifecycle、global consistency 与 ablation 标尺。
- [RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM RL](https://arxiv.org/abs/2606.26997)：complete-group pipeline 与 on-policy 边界。
- [Belayer: Efficient Fault Tolerance for LLM Agentic RL Training](https://arxiv.org/abs/2608.14635)：rollout/environment recovery、prefix consistency 与 recovery correctness 标尺。
本方案对这些工作的判断仅限 2026-08-30 可见的公开论文和实现；该日已逐条核实本清单链接与论文存在性，正式投稿前必须再次进行增量查重。
---
## 14. 执行前 Checklist
- [ ] 创建 paper-e0（paper-eval 冻结）tag，并冻结代码/镜像/模型/数据版本；
- [ ] 扩展注入器支持 R4/R5/Q1/L3（附录 C.1），生成 fault schedule 与 schema（附录 B.2）；
- [ ] 补齐 AReaL 的 RewardTxn adapter（P2，附录 A）或明确降级 claim；
- [ ] 为 A0–A7 建立独立 feature flags（附录 A）；
- [ ] 生成固定 fault schedules 和 5 个训练 seeds；
- [ ] 完成 raw event schema 与自动 oracle（附录 C.2）；
- [ ] 先跑 pilot，冻结单一等价 margin、方差和 GPU-hour 预算，写入 prereg/；
- [ ] 预注册每个实验的样本量、排除条件和停止规则（prereg/ JSON）；
- [ ] E6 两级重复设计、E3 重复矩阵裁剪通过评审（§7）；
- [ ] 通过 CPU regression 后再占用 H100；
- [ ] 每批实验自动生成 verdict 与 artifact manifest；
- [ ] 每阶段按附录 E 提交门禁 JSON、阶段文档并打 tag；
- [ ] 每周增量检查相邻工作，投稿前完成最终 related-work audit。

---
## 15. 分阶段执行细则（P0–P4）

§9 的 P0–P4 表是概览；本节给出每个阶段的详细任务、脚本入口、门禁、产出物、tag 与回退点。所有 tag 从 `phase3-final` 链上创建，命名 `paper-e0 … paper-final`；每阶段完成定义与回退流程见附录 E。实验到脚本的映射见附录 A。

### 15.1 P0：冻结与预注册（约 2 天）

**任务**：
1. 从 `phase3-final` 创建 `paper-e0` tag，冻结代码/镜像/模型/数据版本（扩展 STACKS.lock 加 paper-eval 一节）；
2. 扩展故障注入器支持 R4/R5/Q1/L3（附录 C.1），为 A0–A7 建立独立 feature flags（附录 A）；
3. 生成 fault schedule 生成器与 5 个 seeds（附录 B.2），写入 prereg/；
4. 定义 raw event schema 与自动 oracle（附录 C.2）；
5. 提交预注册 JSON：样本量、排除条件、单一等价 margin、停止规则（§8.1）；
6. 冻结统计脚本（paired bootstrap、Wilson/Clopper–Pearson、TOST）；
7. Pilot：对 E3/E6/E7 的核心 cell 各跑 2–3 次短运行，测量 wall time、方差与 verifier 成本，校准 GPU-hour 预算与 margin。

**门禁**：CPU regression（`phase3_regress.sh`）9/9 全绿；配置 hash 校验通过；prereg/ 文件齐全且已 commit；pilot 预算表落地（§15.6 更新）。

**产出**：`paper-e0` tag、`prereg/*.json`、注入器扩展、fault schedule 生成器、预算表。

**回退点**：`phase3-final`。

### 15.2 P1：E1 跨栈复现 + E2 trace 安全性（3–4 天）

**任务**：
1. E1 Slime 六切点（R2/R3/Q0/L2/C1/C2）×每切点 ≥10 次真实注入（`day2_run.sh`/`phase2_run.sh` + `day4_inject.sh`）；
2. E1 AReaL：先用半天探针确认训练栈可跑通，再完成 R2/R3 与 learner/checkpoint 边界复现（新增 areal_repro 入口，附录 A）；
3. E1 TransferQueue：Q0/Q1 复现 + 最小 trainer consumer（`day3_tq_crash_probe.py` 扩展）；
4. E2：扩展 `phase2_trace_runner.py` 至 12 切点，跑 30,000 次确定性随机化调度（每切点 ≥1,500），接入自动 oracle；
5. E2 真实进程：R3/Q0/L2/C1/C2 ×每组合 ≥20 次 + R5 并发恢复 ≥10 次；
6. 每个注入事件按 §6 记录完整字段，恢复后 state 对拍（hash + step 序列 + oracle delta）。

**门禁**（同 §7 E1/E2）：两个独立系统边界可复现错误；至少一类错误在“完整组 + 普通 dedup”后仍存在；RewardTxn 在真实与 trace 中均为 0 invalid commit；30k trace 零失败（分层口径见 RQ2 门禁）。

**产出**：`runs/E1_*/`、`TRACE_REPORT_PAPER.json`、跨栈复现文档（AReaL/TQ/Slime 分表）、`paper-e1` tag。

**回退点**：前一 tag；若 AReaL 复现失败，按 E1 门禁降级泛化 claim 后继续。

### 15.3 P2：E3 baseline + E4 消融 + S4 adapter（4–5 天）

**任务**：
1. E3：在 1.5B、K=8、U=4、4 卡配置下运行 B0–B6（短训 20–30 步；B0/B2/B5/B6 跑满 4 类故障 ×10 次，B1/B3/B4 减为 2 类故障 ×10 次）；B4/B5 以 mechanism-equivalent 实现并如实表述；配对性能运行仅对 B0/B2/B6；统一 checkpoint 频率，另给“checkpoint 开销剥离”结果；
2. E4：A0–A7 消融（确定性层每变体×故障 ≥100 次；真实系统每模块 1–2 个直接相关故障 ×10 次）；
3. S4：若 P1 AReaL 复现通过，实施最小 invariant adapter（Group Seal + StepToken + recovery 判定），并在 AReaL 上复跑 E3 的关键 cell；
4. B4/B5 策略参数预实验冻结并写入 prereg/。

**门禁**（同 §7 E3/E4）：RewardTxn 唯一覆盖全部 invariant 或明确承认等价；昂贵重算中位节省 ≥30%（B5 差异 <20% 时按预注册声明降级）；每个核心模块至少一个预注册反例。

**产出**：`CORRECTNESS_MATRIX.json`、`ABLATION_MATRIX.json`、S4 adapter 代码与证据、`paper-e2` tag。

**回退点**：前一 tag。

### 15.4 P3：E5 replay 边界 + E6 开销/规模（4–5 天）

**任务**：
1. E5：新增 `replay_boundary.py`，在 K∈{4,8,16}、U∈{1,4,8}、失效组占比∈{1/8,1/4,1/2,1}、verifier 成本（轻量/100ms/1s/5s 注入）网格下对比 whole-group retry / whole-step replay / latest-checkpoint restart / selective replay；报告收益随失效比例与 reward 成本变化的曲线；
2. E6：两级矩阵（§7 E6）——核心 cell 5 次配对运行，其余格点 1 次筛选；2/4/8 卡按 §4.2 拓扑卡号；CUDA Event 与 wall clock 分段计时；checkpoint on/off 分图；
3. 每配置 ≥120 稳定 steps（warmup 20）。

**门禁**（同 §7 E5/E6）：核心 workload 中位节省 ≥30% 且 95% CI 下界 >20%；整步全失效时收益趋近零的边界明确报告；核心 4 卡 1.5B 开销 95% CI 上界 <5%；协议开销不随卡数超线性；CAS/manifest 数据量随 committed groups 近似线性且 retention 后稳定。

**产出**：`REPLAY_BOUNDARY.json`、`OVERHEAD_MATRIX.json`、Figure 2/3/4 数据源、`paper-e3` tag。

**回退点**：前一 tag。

### 15.5 P4：E7 多 seed 训练 + E8 workload/soak（7–10 天）

**任务**：
1. E7：5 组（Clean Oracle / Clean RTX / Faulted Default / Faulted B5 / Faulted RTX）×5 seeds ×≥500 步（1.5B、K=8、U=4、4 卡），使用 `schedule_player.py` 播放预生成 fault schedule；NUMA 0/1 双 4 卡槽并行跑不同 seed；
2. E7 统计：TOST（冻结 margin）、配对 95% CI、效果量；clean 等价失败时先定位数据顺序/分布差异（§7 解释规则）；
3. E8：3 次×8 小时 soak（Poisson fault schedule，平均每 30 分钟 1 次真实故障），`soak_runner.sh` + 持续资源/CAS/checkpoint retention 记录；可选 24h 单次与 7B；
4. 每 50 步保存训练指标并跑固定评测集（附录 C.3）。

**门禁**（同 §7 E7/E8）：clean TOST 等价通过或如实降级；Faulted 组配对比较不劣于 oracle margin；soak 零 invalid commit、恢复均在 RTO 内、连续无进度 ≤10 分钟、retention 后有界；数学与代码 reward 通过相同 group-to-optimizer invariant。

**产出**：`TRAINING_EQUIV.json`、`SOAK_*.json`、Figure 5/6 与 Table 4 数据源、`paper-final` tag、归档验收（附录 E）。

**回退点**：前一 tag。

### 15.6 预算粗算（基于 Phase 1–3 实测，pilot 后校准）

实测口径：1.5B/4 卡 step latency median ≈9.2s、mean ≈50.8s（rollout-bound，wait ratio 0.53）；3B/4 卡 median ≈11.7s；0.5B 显著更低。据此：500 步单 run 估 3–7h；20–30 步短训估 0.4–1h。

<table fit-page-width="true" header-row="true">
<tr><td>阶段</td><td>主要运行量</td><td>4-GPU 槽位估算</td><td>双槽（NUMA0/1）墙钟</td><td>说明</td></tr>
<tr><td>P0</td><td>pilot 短运行 ~20 次 + CPU 回归</td><td>≈0.5 GPU-day</td><td>1–2 天</td><td>预算与 margin 冻结</td></tr>
<tr><td>P1</td><td>真实注入 ~200 runs + 30k trace（CPU）</td><td>≈3–4 GPU-day</td><td>2–3 天</td><td>E1/E2；AReaL 探针半天前置</td></tr>
<tr><td>P2</td><td>~300 runs（裁剪后）</td><td>≈4–5 GPU-day</td><td>2–3 天</td><td>E3/E4/S4；配对仅 B0/B2/B6</td></tr>
<tr><td>P3</td><td>~60 runs</td><td>≈2–3 GPU-day</td><td>1–2 天</td><td>E5/E6；E5 以 CPU/0.5B 为主</td></tr>
<tr><td>P4</td><td>25 训练 runs + 3×8h soak</td><td>≈4–6 GPU-day</td><td>3–4 天</td><td>E7/E8；双槽并行 seeds</td></tr>
<tr><td>合计</td><td>—</td><td>≈14–19 GPU-day</td><td>≈2.5–3.5 周</td><td>含排障余量；pilot 后校准</td></tr>
</table>

预算不足时的裁剪顺序（不触碰核心 baseline、消融与 seed 数）：先删 E8 24h 单次与 7B；再删 8 卡格点；再删 E6 筛选矩阵中远离门禁的格点。

---
## 附录 A 执行映射表（实验 → 脚本 → 环境变量 → 复用资产）

<table fit-page-width="true" header-row="true">
<tr><td>实验单元</td><td>入口脚本</td><td>关键环境变量</td><td>复用/新增资产</td></tr>
<tr><td>环境预检 / 冒烟</td><td>scripts/check_env.sh、scripts/smoke_test.sh</td><td>RTX_PROFILE=smoke</td><td>scripts/resource_gate.py</td></tr>
<tr><td>E1 Slime 六切点复现</td><td>scripts/day2_run.sh、scripts/phase2_run.sh</td><td>fault=skew|crm_crash|dup；RTX_FAULT_START/END（或 RTX_FAULT_WINDOWS）；RTX_SEAL=0/1 配对；RTX_GROUP_RM；RTX_SEAL_AUTO_FIX</td><td>scripts/day2_custom_rm.py、scripts/day4_inject.sh（进程 kill）</td></tr>
<tr><td>E1 TransferQueue</td><td>scripts/day3_tq_crash_probe.py（扩展）</td><td>—</td><td>.venv-tq、TransferQueue release/v0.1.10（8497a52a）</td></tr>
<tr><td>E1 AReaL R2/R3</td><td>新增 scripts/areal_repro 入口（容器内）</td><td>—</td><td>areal-runtime 镜像 + AReaL b83d1f40 源码</td></tr>
<tr><td>E2 确定性 trace</td><td>scripts/phase2_trace_runner.py（扩展至 12 切点 + oracle）</td><td>—</td><td>scripts/phase2_seal_rm.py、scripts/phase2_reconciler.py</td></tr>
<tr><td>E2 真实进程</td><td>scripts/day4_inject.sh、scripts/phase3_auto_recover.sh</td><td>RTX_KILL_AFTER_ITER、RTX_ENTRY、RTX_MAX_RETRIES</td><td>RTX_PROFILE=phase3b（保留 optimizer state）</td></tr>
<tr><td>E3 B0–B6</td><td>scripts/phase2_run.sh 包装 + 新增 RTX_BASELINE_MODE=b0…b6</td><td>RTX_SEAL / RTX_GROUP_RM / RTX_SEAL_AUTO_FIX 组合</td><td>scripts/day5_simulator.py（协议模拟，分表报告不互相代替）</td></tr>
<tr><td>E4 A0–A7</td><td>scripts/phase2_run.sh + 独立 feature flags（新增 RTX_A0…A7）</td><td>每次只关一个模块</td><td>scripts/phase2_seal_rm.py</td></tr>
<tr><td>E5 replay 边界</td><td>新增 scripts/replay_boundary.py</td><td>—</td><td>scripts/day5_g5.py 成本模型、scripts/phase2_reconciler.py</td></tr>
<tr><td>E6 开销/规模</td><td>scripts/phase2_run.sh + 吞吐采集</td><td>RTX_GPUS 拓扑卡号（§4.2）、RTX_SGLANG_CONCURRENCY=16/32/64</td><td>scripts/resource_gate.py</td></tr>
<tr><td>E7 训练语义</td><td>新增 scripts/schedule_player.py + scripts/phase2_run.sh</td><td>RTX_SCHEDULE=&lt;prereg/schedule json&gt;、RTX_SEED</td><td>scripts/phase3_auto_recover.sh</td></tr>
<tr><td>E8 soak</td><td>新增 scripts/soak_runner.sh</td><td>RTX_PROFILE=phase3b、Poisson 故障参数</td><td>scripts/checkpoint_retention.py、scripts/resource_gate.py</td></tr>
<tr><td>门禁判定</td><td>scripts/judge_gates.py（扩展）</td><td>—</td><td>scripts/phase3_gate3a.py / phase3_gate3b.py 模式</td></tr>
<tr><td>归档验收</td><td>scripts/phase3_archive.py（扩展 paper-eval）</td><td>—</td><td>artifact_manifest.json、SHA-256 校验</td></tr>
</table>

模型与数据入口：`RTX_MODEL_DIR` + `RTX_MODEL_CONFIG`（如 qwen2.5-1.5B.sh，位于 third_party/slime/scripts/models/），1.5B 需 `RTX_EXTRA_MODEL_ARGS="--rotary-base 1000000"`（patches/slime-qwen2.5-1.5b-rotary-base.patch）。

---
## 附录 B 术语表与 fault schedule schema

### B.1 协议术语（一句话定义 + 实现/证据）

<table fit-page-width="true" header-row="true">
<tr><td>术语</td><td>定义</td><td>实现/证据</td></tr>
<tr><td>Group Seal</td><td>聚合组内版本、数量与状态：一致=SEALED，不一致=ABORTED；ABORTED 不允许以混合值提交</td><td>scripts/phase2_seal_rm.py、seals.jsonl</td></tr>
<tr><td>Reward CAS</td><td>以 (logical_id, epoch, attempt) 为键的文件锁级 compare-and-set，重复写只保留一条权威记录</td><td>scripts/phase2_seal_rm.py、cas_rejects.jsonl</td></tr>
<tr><td>StepManifest</td><td>记录 checkpoint iteration、组提交范围与恢复所需 sidecar</td><td>scripts/phase2_manifest.py、manifests/</td></tr>
<tr><td>StepToken</td><td>与 checkpoint 内容采样哈希、前序 token 链绑定的持久化提交标记</td><td>manifests/step_token_*.json</td></tr>
<tr><td>Reconciler</td><td>审计 token/manifest，区分已提交/残缺/待恢复数据并给出恢复计划</td><td>scripts/phase2_reconciler.py</td></tr>
<tr><td>Selective Replay</td><td>复用已有 rollout，只重算缺失/失效 reward 并幂等重投递</td><td>replay_result.json、恢复审计报告</td></tr>
<tr><td>Trace Runner</td><td>统一切点 fixture 的确定性回归与随机化调度执行器</td><td>scripts/phase2_trace_runner.py、runs/TRACE_REPORT.json</td></tr>
<tr><td>AUTO_FIX</td><td>ABORTED 组在返回训练器前整组统一为权威 v1 重算</td><td>RTX_SEAL_AUTO_FIX=1、3A 门禁</td></tr>
<tr><td>group-RM</td><td>slime 原生组级 reward 路径；训练消费侧 0 混算保证依赖该路径</td><td>RTX_GROUP_RM=1、3A 门禁</td></tr>
</table>

### B.2 fault schedule schema（E7/E8）

```json
{
  "schedule_id": "e7-s17-faulted",
  "seed": 17,
  "stack": "slime",
  "model": "Qwen2.5-1.5B-Instruct",
  "steps": 500,
  "events": [
    {"step": 47,  "cut": "R3", "params": {"groups": [188, 191], "verifier": "v2", "digest": "d2"}},
    {"step": 102, "cut": "Q0", "params": {"after": "get_meta"}},
    {"step": 213, "cut": "C1", "params": {"ack_drop": true}},
    {"step": 318, "cut": "R2", "params": {"groups": [1272, 1275], "retry": "stale"}}
  ]
}
```

规则：
- 生成器在 P0 冻结；事件 step 与组索引写入 prereg/，同 seed 的 Faulted Default / Faulted B5 / Faulted RewardTxn 三组播放完全相同的 schedule；
- step→group 换算固定为 step = group_index // U；
- E8 soak 用 Poisson 过程（平均每 30 分钟 1 次），进程崩溃/ACK 丢失/队列故障三类轮流注入，事件追加 wall-clock 时间戳；
- 每个事件的完整记录字段见 §6 末段（phase、step、group IDs、attempt/epoch、verifier digest、checkpoint hash、StepToken、退出码、恢复决策、oracle verdict）。

---
## 附录 C 新故障注入、trace oracle、评测集与 sandbox 规格

### C.1 R4/R5/Q1/L3 注入点（P0 实现）

<table fit-page-width="true" header-row="true">
<tr><td>切点</td><td>注入动作</td><td>期望行为</td><td>实现位置</td></tr>
<tr><td>R4</td><td>同 (logical_id, epoch, attempt) 第二次写不同 digest</td><td>CAS 层标记 nondeterministic conflict，阻断提交，禁止 last-write-wins</td><td>phase2_seal_rm.py 注入钩子 + trace fixture</td></tr>
<tr><td>R5</td><td>同一组 2 个 recovery worker 并发写</td><td>唯一 CAS winner；旧 epoch 写入为 0</td><td>多进程并发测试（复用 Phase 3C 底座）+ reconciler 并发 runner</td></tr>
<tr><td>Q1</td><td>consumer get-data 后、StepManifest 前 kill learner</td><td>未提交数据最终恰好一次进入计划</td><td>day3_tq 扩展 + phase2_run.sh 窗口</td></tr>
<tr><td>L3</td><td>optimizer 返回后、checkpoint 前 kill</td><td>仅信任 durable checkpoint/token，不把内存更新当已提交</td><td>day4_inject.sh 时序窗口</td></tr>
</table>

### C.2 trace oracle 判定规则（E2）

invalid committed step 定义为以下任一：
- (a) 混版本组（revision 不一致）进入 committed GroupManifest/StepManifest；
- (b) StepToken 相对权威 step 序列出现重复或缺失；
- (c) checkpoint 内容哈希与 StepToken 绑定不符；
- (d) 错误 reward（与权威 v1 重算对拍不一致）进入 committed gradient。

自动 oracle 逐事件输出 verdict（PASS/FAIL），聚合为每切点失败率 + Wilson/Clopper–Pearson 95% 上界；恢复后的模型/optimizer state 做内容 hash、step 序列与 oracle delta 对拍。

### C.3 评测集划分（E7）

- GSM8K（DAPO 格式 7,473 条）：固定 random split → train 6,973 + eval 500；split 种子与划分文件写入 prereg/；
- DAPO-Math-17k 全量训练；数学评测用 GSM8K eval 500 题，代码评测用 HumanEval 164 题 / MBPP 378 题全量；
- eval 每 50 步执行一次；最终准确率取最后一次 eval（代码为 pass@1）。

### C.4 代码 sandbox 选型（E5/E8）

- 复用 slime 容器内确定性本地 CPU Docker sandbox（本地镜像、无网络）；
- HumanEval 164 题 / MBPP 378 题全量；超时 30s/用例；固定 CPU 线程数保证确定性；
- verifier 成本按 sandbox CPU-seconds 计量，作为 E5 的 reward 成本轴。

---
## 附录 D 风险登记表

<table fit-page-width="true" header-row="true">
<tr><td>#</td><td>风险</td><td>概率</td><td>影响</td><td>缓解</td></tr>
<tr><td>1</td><td>AReaL 训练栈复现失败</td><td>中</td><td>泛化 claim 降级</td><td>P1 半天探针先行；失败走 §12.2 降级路径</td></tr>
<tr><td>2</td><td>8 卡窗口不可得</td><td>高</td><td>Figure 3 / 规模 claim</td><td>4 卡核心结论优先；8 卡单次确认 + 明确标注（§11）</td></tr>
<tr><td>3</td><td>E3/E6 预算超支</td><td>高</td><td>排期</td><td>两级重复设计（§7）；每阶段超支 20% 触发裁剪评审（§9）</td></tr>
<tr><td>4</td><td>30k trace 运行超时</td><td>低-中</td><td>P1 延迟</td><td>pilot 校准速率（1k 次试跑）；分层并行化</td></tr>
<tr><td>5</td><td>7B pinned-memory 未解决</td><td>中</td><td>可选增强丢失</td><td>已设前置条件，不阻塞主线（§4.2）</td></tr>
<tr><td>6</td><td>TOST 等价失败</td><td>中</td><td>claim 收窄</td><td>先定位数据顺序/分布差异；如实降级为系统正确性论文（§12.2）</td></tr>
<tr><td>7</td><td>新近工作撞车</td><td>低</td><td>创新性受损</td><td>§13 每周增量查重</td></tr>
<tr><td>8</td><td>B5 同等安全且成本差 &lt;20%</td><td>低-中</td><td>Selective Replay 不作主贡献</td><td>E3 门禁已预注册该声明（§7）</td></tr>
</table>

---
## 附录 E 阶段 tag 纪律与回退流程

- **每阶段完成定义**：门禁 JSON 全绿 → 阶段文档（runs/PHASE4_*.md，含设计/实现/验证/结论/边界）→ git commit + tag（paper-e0 … paper-final）→ artifact manifest 更新；
- **门禁失败处理**：git checkout 前一 tag 回退，修复后重新进入；失败运行按 §8.3 写 exclusion log（保留原始日志并给出原因），不静默删除；
- **产出物落盘**：每阶段按 §10 数据契约保存 meta/config/events/manifests/metrics/resource/verdict/artifact_manifest；聚合报告保留 figure/table cell → exp_id 反向索引；
- **整体验收**：paper-final 冻结后运行 `scripts/phase3_archive.py verify --require-local-sources --require-final-tag --require-clean`；
- **tag 命名**：paper-e0（冻结）→ paper-e1（P1）→ paper-e2（P2）→ paper-e3（P3）→ paper-final（P4），全部在 phase3-final 链上创建，禁止直接改历史 tag。
