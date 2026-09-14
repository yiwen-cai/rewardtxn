# R 回退排查：异步／批同步最小对照方案

2026-09-14。状态：**方案草案，尚未实现或启动**。本次仅准备实验；参数见同目录 [design.json](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-async-sync-plan-20260914/design.json)。

## 要回答的问题

**R 相对 Oracle 的准确率差距，是否会在取消跨批生成／训练重叠后缩小？** 现有公开结果尚不足以确认“R 稳定回退”；日志中平均滞后接近，也不能排除异步执行对完整学习轨迹的影响。本轮直接干预执行方式，筛查这条机制。

同步会一起改变版本新鲜度、实际并发和消费顺序，因此本轮识别的是“方法 × 执行方式”的交互，不能单独归因 SQLite、滞后或某一种排序。旧 LITE−R 确认最后三轮继续暂缓，不是本方案前置任务。

## 最小规模与固定条件

| 单元 | 奖励实现 | 执行方式 | 训练 seed | 完整训练 |
|---|---|---|---:|---:|
| OA | Oracle（O） | 原 fully async | 193 | 500 步 |
| RA | RewardTxn（R） | 原 fully async | 193 | 500 步 |
| OS | Oracle（O） | 批同步 | 193 | 500 步 |
| RS | RewardTxn（R） | 批同步 | 193 | 500 步 |

- 拟用新训练 seed193；已查的 pilot、初筛、原确认、Oracle 重复和 SQLite 清单中未使用。**数据／rollout seed 仍为 42，评估 seed 仍为 29**，三个种子用途不混用。
- 四次串行、各从同一个 Qwen2.5-0.5B-Instruct 基座重新启动。初始列表 `[OA, RA, OS, RS]` 用独立顺序种子 `20260914`、Python `random.Random(...).shuffle` 一次抽签，结果恰为 **OA → RA → OS → RS**，不重抽。单个顺序不能排除时段效应。
- 同一最终源码／镜像／观测补丁、训练集、提示模板、存储位置、优化器与保存策略。GPU 固定物理 **1/2/3/4**，1 为 actor、2/3/4 为 rollout；未来预检绑定 UUID，不沿用历史空闲快照。
- 沿用已保存实际参数：每步 4 题 × 每题 8 回答，global batch32；lr1e−6、constant、warmup10、每步更新 rollout 权重，500 步日程；temperature1、top_p1、响应上限1024、单引擎并发上限24、每50步保存并保留全部 checkpoint。其余共同参数与来源哈希列在 `design.json`。
- O/R 保持各自原奖励路径；R 保留磁盘 SQLite、事务、CAS/Seal 和原日志。本轮不叠加 DBM/LITE，不改奖励、优势、PPO、采样确定性配置或数据标签。
- 固定**题目源的打乱与派发序列**；异步按原规则接收完成结果，同步每次取下一批4组并沿用训练前的组内／批内整理规则。不同执行方式可以消费不同顺序、产生不同回答，这正是待观察路径。相同 seed 不保证逐 token 相同。
- 唯一质量终点：`iter_0000499_hf`，原 validation100，greedy，seed29、batch8、max_new_tokens2048；评估配置完全相同，不使用 reserved test500。

## 批同步必须真的做到什么

每批严格执行：**全部 rollout 引擎确认当前权重 → 派发4组 → 生成及原 RM 全部完成 → 消费这32条并训练一步 → 更新权重并等待全部引擎确认 → 下一批**。

1. 同时关闭 driver 的下一批预启动和后台 worker 的跨批预取；同步运行从空队列、新 worker 开始。
2. 优化器更新前，本批生成／RM 已结束，生成队列和在途任务均为零；不得靠丢弃额外预生成回答伪装成同步。
3. 下一批请求必须晚于所有引擎的权重确认。按实际确认的版本编号校验，不假定版本号一定等于训练步号；所有本批回答的版本记录须一致。
4. 版本字段目前在回答结束时记录，并非逐 token 记录；“无跨版本在途生成”须由派发、屏障和权重确认事件共同证明。任一屏障无法证明或被违反，按技术失败处理。

实现应在**同一训练 driver** 中增加小范围执行分支，复用 `generate_and_rm_group`、数据源、训练与退出逻辑，异步分支保留原行为。仅切到 `train.py` 不够：它没有消除所选 fully-async rollout 的常驻生成线程，还会引入其他生命周期差异。也不能直接换用未经核对的默认 batch rollout 函数，其取消／补采样逻辑不同。

## 最少要留下的观测

- **每步沿用完整消费批记录和 actor 聚合指标**：题目／组／样本 ID、回答及 token、奖励、长度、行为 logprob、消费位置；PG loss、KL、clip 比例、训练与行为 logprob 差、梯度范数。记录训练／生成／RM 耗时和队列状态。
- **补足时序链**：请求派发与完成、消费、更新及各引擎确认事件，关联实际版本与样本 ID；时间基准须可比较，不能直接混用不同进程不明来源的时钟。同步每批保存零在途屏障证据。
- **固定第 0、1、49、249、499 步**保存对齐的输入／响应 token、mask、优势、行为／参考 logprob，以及 PPO loss 实际 forward 使用的训练 logprob和样本—微批映射。直接保存该 forward 的 detached 数据，不另做前向；不保存全量梯度或逐步模型副本。
- 四组观测代码和采样步一致；异步分支不因日志新增等待屏障。`save-debug-train-data` 单独开启不能保证拿到实际训练 forward 的 logprob，须检查真正落盘内容。最终公共观测开销沿用既有5%门槛，未实测不能宣称无扰动。

## 事前固定的读数与解释

准确率统一用百分数，差值单位为百分点（pp）：

```text
ΔA = Acc(RA) − Acc(OA)                  原异步下的 R−O
ΔS = Acc(RS) − Acc(OS)                  批同步下的 R−O
I  = ΔS − ΔA = RS − OS − RA + OA        差距是否随同步缩小
```

同时报告 `RS−RA` 和 `OS−OA`、四组完整时序／学习信号摘要，不只挑主指标。

| 观察 | 本轮允许的解释 |
|---|---|
| ΔA ≥ 0 | 该 seed 未复现异步下 R 的负差；不能据此判断回退机制。 |
| ΔA < 0 且 I > 0 | 与“执行方式影响 R−O 差距”方向一致；ΔS 仍负为部分缩小，ΔS 非负也不等于证明两者等效。 |
| 上述差距缩小，但 RS 没有高于 RA | 必须写明可能是 Oracle 下降，不能称 R 恢复。 |
| ΔA < 0 且 I ≤ 0 | 本轮未观察到同步缓解，不构成正式排除该机制。 |
| 任一单元／同步屏障失败 | 四格对照不完整，保留现场，不作完整交互结论。 |

这是一个 seed 的筛查：不设显著性／等效性结论，不把100道评估题或500个训练步当作独立训练重复；不能确认稳定回退、唯一根因或长期收益。四组都完成后统一比较，不按前两组质量提前停或改设计。即使出现正向交互，后续确认仍需另定方案，不自动追加 seed。

## 预算、实施顺序与完成条件

1. **先实现并验证控制流程**：新隔离副本中实现双层停止预取、权重屏障与少量观测；用可控 CPU 任务验证无跨批在途生成、组数／顺序守恒和异步路径兼容，再记录最终源码哈希。此处尚未实施。
2. **拟做四组各10步工程预检**：检查实际版本、屏障、32条／步、训练 forward 数据、成功退出与导出；仍按500步优化器日程，不进行质量评估。预检模型不用于正式训练，完整训练重新从基座开始。预检也属于训练成本，不能记成零。
3. **验收后才考虑完整四轮**：四组有效参数除方法、模式和产物标识外一致；绑定模型／数据／镜像／源码／GPU UUID，独立新结果目录。每轮需要500步消费、成功退出、终点导出哈希及100题评估凭证；失败即停，保留全部产物，不自动重试、换卡、换 seed 或补跑旧确认。

本方案新增完整训练预算为 **4×500 = 2,000 步、64,000 条消费回答、400 条终点评估回答**；另列预检 **4×10 = 40 步、1,280 条消费回答**。异步尾部预生成会使实际生成量更多。同步耗时尚无实测，先用未来预检估算，不按四轮异步耗时直接承诺总时长；全量 checkpoint、日志及预生成尾部空间也在启动前实测预算，保留既有200GiB空间余量要求。

候选起点 `isolated_v2_driver_exit` 仍有独立复审／公共开销／同版本 smoke 缺口，**不是可运行放行版本**。本轮短程验收不替代旧候选门禁，旧门禁所需工作的成本另列；SQLite 单次可行性例外不自动迁移。本次只完成方案，未提交可运行启动器，未启动任何预检或训练。

## 已核查的实现依据

- [driver 提前启动下一批](/public/home/caiyiwen/rewardtxn/third_party/slime/train_async.py:32)与[常驻后台 worker](/public/home/caiyiwen/rewardtxn/third_party/slime/slime/rollout/fully_async_rollout.py:100)：同步需同时处理两层预取。
- [原组生成／评分函数](/public/home/caiyiwen/rewardtxn/third_party/slime/slime/rollout/sglang_rollout.py:295)与[回答结束时追加版本](/public/home/caiyiwen/rewardtxn/third_party/slime/slime/utils/types.py:408)：复用语义，补足屏障证明。
- [已有训练数据保存](/public/home/caiyiwen/rewardtxn/third_party/slime/slime/backends/megatron_utils/actor.py:524)与[实际训练 logprob](/public/home/caiyiwen/rewardtxn/third_party/slime/slime/backends/megatron_utils/loss.py:912)：两者并非自动包含关系。
- [现有可重复性证据](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-reproducibility-20260914/REPORT.md)及[候选验收边界](/public/home/caiyiwen/rewardtxn/docs/experiments/rewardtxn-ablation-20260911/implementation/isolated_v2_driver_exit/CPU_RESULTS.md)。来源文件哈希记录在 `design.json`，属于方案编写时的快照，不能充当尚未完成实现的运行冻结。
