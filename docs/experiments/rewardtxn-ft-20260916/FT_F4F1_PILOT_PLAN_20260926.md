# F4/F1 试点方案 + 统一 RTO 口径（2026-09-26，待审计）

用户决定（2026-09-26）：
- 同意统一两臂 RTO 口径；
- F4/F1 **先做试点**，暂不跑 20+10 对的完整矩阵；
- R 使用当前 HEAD（lag-1 版本，见 [阶段 2 门控](PERF_GATE2_RESULT_20260926.md)）。

试点只用于工程验证与参数冻结，不计入正式样本。

## 1. 统一 RTO 定义

**RTO = 故障信号时刻 → 受影响工作首次被正确且持久地纳入该臂恢复权威的时刻**（均为宿主 monotonic 时钟）。

| 臂 | 恢复权威 | "持久纳入"时刻 |
|---|---|---|
| A+R | R 的已提交链（token） | 首个使目标行全部进入保留链的 `committed` 事件，含 `via=recovery`。现有实现不变 |
| A | AReaL 原生 recover checkpoint（DCP + recover_info） | 故障后首个在训练中消费了目标行的更新，其 recover checkpoint 的异步写盘被 finalize 成功返回的时刻；如果 recover_info 写得更晚，取两者较晚者 |

**目标行**
- F4/F1：冻结的 `source_row_id=5518` 组（与现有契约一致）；
- F2：被杀更新的 4 个 source row（现有推导）。

**A 端需要补的观测**
- 在两臂共用的观测器（`areal_pilot_hooks.install_hooks`）里，对 `AsyncCallsQueue.schedule_async_request` / `maybe_finalize_async_calls` 做类级包装，发出 `dcp_scheduled`（call_id、pid）和 `dcp_finalized`（call_ids、pid）。
- R 在实例上另有一层包装，调用会穿过类级包装，两臂事件一致。
- 每个 `checkpoint_save_start` 按同一进程内的顺序配对到其后第一个 `dcp_scheduled`。
- 目标行到更新的映射沿用 `finalize_ft1_fault` 中 F2 的方法：`batch_taken` 的批次指纹 → 生成指纹 → source row。

**`finalize_ft1_fault` 的变化**
- A 分支的端点从"最终 checkpoint 上界"改为上表口径，`persistence_time_kind` 两臂都记为 `durable_commit`，`formal_rto_eligible=true`；
- 任一臂缺观测时，记为不可判定并停跑，不回退到上界。

**影响说明**
- lag-1 下 R 的端点晚一步，这是如实计量，不做修正；
- A 的 finalize 可能被推迟到下一次保存时的非阻塞 reap，这同样如实计入。

## 2. 最小流程接入 F1/F4

- `run_ft_minimal.minimal_config`：
  - 接受 `F1`/`F4`，重试次数与现有 FT1 故障配置一致（`retries: 1`）；
  - F1 仍复制 `ft1-f1-target.json`（`run_training_fault` 已支持）。
- 故障钩子沿用 `ft1_fault_hooks.contract`：
  - F4：目标组 8 条回答全部生成后，第 4 次评分执行开始时杀掉评分进程；
  - F1：生成引擎在至少一条回答完成后被杀。
- 未命中切点时记为 `fault_cut_missed`，属技术无效，按冻结规则最多补做一次。
- `run_ft_formal` 的 `perf_gate` 分支改为通用的 `pilot` 类型：非正式样本，场景 ⊆ {F1, F2, F4, no_fault}。

## 3. 试点冻结草案

| 顺位 | 场景 | 对数 | 顺序 |
|---|---|---:|---|
| 1–2 | F4 | 2 | A→R、R→A |
| 3–4 | F1 | 2 | R→A、A→R |

- 种子：`random.Random(20260927)` 从未用过的种子中抽 4 个。
- 每 run 10 次更新，其余训练参数与 2026-09-24 正式冻结相同。
- 预算约 8 GPU·h。
- 试点判据：
  1. 切点命中（`valid_hit`）；
  2. 两臂验收通过，分类可判定；
  3. 两臂 RTO 都能按 §1 测出；
  4. F4 两臂对目标组的"重生成 / 重评分"次数可从证据中计数，作为恢复代价的分解依据。
- 试点**不做**统计检验，也不报告 R 对 A 的节省比例，只报告能否测量以及数值量级，用于决定正式矩阵的规模与参数。

## 4. 需要审计的风险
1. **F4 可能被原生评分重试完全吸收**（严格评分每个 wrapper 最多 2 次尝试）：此时两臂都只重算 1 条评分，RTO 差异趋于零。这属于如实结果，但会影响正式矩阵是否值得跑，试点要能识别出这种情况。
2. **F1 的最小流程接入**：F1 会杀生成引擎，AReaL 需要重启推理服务。10 步内能否恢复并训练到目标行，还要看 900 秒观察窗口是否够用。
3. **A 的 finalize 观测只在主进程内做类级包装**：要确认 DCP 在 fork 出的 writer 中不会走另一条路径，也不会被 R 的实例包装绕过。
4. **测量口径不对称**：R 的端点是 token 写入，A 的端点是 finalize 返回，两者都是本臂恢复所依赖的最后一步持久化，但开销组成不同，报告中要分开列出。

## 5. 流程
本方案 → 独立审计 → 用户批准 → 实现与 CPU 验证 → 试点冻结 → GPU 试点（约 8 GPU·h）→ 报告后再决定正式矩阵。

## 6. v1.1：落实[审计](FT_F4F1_PILOT_PLAN_AUDIT.md) 8 条必改项（F4 设计另待用户决定）

1. **A 端点**：改称"首次瞬时完整（下界）"。另外记录两项：A 在每次 save 后覆盖写同一目录的不可恢复窗口；故障发生时是否正处于该窗口。窗口内的样本单独分类。口径写明：持久 = 对进程崩溃持久（A 的 step_info 没有 fsync）。
2. **类级包装**：在 R 的 `attach` 之前安装并加断言；只在返回的 call 列表非空时发事件；因为两臂共用的观测器变了，需要登记新冻结。
3. **save 与写盘调度的配对**：用 `meta.path` 或线程局部的 save_id 显式绑定，不按出现顺序配对。
4. **目标行映射**：读取所有 pid 的观测文件，断言 5518 组的 8 条都能映射到。R 端的端点改为沿 token 链累计。
5. **F4**：判据增加"训练器是否重启、评分重试了几次"；预先写定：如果被原生重试吸收，F4 只做无回归检查，不进入 RTO 比较。
6. **F1**：
   - 结果分为"被吸收 / 整组重启 / 挂死"三类，挂死设 900 s 超时；
   - 冻结时只挑 5518 会在故障后 10 步内被消费的种子；
   - 确认重启时会产生 job-lifecycle 回执。
7. **接入清单补全**：`run_ft_minimal`（场景白名单、F2 专用检查）、`areal_pilot_hooks`、`check_ft1_fault`、`check_ft1_chain`/`check_training_*` 在 F4/F1 下的回归、`finalize_ft1_fault` 新字段，以及相应的 CPU 测试。
8. **决策规则与预算上限**：试点最多 4 对、10 GPU·h；技术无效每对最多补做一次；超出预算即停。

## 7. v1.2：F4 改为"评分进行中杀训练进程"（F4'，用户 2026-09-26 决定）

**动机**：审计确认，原 F4（杀评分子进程）会被严格评分的原生重试吸收，两臂都只多评一次，测到的不是恢复能力。

**F4' 切点**
- 在两臂共用的父进程包装 `RLVRWorkflow._compute_rewards` 返回处计数。目标组 `source_row_id=5518` 的 8 条回答都已生成，且第 4 条的评分结果已返回到 workflow（父进程）时，trainer 客户端发出 ready，控制器 SIGKILL 整个 trainer 进程。rollout 与评分都在 trainer 进程内。
- 实现复用 F2 的 trainer 目标（`Client('trainer')`，在 install 时注册）。契约改为：`event_id=ft1-f4t-trainer`、`target=trainer`，evidence 为 `{phase: 'fourth_target_score_returned', source_row_id: 5518, k: 8, ordinal: 4}`。
- 原 F4 契约保留，不删除。
- 两臂杀的是同一类进程、同一个逻辑切点，损失范围相同：当前所有在途的生成与评分全部丢失，训练器重启一次（`retries: 1`）。

**对称性说明（如实登记）**
- 切点处，R 可能已持久化前 3 条的 reward 阶段产物和若干条 response，第 4 条的 reward 在切点之后才写，因此随进程丢失。
- A 不持久化任何 rollout 中间结果。
- R 的优势正是来自这种差异，属于被测对象。

**RTO 与目标行**
- 目标行：5518 组。
- A 端点：§1 与 §6.1 的"首次瞬时完整"下界。
- R 端点：沿 token 链累计，首个使目标组进入保留链的提交。

**恢复代价分解**：两臂分别统计目标组在故障后的重生成次数、重评分次数和采纳次数（R 的 adoption）。

**试点判据补充**
- `valid_hit`、trainer 确实重启；
- 两臂都能测出端点；
- R 至少采纳 1 条先前的 response 或 reward。若 R 的采纳次数为 0，报告原因（例如 max_head_offpolicyness 使旧回答过期），不进入正式矩阵。

**不变的部分**：F1 各项、试点规模（F4' 2 对 + F1 2 对、上限 10 GPU·h）、决策规则。

## 8. v1.3：主指标修订与实现要点（用户 2026-09-26 批准）

- **主指标**：F4' 改用"故障时已完成但未训练的工作中，有多少以同一物理评分执行进入最终保留链（被保留 / 被丢弃）"，与 F2 同类。RTO 按 §1 与 §6.1 只作描述。依据见[离线复用率预估](F4_ADOPTION_ESTIMATE_20260926.md)：R 约 96 条在途已入账样本中，约 98% 可复用；A 按加载器位置跳过这些样本。
- **目标行改为 2602**：采样器种子固定为 0（`DistributedSampler` 默认值），批次组成与训练种子无关；5518 永远落在第 2 批，task_id 为 0，太早。2602 在第 6 批（task_id 20，训练第 6 步），处于稳态。F1 仍用 5518，其 task_id=0 的断言成立。审计 v1.2 的"按种子筛选 2≤s≤8"因此由固定目标行满足。
- **实现**：
  - `ft1_fault_hooks`：F4T 契约与 `install_f4t`，只计成功返回，要求 8 条已进入评分，已触发后的重启不再布防；
  - `areal_pilot_hooks`：类级 `dcp_scheduled`/`dcp_finalized` 观测，用 `meta.path` 显式配对；R 在 `_wrap_queue` 中断言它已先安装；
  - `check_ft1_fault` 增加 F4T 分支，并记录被杀时主线程所处阶段和 ready→signal 的时间差；
  - `finalize_ft1_fault` 使用统一端点：R 只计保留链内的提交，A 取"首次瞬时完整"下界；
  - `check_ft_minimal_source.verify_inflight` 计算在途已完成工作的保留与丢弃；
  - `run_ft_minimal`/`run_ft_formal` 接入 F1、F4T 与 `pilot` 类型。
- **CPU**：新增 `test_f4t_hook` 4 项（触发、未全部生成时判为 missed、重启后不布防、非目标行不触发）。容器全量回归与阶段 2 一致。
- **未覆盖**：新验收字段在真实证据上的运行，由 GPU 试点完成。
