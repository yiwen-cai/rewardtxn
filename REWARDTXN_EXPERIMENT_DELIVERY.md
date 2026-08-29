# RewardTxn 全实验交付说明

> 本文是 RewardTxn 实验的独立交付文档。读者无需先阅读其他设计文档，即可了解实验动机、环境、执行方法、协议实现、全部阶段结果、复现入口、证据位置和已知边界。

- **项目**：RewardTxn —— 面向 RLVR/GRPO 异步训练的奖励组崩溃一致提交协议
- **实验周期**：2026-08-24 至 2026-08-29
- **目标环境**：8× NVIDIA H100 PCIe 主机
- **主力实验栈**：THUDM/slime v0.3.1
- **最终实验状态**：Phase 1 GO（7/7）、Phase 2 PASS（12/12）、Phase 3 PASS（9/9）
- **最终版本**：`phase3-final`
- **最终交付结论**：研究原型实验目标完成；后续规模增强或生产化工作应另立 Phase 4，不属于本交付的未完成项。

---

## 1. 交付范围与最终结论

### 1.1 交付范围

本次实验验证并实现了以下完整链路：

1. 证明异步 RLVR/GRPO 栈在奖励版本混算、消费后崩溃、Learner 崩溃和 checkpoint/ACK 窗口中存在真实错误；
2. 实现 Group Seal、Reward CAS、StepManifest、StepToken、Reconciler、Selective Replay 和 Trace Runner；
3. 使混版本奖励在训练消费前被检测并统一为权威 verifier v1；
4. 在 checkpoint 存在时实现训练进程崩溃后的无人干预自动恢复；
5. 通过长程收敛、模型升级、多进程 CAS、资源保护和全切点回归验证协议闭环；
6. 将实验数据、门禁结果、运行日志、复现脚本和清理策略封存。

### 1.2 最终结论

- **问题成立**：观察到跨版本混算和陈旧结果混入造成的静默 reward 污染，以及队列和训练进程崩溃造成的数据丢失/恢复死锁。
- **协议可行**：RewardTxn 将混算组变为显式 `ABORTED`，只重算受影响 reward，不重生成整个 rollout。
- **消费侧闭合**：真实训练中，注入组返回给训练器的 reward 与权威 v1 重算值 0 不一致。
- **恢复闭合**：kill-9 后自动识别已提交 checkpoint、自动重启并继续训练，已提交步骤不重复计算。
- **性能结果**：确定性元数据开销约 0.39%；Selective Replay 相对整步重跑节省约 92.4%；新配置 100 步 loss 对齐 MAE 约 0.006466。
- **阶段结论**：Phase 3 已完成并归档。8 卡 4+4、大模型、多 seed 和生产接入是后续增强方向，而非本阶段缺口。

---

## 2. 实验问题与协议模型

### 2.1 被验证的问题

异步 RLVR/GRPO 训练通常包含 rollout、reward/verifier、队列消费、Learner 更新和 checkpoint。各环节之间存在并发、重试和崩溃窗口。如果同一组样本的 reward 不是同一版本、样本被消费后进程死亡，或训练状态与队列 ACK 不一致，系统可能：

- 在组完整的情况下静默接受混版本 reward；
- 丢弃残缺组但仍继续训练；
- 消费标记已写入而数据无法重新取回；
- 已提交训练步骤重复 apply 或未被识别；
- 恢复时因 scheduler 参数不一致而死锁或空转。

GRPO 的组归一化会放大单个错误 reward 对同组 advantage 和后续梯度的影响，因此需要“组级原子提交”而不是仅做单样本字段校验。

### 2.2 固定实验语义

| 符号/字段 | 含义 |
|---|---|
| `K=8` | 每个 reward 组包含 8 个样本 |
| `U=4` | 每个训练 batch 包含 4 个组 |
| `global_batch_size=32` | 每步训练样本数 |
| `seed=42` | 主要实验随机种子 |
| `lr=1e-4` | 学习率 |
| `kl_loss_coef=0.01` | KL loss 系数 |
| `v1` | 权威 verifier 版本 |
| `v2` / `retry-stale` | 注入的不同版本或陈旧结果 |
| `SEALED` | 组内版本一致，可进入正常协议流 |
| `ABORTED` | 组内版本不一致或需要恢复，不允许以混合值提交 |
| `logical_id` | `(group_index, index, rollout_id)`，CAS 去重键 |
| `StepToken` | 与 checkpoint 内容哈希绑定的持久化提交标记 |

最终生产训练消费侧要求使用 slime 原生 `--group-rm` 路径。非 group-RM 单样本路径可以记录和修正持久化审计流，但无法撤回已经返回给训练器的单样本值。

---

## 3. 实验环境与资产

### 3.1 硬件和主机

- 8× NVIDIA H100 PCIe，单卡约 80GiB 显存；
- Docker、CUDA GPU runtime、Ray、SGLang 可用；
- 目标主机存在外部任务动态占用，因此正式实验主要采用 3/4 卡布局，而不是强行占用 8 卡；
- `/public` 和根盘均通过启动资源门禁检查；训练期间持续记录主机内存、磁盘和 GPU 状态。

### 3.2 版本锁定

| 组件 | 版本/提交 | 用途 |
|---|---|---|
| slime | v0.3.1，`a6272da0d4f3d0a08520c99a2f3b4f6c887960dc` | 主力 RLVR/GRPO 训练栈 |
| slime 镜像 | `slimerl/slime:v0.3.1`，digest 以 `sha256:2feaad36...` 开头 | 容器训练环境 |
| AReaL | `b83d1f40196e5bd7d9f83092563443561870d550` | B2 机制复核和对照 |
| TransferQueue | release/v0.1.10，`8497a52a5c4347c4d67c97f7ce500e544426a08a` | 队列消费崩溃窗口 |
| Python CPU 环境 | `.venv-tq` | TransferQueue 与 Phase 0 工具 |

完整版本锁定记录见 `STACKS.lock`；网络受限时使用的镜像和手工导入策略记录在 `ENVIRONMENT_SETUP.md`。

### 3.3 模型和数据

- `Qwen2.5-0.5B-Instruct`：主要长程和新配置实验，已转换为 torch_dist；
- `Qwen2.5-1.5B-Instruct`：Phase 1 正式配置、3A 消费侧实验、3C 模型升级复验；
- `Qwen2.5-3B-Instruct`：Phase 2D 端到端替代模型；
- `Qwen2.5-7B-Instruct`：已下载并转换，但宿主多进程 pinned memory 不稳定，未作为主门禁模型；
- `Qwen3-4B`：HF 到 slime 转换存在上游 shape mismatch，未作为主门禁模型；
- `dapo-math-17k`：17,398 条，主训练数据；
- GSM8K：转换为 DAPO 格式后约 7,473 条，用于 Phase 1 正式配置。

1.5B 模型运行所需的 `rotary-base 10000→1000000` 是模型配置兼容修正，不是 RewardTxn 协议或训练循环改动，补丁已保存为 `patches/slime-qwen2.5-1.5b-rotary-base.patch`。

---

## 4. 分阶段执行与结果

## 4.1 环境预检和 Phase 0

环境预检完成：GPU runtime、Docker、三栈导入、模型/数据挂载和 slime 冒烟均验证通过。

`scripts/phase0_diff_oracle.py` 提供 CPU 差分 Oracle，用于演示组内单样本 reward 扰动如何改变 advantage、梯度和参数更新；运行它需要当前 Python 环境已有 PyTorch。仓库没有单独封存 K∈{4,8,16} 的 Phase 0 完整数据包；实际项目采用 Phase 1 的真实 reward 注入和 g3 受控梯度实验作为更强的主证据：参数更新 Delta L2≈0.2598、梯度余弦相似度≈0.7548。该状态已在交付文档中如实注明，不把未单独封存的 Phase 0 数据冒充为已完成门禁。

## 4.2 Phase 1：Failure Probe（GO 7/7）

目标是证明问题真实存在，并为协议设计提供基线。

| 时间/实验 | 故障窗口 | 关键观察 |
|---|---|---|
| Day 1 | B0 干净基线 | 0.5B、20 步正常完成，建立 lineage 和基础吞吐 |
| Day 2 | R1/R2/R3 reward/verifier | R3 跨版本混算完全静默，组偏移约 -0.2562；R2 陈旧结果混入完全静默，组偏移约 -0.2750；R1 RM 异常导致 32 样本丢失但任务仍可成功结束 |
| Day 3 | Q0/Q1/L0 | TransferQueue `get_meta` 后崩溃，32 样本无法重取；Learner actor 崩溃后 Ray 不自动重启，已完成步骤丢失 |
| Day 4 | L2/L3/C1/C2/B5 | scheduler 恢复参数死锁、fully-async 轮次无法续接；无 StepToken 时无法区分已提交/未提交；逐步 checkpoint 保存约占 step 时间 65%–100% |
| Day 5 | B0–B5 矩阵 | 协议语义模拟器得到 0 Invalid Commit、2.54% 协议开销；Selective Replay 相比 B2/B5 节省约 85.9%/95.0% |
| 补充 | 配置对齐、R3 重复、loose 模式 | 1.5B 干净基线完成；R3 strict 偏移 -0.2625，loose 偏移 -0.2437，结论可重复且不依赖单一 v2 构造 |

Phase 1 最终裁决为 **GO 7/7**。证据包括 `runs/PHASE1_SUMMARY.md`、`runs/DAY5_SUMMARY.md`、`runs/DAY5_MATRIX.json`、`runs/DAY5_VERDICT.json`、`runs/DAY5_SIMULATOR.json` 和 `runs/DAY5_G5.json`。

### Phase 1 故障注入定义

| 类别 | 注入位置和动作 | 失败语义 |
|---|---|---|
| R1 | reward/verifier 窗口内抛 `RuntimeError` | slime fully-async 将组静默丢弃，残缺 batch 仍可能训练 |
| R2 | 重试/重复消费返回陈旧 reward | 陈旧值与当前值混入同一组，默认栈没有版本栅栏 |
| R3 | 同组前 4 条使用 v1、后 4 条使用 v2 | 组仍完整但版本混合，默认栈完全静默接受 |
| Q0 | TransferQueue `get_meta` 后已 mark-consumed，随即 SIGKILL | 重启后 `Available: 0`，32 个样本永久不可重取 |
| L0 | kill slime 的 MegatronTrainRayActor | Ray 报 `ActorDiedError`，无自动重启和 checkpoint 恢复 |
| L2 | optimizer 阶段 kill trainer 后尝试 resume | scheduler 参数断言与 fully-async 轮次续接共同造成死锁/空转 |
| C1/C2 | checkpoint/ACK 窗口 kill 并重启 | 没有 Durable StepToken 时无法可靠区分已提交和未提交步 |

### B0–B5 基线为何不足

| 基线 | 能覆盖的内容 | 在本实验中的缺口 |
|---|---|---|
| B0 默认开源栈 | 正常训练 | R2/R3 混算静默，R1/Q0/L0/L2 存在丢失或恢复问题 |
| B1 组 ID/大小校验 | 残缺组 | 完整的混版本组仍然穿透 |
| B2 丢弃 incomplete group | 不完整组安全丢弃 | 混版本组是完整的；且丢组会浪费全部 rollout |
| B3 幂等去重 | 防止重复消费 | 不处理消费后崩溃窗口 |
| B4 Occupy/Consume | 基本占用/消费状态 | 无 timeout/reclaim，窗口仍可丢数据 |
| B5 每步 checkpoint | 粗粒度恢复 | 保存占 step 时间约 65%–100%，恢复通常整步/全 rollout 重跑 |
| RewardTxn | 版本 Seal、持久提交、选择性重放 | 本实验的目标方案；进入 Phase 2/3 实测 |

### 实验期间的关键校准和修复

这些修复用于让故障实验测到真实 reward，而不是数据或模型配置错误：

- GSM8K 转换补上每条 user message 的 `role` 字段；
- 0.5B 竞赛题有效信号不足，正式 Day 2 起换用 1.5B；
- deepscaler v1 兼容无 `</think>`/`###Response` 分隔符的 Qwen response，直接从完整 response 提取答案；
- Qwen2.5-1.5B 的 `rotary-base` 按 HF `rope_theta=1000000` 修正；
- custom-RM 使用 `模块.函数` 形式挂载，运行时将 `/workspace/scripts` 加入 Ray Python path；
- 容器使用独立 64GiB shm，Ray spill/CAS 小索引放本地 scratch，并按可用 GPU 动态选择卡号。

## 4.3 Phase 2：最小事务协议（PASS 12/12）

Phase 2 实现七个协议模块，每个子阶段均有门禁、报告和 tag。

| 阶段 | 交付 | 验证结果 |
|---|---|---|
| 2A | Group Seal + Reward CAS | 注入组 20/20 检出；干净组 1,245/1,245 放行；确定性开销 0.39% + 约 21µs/样本 |
| 2B | StepManifest + StepToken | 5 个 checkpoint sidecar 完整；崩溃后识别已提交 `[3,7]`；token 对内容变化敏感且幂等 |
| 2C | Reconciler + Selective Replay + Trace Runner | R1/Q0/L2 恢复计划成立；节省 92.7%–100%；全切点语义回归通过 |
| 2D | 端到端训练评测 | Qwen2.5-3B 4 卡完成；恢复决策 1.48s；3B/1.5B loss 差均小于 0.05 |

由于 8 卡窗口、7B pinned memory 和 Qwen3-4B 转换器问题，2D 使用 Qwen2.5-3B 作为同族替代，并将 8 卡 4+4 设为门禁外可选项。完整结果见 `runs/PHASE2_SUMMARY.md` 和 `runs/PHASE2_FINAL.md`；版本点为 `phase2-final`。

## 4.4 Phase 3：正确性、可用性和规模化（PASS 9/9）

### 3A：消费侧正确性

实现 `RTX_GROUP_RM=1` 的组级 reward 调用和 `RTX_SEAL_AUTO_FIX=1`：检测到混版本后，整组在返回训练器前统一重算为 v1。

- 1.5B 注入实验：20/20 窗口组 `ABORTED + autofix`；11,392 条训练消费样本相对 v1 为 0 mismatch；
- 干净实验：13,600 条样本无误改、0 autofix；同一完整组 AUTO_FIX on/off 配对 8/8 一致；
- 同为 group-RM 的注入/干净对比吞吐下降 2.78%，低于 5% 门槛。

### 3B：自动恢复

`phase3_auto_recover.sh` 在训练失败后读取 latest checkpoint、记录恢复历史、自动重启并使用 `--load + --override-opt-param-scheduler` 绕过 slime/Megatron scheduler 恢复死锁。

- iter9 checkpoint 落盘后真实 kill-9；
- 自动识别 `committed_step=9`；
- 自动重启并完成 step10–19；
- 已提交 step0–9 各出现一次；
- 恢复延迟 202s，内存/IO 优化复验 148s，均小于 5 分钟。

### 3C：长程、升级和回归

- 0.5B 新配置 Seal 100 步 vs 历史同 Seal 100 步：逐步 loss 平均绝对差 **0.006466 < 0.05**；
- 新 100 步运行均值 0.006333，后 50 步均值 0.003331；
- 1.5B skew 20 步：20/20 组 autofix，4,296 条 reward 全部为 v1；
- 4 进程并发同组 CAS：恰好 1 份权威记录；
- 资源保护：SGLang concurrency 64、Ray object store 16GiB、SQLite CAS、checkpoint 滚动保留和 resource gate 均在线验证。

Phase 3 门禁文件为 `runs/PHASE3_GATE3A.json`、`runs/PHASE3_GATE3B.json`、`runs/PHASE3_GATE3C.json`；一键回归结果为 `runs/PHASE3_REGRESSION.json`。

---

## 5. 实现架构和数据流

```text
Rollout response
      │
      ▼
Group-RM 批处理 ──> Reward CAS（logical_id 幂等）
      │
      ▼
Group Seal：版本一致？
   ┌──┴──────────────┐
   │                 │
 SEALED            ABORTED
   │                 │
 原值放行      AUTO_FIX：整组权威 v1 重算
   │                 │
   └──────┬──────────┘
          ▼
   sample.reward 进入训练器
          │
          ▼
 checkpoint 保存
          │
          ▼
 StepManifest + StepToken（提交步/内容哈希/前序链）
          │
          ▼
 崩溃后 Reconciler
   ┌──────┴──────────────┐
   │                     │
 已提交 checkpoint       未提交/异常组
 不重复 apply             只重算 reward 或重新投递
          │
          ▼
 自动恢复入口：load + scheduler override + 继续训练
```

设计原则：协议逻辑通过 custom-RM、环境变量、checkpoint sidecar 和 entrypoint wrapper 接入，不改 slime 的协议训练循环源码；唯一的 1.5B 配置兼容 patch 单独归档并明确标注。

### 七个模块的职责

| 模块 | 职责 | 主要实现/证据 |
|---|---|---|
| Group Seal | 聚合组内版本、数量和状态，`SEALED` 或 `ABORTED` | `scripts/phase2_seal_rm.py`、`seals.jsonl` |
| Reward CAS | 以 logical ID 做文件锁级 compare-and-set，重复写只保留一条权威记录 | `phase2_seal_rm.py`、CAS 回归 |
| StepManifest | 记录 checkpoint iteration、组提交范围和恢复所需 sidecar | `scripts/phase2_manifest.py` |
| StepToken | 将提交步与 checkpoint 内容采样哈希、前序 token 链绑定 | `manifests/step_token_*.json` |
| Reconciler | 审计 token/manifest，区分已提交、残缺和待恢复数据 | `scripts/phase2_reconciler.py` |
| Selective Replay | 复用已有 rollout，仅重算缺失/失效 reward 并幂等重投递 | `replay_result.json`、恢复审计报告 |
| Trace Runner | 用统一切点 fixture 回归 R1/R2/R3/Q0/L0/L2 语义 | `scripts/phase2_trace_runner.py`、`runs/TRACE_REPORT.json` |

---

## 6. 复现和验收操作

### 6.1 预检

```bash
cd /public/home/caiyiwen/rewardtxn
bash scripts/check_env.sh

# Phase 3 CPU 门禁/归档所需依赖（已有环境可跳过）
python3 -m pip install -r requirements-phase3-archive.txt
# phase0_diff_oracle.py 另需当前 Python 环境已有 PyTorch（可在 slime 容器内运行）
```

启动新实验前必须确保目标 GPU、`/public` 和根盘通过 resource gate；每次使用新的 `RTX_EXP_ID`，不要复用已有运行目录。

### 6.2 冒烟和 Phase 0

```bash
bash scripts/smoke_test.sh
.venv-tq/bin/python scripts/phase0_diff_oracle.py
```

冒烟会验证 Ray、SGLang、CAS、Seal、checkpoint 和 retention 链路。Phase 0 脚本仅是 CPU 差分演示，不替代 Phase 1 真实故障实验。

### 6.3 Phase 1/2 训练入口

```bash
# 每次复现使用新的目录名；以下变量只是示例，可自行替换
EXP_ID=rtx-repro-$(date +%Y%m%d-%H%M%S)

# Day 1 干净基线
bash scripts/day1_run.sh "${EXP_ID}-day1"

# Day 2 reward 故障；窗口使用 group_index，step = group_index // 4
RTX_MODEL_DIR=/root/models/Qwen2.5-1.5B-Instruct \
RTX_MODEL_CONFIG=qwen2.5-1.5B.sh \
RTX_EXTRA_MODEL_ARGS="--rotary-base 1000000" \
bash scripts/day2_run.sh skew 20 39 20 "${EXP_ID}-day2"

# Phase 2 Seal/CAS 包装层
RTX_CUSTOM_RM=phase2_seal_rm.rm_function \
RTX_SEAL=1 RTX_GROUP_RM=1 RTX_SEAL_AUTO_FIX=1 \
bash scripts/phase2_run.sh skew 20 39 20 "${EXP_ID}-phase2"
```

`fault` 可选 `none`、`skew`、`crm_crash`、`dup`；脚本会自动生成 `meta.json`、训练日志、reward/seal 审计和 checkpoint 目录。

### 6.4 Phase 3 自动恢复和回归

```bash
# 受控恢复实验：故障发生后自动关闭注入并续跑
RTX_PROFILE=phase3b \
RTX_CUSTOM_RM=phase2_seal_rm.rm_function \
RTX_SEAL=1 RTX_GROUP_RM=1 RTX_SEAL_AUTO_FIX=1 \
RTX_ENTRY=phase3_auto_recover.sh RTX_KILL_AFTER_ITER=9 \
bash scripts/phase2_run.sh none -1 -1 20 "${EXP_ID}-phase3b"

# GPU 无关的全切点回归
bash scripts/phase3_regress.sh
```

回归覆盖 R1/R2/R3/Q0/L0/L2，以及 Seal AUTO_FIX、AUTO_FIX on/off 配对、CAS 单/多进程、Reconciler、Replay、Manifest、零修改和恢复历史。

### 6.5 交付归档验收

```bash
# 重新生成收敛证据和 SHA-256 归档索引
python3 scripts/phase3_archive.py write

# 校验 portable evidence、本地长程日志和 canonical checkpoint 清单
python3 scripts/phase3_archive.py verify --require-local-sources

# 最终版本校验：tag 指向当前提交且工作区干净
python3 scripts/phase3_archive.py verify \
  --require-local-sources --require-final-tag --require-clean
```

---

## 7. 交付物清单

### 7.1 最终报告和门禁

- `REWARDTXN_EXPERIMENT_DELIVERY.md`：本文，整个实验的自包含交付说明；
- `runs/PHASE1_SUMMARY.md`、`runs/PHASE1_SUPPLEMENT.md`：Failure Probe 总结；
- `runs/PHASE2_SUMMARY.md`、`runs/PHASE2_FINAL.md`：最小事务协议总结；
- `runs/PHASE3_3A.md`、`runs/PHASE3_3B.md`、`runs/PHASE3_3C.md`：Phase 3 分阶段报告；
- `runs/PHASE3_FINAL.md`：Phase 3 结项报告；
- `runs/DAY2_SUMMARY.md`、`runs/DAY3_SUMMARY.md`、`runs/DAY4_SUMMARY.md`、`runs/DAY5_SUMMARY.md`：逐日实验记录；
- `runs/PHASE2_GATE2A.json`、`runs/PHASE2_GATE2B.json`、`runs/PHASE2_GATE2C.json`、`runs/PHASE2_GATE2D.json`：Phase 2 四阶段门禁；
- `runs/PHASE3_GATE3A.json`、`runs/PHASE3_GATE3B.json`、`runs/PHASE3_GATE3C.json`：Phase 3 三阶段门禁；
- `runs/PHASE3_REGRESSION.json`：最终 9/9 回归；
- `runs/TRACE_REPORT.json`：Phase 2 Trace Runner 证据。

### 7.2 实现脚本

- `scripts/phase2_seal_rm.py`：Seal、CAS、AUTO_FIX、group-RM；
- `scripts/phase2_manifest.py`：StepToken/Manifest；
- `scripts/phase2_reconciler.py`：恢复计划和 Selective Replay；
- `scripts/phase2_trace_runner.py`：原始协议切点回归；
- `scripts/phase3_auto_recover.sh`：自动恢复入口；
- `scripts/phase3_regress.sh`：R1/R2/R3/Q0/L0/L2 回归；
- `scripts/phase3_gate3a.py`、`scripts/phase3_gate3b.py`：门禁判定；
- `scripts/resource_gate.py`：启动和训练期间资源健康检查；
- `scripts/checkpoint_retention.py`：checkpoint 滚动保留；
- `scripts/phase3_archive.py`：归档生成和完整性验证。

### 7.3 证据和归档策略

`runs/PHASE3_ARCHIVE_MANIFEST.json` 当前记录：

- **123 个 portable evidence 文件**：报告、门禁 JSON、脚本、审计 JSONL 和训练日志，进入 Git 并绑定 SHA-256；数量与该 manifest 的 `portable_evidence` 字段一致；
- **13 个 canonical runs**：代表 3A/3B/3C 的关键成功实验；
- canonical checkpoint 的本地路径、iteration、文件数、字节数和 inventory hash；
- 5 个失败/被替代运行已删除，不参与门禁。

大 checkpoint 不进入 Git。当前 `runs/` 约 81G，其中约 80G 是仍保留的 canonical checkpoint，用于审计或恢复；Phase 3 CPU 回归不依赖这些大文件。

---

## 8. 已知限制和解释边界

1. **Phase 0**：没有单独封存 K=4/8/16 完整数据包；其目的由 Phase 1 真实注入和 g3 受控梯度证据覆盖。
2. **GPU 规模**：受外部任务占用影响，主实验实际使用 3/4 卡；8 卡 4+4 是门禁外增强项。
3. **模型规模**：Qwen2.5-3B 用于 Phase 2D，1.5B 用于 Phase 3 升级；7B pinned memory 和 Qwen3-4B 转换器问题已记录，不能把替代实验描述成 Qwen3-4B/7B 实测。
4. **长程口径**：100 步收敛对齐是 0.5B 新配置 Seal 与历史同 Seal 配置；1.5B 实验是 20 步协议升级复验，不是 1.5B 的 100 步 clean 对照。
5. **吞吐测量**：端到端运行环境噪声较大；0.39% 是确定性协议元数据开销，2.78% 是同 group-RM 模式注入/干净运行的观测吞吐差，不宣称一次运行可以精确测出完整系统 overhead。
6. **单样本 RM**：非 group-RM 路径返回值不可撤回，因此训练消费侧 0 混算保证依赖 group-RM。
7. **恢复边界**：本轮实弹主要验证 checkpoint 已落盘后的 resume；checkpoint 前崩溃走冷启动和协议记录重放，代价可能包含未提交步骤重训。
8. **生产状态**：当前是经过门禁验证的研究原型，不等同于已经完成生产插件化、告警、CI 和长期 soak test。

---

## 9. 最终交接

**Phase 3 无剩余必做实验。** 实验方案已经结束，最终版本由 `phase3-final` 标识。当前没有正在运行的训练任务，工作区应保持干净。

如果继续工作，应另立 Phase 4 计划，明确是：

- **论文增强**：8 卡、大模型、多 seed、低噪声 benchmark、置信区间；或
- **生产化**：正式插件/API、TransferQueue 服务级回滚、告警、CI、长期稳定性测试。

这些方向不应回填为 Phase 3 的“未完成实验”。
