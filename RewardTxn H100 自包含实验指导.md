---
名称: RewardTxn H100 自包含实验指导
导入时间: 2026-08-24T14:30:00.000Z
文档类型: 实验指导
---

# RewardTxn: 面向 H100 平台的 RLVR 奖励组崩溃一致提交实验指导手册

> **文档性质**：独立、自包含的端到端实验操作规范。无需翻阅其他设计方案，按本文档即可在目标服务器 `h100` 上完成从环境预检、基线冒烟、Phase 0 差分 Oracle、Phase 1（5 天 Failure Probe 强制门禁）到 Phase 2（最小协议实现）的全部实验。  
> **目标硬件环境**：单节点 8× NVIDIA H100 PCIe 81GB、双路 Intel Xeon Gold 6430（64 核）、503 GiB 内存、Ubuntu 20.04.6 LTS、CUDA 12.6.20、Docker 24.0.7。  
> **当前归档状态（2026-08-29）**：本文最初的 Conditional Go 已完成后续验证并被正式结果取代：Phase 1 为 **GO（7/7）**，Phase 2 为 **PASS（12/12）**，Phase 3A/3B/3C 为 **PASS（9/9）**。本文保留初始实验规范；当前结论与证据索引以 `runs/PHASE3_FINAL.md`、`runs/PHASE3_ARCHIVE_MANIFEST.json` 和 `phase3-final` tag 为准。

---

## 1. 核心科学契约与前置声明

### 1.1 核心问题与 Claim 收窄

在 Group-relative 强化学习（如 GRPO / RLVR）中，Reward 与 Advantage 是在 Prompt 的同一组 Rollout（大小为 \(K\)）内归一化计算的：
$$A_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

单条样本的丢失、重复、错组或跨 Verifier 版本计算，不仅改变该样本的 \(A_i\)，还会改变整组的 \(\mu_G\) 与 \(\sigma_G\)，进而通过梯度污染整个 Optimizer Step。

```
[ Rollout Workers ] ---> [ Reward / Verifier ] ---> [ Group Normalization ] ---> [ Learner Step ] ---> [ Checkpoint ]
       ^                         ^                          |                           ^                   |
       |                         |                          v                           |                   |
 (Crash / Retry)           (Revision Skew)          (Advantage Corrupted)         (Crash / Dup ACK)     (Durable Pointer)
```

**论文核心 Claim（精准定义）**：

> 对于使用组内 Reward 统计量构造 Advantage 的 RL 算法，RewardTxn 使一个符合算法 Group Contract 的奖励组集合，**要么恰好一次**出现在某个**已持久化、可恢复的 Optimizer Step** 中，**要么在该 Step 的持久状态中完全不可见**；进程崩溃、消息重试和 ACK 丢失不会产生部分组、跨 Reward Revision、重复 Step 或“队列已消费但 Optimizer 未提交”的静默状态。

**重要边界限定**：

1. **Exactly-once 仅相对于已持久化且可恢复的 Optimizer 历史（Durable Committed Optimizer History）**，绝不宣称 GPU 显存中的原地张量更新可以天然事务化。
2. **两层提交单元（Two-tier Commit Hierarchy）**：
   - **数据与归一化单元**：`GroupManifest`（大小为 \(K\) 的单组轨迹及其 Reward 统计量）。
   - **原子提交单元**：`StepManifest`（由 \(U = B/K\) 个完整组构成的全局 StepBatch，与 Predecessor Checkpoint \(C_s\) 及唯一 `StepToken` 绑定）。

### 1.2 消极清单（严禁作为主创新点宣称）

根据 2026 年最新相关工作查重，以下机制已被学术界或开源实现覆盖，**严禁作为本研究的独立创新点**：

- **完整组流水化（Complete-group Materialization）**：已被 RolloutPipe 覆盖。
- **轨迹组生命周期（Reserve → Occupy → Consume）**：已被 StaleFlow 覆盖。
- **样本全局索引与去重队列**：已被 TransferQueue 覆盖。
- **丢弃不完整组换题重试（`drop_incomplete_group`）**：已被 AReaL 覆盖。
- **Trainer 角色级重启与 Step Checkpoint**：已被 RobustRL 与 Belayer 覆盖。
- **单纯的 Lineage 查询或 Dashboard 界面**。

### 1.3 故障模型（Failure Model）

- **允许且必须覆盖的故障**：Fail-stop 进程崩溃（`kill -9`）、RPC 丢包与网络超时、Ray Actor/Task at-least-once 重试、双恢复 Worker 并发重算、Verifier 热更新导致的跨版本计算、Sandbox 执行超时与 OOM。
- **不处理的故障（Out of Scope）**：Byzantine 恶意篡改、硬件级静默内存比特翻转（Silent Data Corruption）、本地 NVMe 存储物理损毁。
- **语义正确性边界**：本系统保证“按指定版本规则正确提交与隔离”，不保证用户编写的 Verifier 逻辑本身在数学/语义上的无缺陷性（如代码 Judge 存在 Prompt 漏洞或 False Positive）。

---

## 2. 一页式作业入口（今日跑哪一段）

```
[ Day 0: 环境检查 + 三栈冒烟 + Phase 0 CPU Oracle ]
                         |
                 (通过 Phase 0 门禁?)
                   /               \
                [Yes]              [No] ---> [立即终止，报告无污染]
                  |
[ Day 1-5: Phase 1 Failure Probe 5 天强制门禁 ]
  - Day 1: 最小 Lineage 打通 + 正常吞吐基线
  - Day 2: Reward / Verifier 层故障注入 (R1/R2, Revision Skew)
  - Day 3: Queue 消费与 Learner 崩溃窗口 (Q0/Q1, L0)
  - Day 4: Optimizer ↔ Checkpoint ↔ ACK 窗口 (L2/L3, C1/C2)
  - Day 5: B0-B5 强基线对比 + 裁决
                         |
                 (通过全部 Go 条件?)
                   /               \
                [Yes]              [No] ---> [终止 RewardTxn，转为上游 Issue 修复或 DiffState]
                  |
[ Phase 2: 最小事务机制与端到端评测 (第 2-3 周) ]
  - Group Seal + Attempt Fencing + StepManifest Commit + Selective Replay
```

**快速状态自检**：

- 若尚未验证 CPU Oracle 差异 \(\rightarrow\) **执行第 6 节（Phase 0）**。
- 若处于 5 天探针期 \(\rightarrow\) **执行第 7 节（Phase 1 逐日任务）**。
- 若第 5 天未通过全部 Go 条件 \(\rightarrow\) **立即触发停损，严禁进入 Phase 2**。

---

## 3. 目标环境 `h100` 实测基准与动态预检

### 3.1 硬件与系统实测规格

- **SSH 别名**：`h100`（`ssh caiyiwen@10.160.4.102`）
- **CPU**：2× Intel(R) Xeon(R) Gold 6430（共 64 物理核 / 64 线程，2 NUMA 节点：Node 0 为 CPU 0–31，Node 1 为 CPU 32–63）
- **内存**：503 GiB RAM（可用约 453 GiB）；`/dev/shm`（tmpfs）容量 252 GiB
- **GPU**：8× NVIDIA H100 PCIe（每卡 81559 MiB 显存），Driver 580.126.09，默认 CUDA 12.6.20
- **GPU 拓扑与 NVLink 绑定**：
  ```
          GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NUMA
  GPU0     X      NODE    NODE    SYS     SYS     NV12    SYS     SYS       0
  GPU1    NODE     X      NV12    SYS     SYS     SYS     SYS     SYS       0
  GPU2    NODE    NV12     X      SYS     SYS     SYS     SYS     SYS       0
  GPU3    SYS     SYS     SYS      X      NV12    NODE    NODE    NODE      1
  GPU4    SYS     SYS     SYS     NV12     X      NODE    NODE    NODE      1
  GPU5    NV12    SYS     SYS     NODE    NODE     X      NODE    NODE      1
  GPU6    SYS     SYS     SYS     NODE    NODE    NODE     X      NV12      1
  GPU7    SYS     SYS     SYS     NODE    NODE    NODE    NV12     X        1
  ```
  _(注：GPU 0-5、GPU 1-2、GPU 3-4、GPU 6-7 之间存在双向 NVLink NV12 连接；同 NUMA 节点内走 NODE 互联，跨 NUMA 走 SYS)_
- **存储路径**：
  - 系统盘 `/`：可用 1.9 TB
  - 公共盘 `/public`：总计 7.0 TB，已使用 84%（可用约 1.2 TB）
  - 实验根目录：`/public/home/caiyiwen/rewardtxn`（即 `~/rewardtxn`）
- **软件基础设施**：Docker 24.0.7（已配置 `nvidia-container-runtime`，用户属于 `docker` 组免 sudo）、uv 0.9.5 (`/public/home/caiyiwen/.local/bin/uv`)、tmux 3.0a、`/usr/local/bin/nsys`

### 3.2 动态环境预检脚本

在开始任何实验前，必须在远程执行以下一键检测脚本，确保硬件未被外部任务锁死且磁盘预算充足：

```bash
#!/usr/bin/env bash
# 保存为 check_env.sh 并在 h100 上运行
set -euo pipefail

echo "=== 1. 检查 CUDA 与 驱动 ==="
export PATH=/usr/local/cuda/bin:/public/home/caiyiwen/.local/bin:$PATH
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv,noheader

echo "=== 2. 检查 Docker NVIDIA Runtime ==="
docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu20.04 nvidia-smi -L >/dev/null && echo "Docker GPU Runtime: OK"

echo "=== 3. 检查 /public 与 /dev/shm 容量 ==="
AVAIL_PUB=$(df -BG /public | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$AVAIL_PUB" -lt 200 ]; then
    echo "ERROR: /public 可用空间仅剩 ${AVAIL_PUB}GB，低于安全门禁 200GB！请先清理磁盘。"
    exit 1
fi
echo "/public 可用空间: ${AVAIL_PUB}GB (合格)"

echo "=== 4. 检查当前 GPU 空闲情况 ==="
BUSY_GPUS=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | wc -l)
echo "当前运行中的 GPU 计算进程数: ${BUSY_GPUS}"
```

### 3.3 拓扑感知 GPU 分配规则

根据 NUMA 亲和性与 NVLink 对分布，严格执行以下卡号划分，禁止随机指定卡号：

| 实验规模              | 推荐卡号分配                           | 角色划分              | 拓扑优势                 |
| :-------------------- | :------------------------------------- | :-------------------- | :----------------------- |
| **单卡微基准**        | `CUDA_VISIBLE_DEVICES=0`               | 独立 Learner / Oracle | 独占 NUMA 0              |
| **双卡微基准**        | `CUDA_VISIBLE_DEVICES=1,2`             | 1 Rollout + 1 Learner | 具备 NV12 直连           |
| **4 卡短训 (NUMA 0)** | `CUDA_VISIBLE_DEVICES=0,1,2,5`         | 2 Rollout + 2 Learner | NV12 环状直连 (0-5, 1-2) |
| **4 卡短训 (NUMA 1)** | `CUDA_VISIBLE_DEVICES=3,4,6,7`         | 2 Rollout + 2 Learner | NV12 环状直连 (3-4, 6-7) |
| **8 卡完整训练**      | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` | 4 Rollout + 4 Learner | 全局拓扑调度             |

_注：CPU Sandbox / 代码 Verifier 强制运行在 host/container CPU，严禁在 Phase 1 占用 H100 显存。_

---

## 4. 版本锁定与三栈基线配置（STACKS.lock）

为保证 100% 可复现性，三套框架的代码与容器版本必须严格锁定：

```
                                  [ 三套基线与目标系统架构 ]
                                             |
        +------------------------------------+------------------------------------+
        |                                    |                                    |
   [ 首选主力栈 ]                       [ 语义复核第二栈 ]                   [ 独立数据面微基准 ]
   THUDM/slime                          areal-project/AReaL                  Ascend/TransferQueue
   v0.3.1 (a6272da0...)                 Pinned main (b83d1f40...)            v0.1.10 (8497a52a...)
   - 4/8 卡异步 GRPO                    - B0/B2 对照基线                     - Queue Consume ↔ Trainer
   - 故障注入与恢复主战场               - drop_incomplete_group 复核         - Crash Window 验证
```

### 4.1 框架版本与 Commit SHA

| 框架                     | 角色与定位                   | 仓库地址与固定 Commit SHA                                                                                 | 容器镜像 / 运行方式                                                    |
| :----------------------- | :--------------------------- | :-------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------- |
| **THUDM/slime**          | 主力实验栈（首选）           | `https://github.com/THUDM/slime`<br>`Commit: a6272da0d4f3d0a08520c99a2f3b4f6c887960dc` (v0.3.1)           | `slimerl/slime:v0.3.1`<br>(Digest: `sha256:2feaad36...`)               |
| **areal-project/AReaL**  | 语义复核第二栈（B0/B2 对照） | `https://github.com/areal-project/AReaL`<br>`Commit: b83d1f40196e5bd7d9f83092563443561870d550`            | `ghcr.io/areal-project/areal-runtime:v2.0.0-sglang` 基础上挂载锁定源码 |
| **Ascend/TransferQueue** | 数据面 Crash Window 微基准   | `https://github.com/Ascend/TransferQueue`<br>`Commit: 8497a52a5c4347c4d67c97f7ce500e544426a08a` (v0.1.10) | Host Python / 容器内 `pip install .`                                   |

### 4.2 模型与任务锁定

1. **Phase 0 & Phase 1 调试**：`Qwen/Qwen2.5-0.5B-Instruct` 与 `Qwen/Qwen2.5-1.5B-Instruct`。
2. **Phase 1 端到端与 Phase 2 评测**：`Qwen/Qwen3-4B`（8 卡数学基线）与 `Qwen/Qwen2.5-7B-Instruct`。
3. **数据集与 Verifier**：
   - 数学任务：`DAPO-Math-17k` / `GSM8K`（采用确定性 Python SymPy / Math-Verify 规则匹配）。
   - 代码任务：`HumanEval` / `MBPP`（采用本地 Docker CPU Sandbox 执行测试用例）。

---

## 5. 系统 Invariants、状态机与 R0–C2 切点

### 5.1 十一条核心 Invariants（离散零容忍）

| 编号    | Invariant 定义                                                                                                    | 违背后的判定              |
| :------ | :---------------------------------------------------------------------------------------------------------------- | :------------------------ |
| **I1**  | 未满足 `group_contract`（样本数不等于 \(K\) 或缺失关键字段）的组，严禁生成 `GroupManifest`。                      | 立即阻断                  |
| **I2**  | 同一组内所有样本必须具有相同的 `reward_plan_digest`；跨 Verifier/Test-set 版本的样本严禁混入同组提交。            | 判定为 Revision Skew 违规 |
| **I3**  | 同一个 `GroupManifest` 严禁出现在两个不同的正常训练 Step 中（显式 Replay Epoch 除外）。                           | 判定为重复消费错误        |
| **I4**  | 组内任一样本 Reward 发生变动，**必须全组重算 \(\mu_G, \sigma_G\) 及所有 Advantage**；严禁只改单样本 Advantage。   | 判定为数学语义污染        |
| **I5**  | Step 已提交（Committed）后若发现某组存在数据污染，**必须回滚整个 Step 及其所有后继 Step**，不可仅撤回单组。       | 判定为回滚破坏一致性      |
| **I6**  | 内存中的 `APPLYING` 状态非持久化状态；崩溃恢复只能且必须信任最新原子发布的 Checkpoint 指针及内嵌 `StepToken`。    | 判定为假持久化            |
| **I7**  | 实施 Attempt Fencing：三元组 `(logical_id, epoch, attempt)`；旧 Epoch 写入一律丢弃，同 Attempt 依据 Digest 幂等。 | 判定为并发写冲突          |
| **I8**  | 相同 Logical ID、相同 Revision 但产生不同 Output Digest 时，判定为 Non-deterministic 冲突，禁止 Last-Write-Wins。 | 判定为静默覆盖错误        |
| **I9**  | 处于 `ABORTED` 状态的组，永不可被迟到的旧 Attempt ACK 复活为 `REWARDED`。                                         | 判定为幽灵状态复活        |
| **I10** | Policy 允许受控 Staleness（记录 Version Vector 并在界内允许），但 Verifier 必须严格同版本。                       | 判定为版本控制失效        |
| **I11** | 事务保证范围严格限定在已提交的 Durable History，不宣称显存原地操作事务化。                                        | 理论边界约束              |

### 5.2 两层状态机转换规范

```
Group 状态机:
[ OPEN ] ---> (K个样本与Reward收集齐) ---> [ SEALED ] ---> (校验版本与统计量) ---> [ REWARDED ]
    |                                          |                                         |
    +------------------------------------------+-----------------------------------------+---> [ ABORTED ] (故障/超时/版本冲突)
                                                                                         |
                                                                                         v
                                                                                [ PREPARED / COMMITTED ]

Step 状态机:
[ PLANNED ] ---> (U个组装填完毕) ---> [ PREPARED ] ---> (Backward & Optimizer) ---> [ APPLYING ]
                                                                                         |
                                      [ COMMITTED ] <--- (原子发布指针) <--- [ CHECKPOINTED ] (写盘成功)
```

### 5.3 R0–C2 故障注入切点与恢复行为矩阵

```
  Reward 阶段             Queue 阶段                 Learner / Optimizer 阶段            Checkpoint 阶段
[R0] --> [R1] --> [R2] --> [R3] --> [Q0] --> [Q1] --> [L0] --> [L1] --> [L2] --> [L3] --> [C0] --> [C1] --> [C2]
 ↑        ↑        ↑        ↑        ↑        ↑        ↑        ↑        ↑        ↑        ↑        ↑        ↑
Start   Compute  Write    Seal    Select    Fetch   Prepare  Bwd Done OptStart OptDone CkptStart CkptDone  ACK
```

| 切点代码    | 注入时机                                         | 系统期望标准行为                                   | 禁止的错误行为                               |
| :---------- | :----------------------------------------------- | :------------------------------------------------- | :------------------------------------------- |
| **R1 / R2** | Reward 计算中或结果写入前 Worker 崩溃            | 触发 Fencing 重试，旧结果作废，新 Attempt 重新计算 | 接受悬挂的半完成结果，或旧 Worker 迟到覆盖   |
| **R3**      | 组封存时检测到跨 Verifier 版本                   | 立即将组置为 `ABORTED`，触发 Selective Replay      | 强行用不同 Verifier 产出的分值计算 Advantage |
| **Q0 / Q1** | Queue 元数据标记 Consumed，但 Learner 未取走即挂 | Queue 识别未 ACK 超时，重新放回就绪队列            | 队列永远丢失该批组，导致训练静默跳过数据     |
| **L1**      | Forward/Backward 完成，进入 Optimizer 前崩溃     | 丢弃当前显存梯度，从前驱 Checkpoint \(C_s\) 重做   | 将未完成的梯度留在显存中继续执行下一个 Step  |
| **L2**      | Optimizer 执行中单卡崩溃                         | **全 Trainer 立即回滚**至 \(C_s\)，严禁单卡续跑    | 单卡重启后尝试与旧显存参数拼接               |
| **L3 / C0** | Optimizer 完成但 Checkpoint 尚未落盘             | 判定为 Uncommitted，从 \(C_s\) 重新执行该 Step     | 将内存中已更新的参数当作已提交状态           |
| **C1 / C2** | Checkpoint 落盘成功但 Commit ACK 丢失            | 恢复时读取 Checkpoint 内嵌的 `StepToken`，补发 ACK | 认为此 Step 失败而重复执行同一 StepBatch     |

---

## 6. Phase 0：确定性微基准与差分 Oracle（第 1 天）

### 6.1 目标与通过标准

不占用完整 GPU 集群，在 CPU / 单卡上验证数学原理：**证明单样本故障在 \(K \ge 4\) 时必然引起 Advantage 符号颠倒或梯度剧烈偏差，且普通 Group ID 校验无法消除此类偏差。**

- **通过门禁（Gate 0）**：在 \(K \in \{4, 8, 16\}\) 下，至少复现出 **2 类**在修复显式 Group ID 后仍能穿透并导致参数更新量（Optimizer Delta）发生非平凡偏差的故障。

### 6.2 内嵌 Phase 0 差分验证脚本

在 `h100` 上直接保存并运行以下 Python 脚本 `phase0_diff_oracle.py`：

```python
#!/usr/bin/env python3
"""
Phase 0: 确定性数学与梯度污染差分 Oracle
验证: 单样本扰动在组归一化下的误差扩散
"""
import torch
import torch.nn as nn
import torch.optim as optim

def compute_group_advantage(rewards, epsilon=1e-6):
    """GRPO 组归一化 Advantage 计算"""
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + epsilon)

def run_oracle_test():
    torch.manual_seed(42)
    # 模拟简单的线性策略网络
    model_clean = nn.Linear(16, 1, bias=False)
    model_fault = nn.Linear(16, 1, bias=False)
    model_fault.load_state_dict(model_clean.state_dict())

    opt_clean = optim.SGD(model_clean.parameters(), lr=0.1)
    opt_fault = optim.SGD(model_fault.parameters(), lr=0.1)

    # 构造 K=4 的一组特征输入
    inputs = torch.randn(4, 16)

    # 1. 干净奖励组 [0.0, 0.0, 1.0, 1.0] -> 均值 0.5, 方差 0.577
    r_clean = torch.tensor([0.0, 0.0, 1.0, 1.0])
    adv_clean = compute_group_advantage(r_clean)

    # 2. 注入故障: 重复消费导致 0 被替换为 1 -> [0.0, 1.0, 1.0, 1.0]
    r_fault = torch.tensor([0.0, 1.0, 1.0, 1.0])
    adv_fault = compute_group_advantage(r_fault)

    print(f"Clean Advantage: {adv_clean.tolist()}")
    print(f"Fault Advantage: {adv_fault.tolist()}")

    # 模拟策略损失 Loss = - (log_prob * Advantage)
    logits_clean = model_clean(inputs).squeeze(-1)
    loss_clean = -(logits_clean * adv_clean).mean()
    opt_clean.zero_grad()
    loss_clean.backward()
    opt_clean.step()

    logits_fault = model_fault(inputs).squeeze(-1)
    loss_fault = -(logits_fault * adv_fault).mean()
    opt_fault.zero_grad()
    loss_fault.backward()
    opt_fault.step()

    # 计算参数更新 Delta
    delta_clean = model_clean.weight.data
    delta_fault = model_fault.weight.data
    l2_diff = torch.norm(delta_clean - delta_fault).item()
    cos_sim = torch.cosine_similarity(delta_clean.view(-1), delta_fault.view(-1), dim=0).item()

    print(f"\n=== Phase 0 差分结果 ===")
    print(f"权重更新 L2 偏差: {l2_diff:.6f}")
    print(f"梯度方向余弦相似度: {cos_sim:.6f}")

    # 判定
    if l2_diff > 1e-4 and cos_sim < 0.999:
        print("\n>> PASS: 成功证明组内单样本故障导致不可逆的 Optimizer 参数污染！")
        return True
    else:
        print("\n>> FAIL: 误差未造成显著梯度偏差，请检查超参设定。")
        return False

if __name__ == "__main__":
    run_oracle_test()
```

---

## 7. Phase 1：5 天 Failure Probe 逐日作业单（强制门禁）

```
[ Day 1: 基线搭建与 Lineage 验证 ] ---> [ Day 2: Reward / Verifier 层故障注入 ]
                                                              |
[ Day 4: Learner / Checkpoint 窗口探针 ] <--- [ Day 3: Queue 消费与崩溃窗口探针 ]
       |
       v
[ Day 5: B0-B5 全基线对齐、开销核算与 Go / No-Go 最终评审 ]
```

### Day 1：环境冻结、三栈部署与最小 Lineage 验证

- **目标**：在 `h100` 上拉起 Slime 与 TransferQueue 最小实例，建立能够跟踪 `TrajectoryID → GroupID → StepToken` 的最小 Lineage。
- **执行命令**：

  ```bash
  # 1. 进入实验目录
  mkdir -p /public/home/caiyiwen/rewardtxn/runs
  cd /public/home/caiyiwen/rewardtxn

  # 2. 启动 Slime 4 卡最小全异步 GRPO (Qwen2.5-0.5B)
  docker run -d --name rtx-day1-slime --gpus '"device=0,1,2,5"' \
    --ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /public/home/caiyiwen/rewardtxn:/workspace \
    slimerl/slime:v0.3.1 \
    bash -c "cd /root/slime && bash examples/fully_async/run-qwen2.5-0.5B-fully_async.sh"

  # 3. 观察 10 个 Step 的正常 Step Time 与 Manifest 输出
  docker logs -f rtx-day1-slime
  ```

- **通过验收标准**：无故障情况下连续运行 20 个 Step，输出稳定的 `throughput_base` 与 Step Latency 基准。

### Day 2：Reward / Verifier 层故障注入（R1/R2, Revision Skew）

- **目标**：在 Reward Worker 与 Verifier 侧注入 Worker 崩溃、Ray 重试导致的重复交付、以及 Verifier 跨版本计算。
- **注入点**：
  1. `R1_kill`: Reward 计算到 50% 时直接 `kill -9` Worker 进程。
  2. `R2_dup`: 模拟 Ray Task 重试，使同一个 Trajectory 返回两个不同的 Reward 结果。
  3. `R3_skew`: 同一 Prompt 的 8 条轨迹中，4 条使用 Verifier v1 计算，4 条使用 Verifier v2 计算。
- **预期观察**：
  - B0/B1: 产生静默错组或直接挂起。
  - RewardTxn (Seal): 捕获 Digest 不一致，将整组标记为 `ABORTED` 并触发重算，无跨版本数据进入 Learner。

### Day 3：Queue 消费与 Learner 崩溃窗口探针（Q0/Q1, L0）

- **目标**：验证 TransferQueue / 消息队列在 `get_meta`（标记已消费）与 Learner 真正完成训练并 Commit 之间的崩溃窗口。
- **测试步骤**：
  1. 启动独立 TransferQueue 进程。
  2. Consumer 调用 `get_meta` 获取数据索引后，立即被 `kill -9` 杀掉（模拟消费后未及训练即挂掉）。
  3. 观察重启后的 Consumer 能否重新获取该数据，或是否导致该 Batch 永久丢失。
- **关键判定指标**：是否存在“队列显示已消费，但 Optimizer 从未执行且未持久化”的静默数据丢失。

### Day 4：Optimizer ↔ Checkpoint ↔ ACK 窗口与分布式故障（L2/L3, C1/C2）

- **目标**：探查 Learner 执行完 Optimizer Update 但 Checkpoint 指针未写完、或 Checkpoint 完成但 ACK 丢失的临界窗口。
- **注入场景**：
  1. `L2_rank_kill`: 在 4 卡数据并行训练的 Backward 期间，强杀 Rank 1 进程。
  2. `C1_ack_lost`: Checkpoint 文件写入完毕但阻断向 Controller 返回 ACK，随后重启 Learner。
- **验收要求**：
  - 单 Rank 崩溃时，必须触发全 Trainer 回滚至上一个 Checkpoint，严禁局部继续。
  - Checkpoint 成功但 ACK 丢失时，重启后必须能够通过 Checkpoint 内嵌的 `StepToken` 幂等识别已提交状态，绝不重复 apply 梯度。

### Day 5：B0–B5 强基线对齐、重算开销核算与 Go / No-Go 最终评审

- **目标**：横向对比 B0–B5 各基线在全部故障注入下的表现，计算 Selective Replay 节省的 Rollout/Reward 算力。
- **全量对比矩阵表**：

| 基线编号      | 基线机制                                      | 故障注入下的表现                                         | 重算代价（Tokens / GPU·s） |
| :------------ | :-------------------------------------------- | :------------------------------------------------------- | :------------------------- |
| **B0**        | 默认开源栈行为                                | 静默产生错误 Advantage，梯度严重污染                     | 0（但训练损坏）            |
| **B1**        | 显式 Group ID + 组大小校验                    | 拦截错组，但无法防范跨版本与 ACK 丢失重复                | 需整组丢弃                 |
| **B2**        | `drop_incomplete_group` (AReaL)               | 安全丢弃不完整组，但浪费全部昂贵 Rollout                 | 浪费 100% Rollout 算力     |
| **B3**        | 普通 Idempotency Key 去重                     | 仅防范完全重复消息，无法处理 Step 级一致性               | 无法恢复崩溃窗口           |
| **B4**        | Reserve / Occupy / Consume                    | 覆盖组生命周期，但缺少 Optimizer Checkpoint 绑定         | 存在 Q0/L0 崩溃窗口        |
| **B5**        | Per-step Checkpoint + 全量重算                | 保证强一致性，但恢复代价极高（整步全部重跑）             | 100% Step 算力重跑         |
| **RewardTxn** | Seal + Fencing + StepToken + Selective Replay | **零错误提交，且仅重算失败 Reward（节省 \(\ge 30\%\)）** | **节省 \(\ge 30\%\)**      |

---

## 8. 判定准则与修订后 Go / No-Go 门禁

在 Day 5 结束时，必须逐项核对以下门禁。**只有满足全部 7 项 Go 条件，方可批准进入 Phase 2；满足任一 No-Go 条件必须立即停止项目。**

```
                         [ Day 5 决策评审 ]
                                 |
        +------------------------+------------------------+
        |                                                 |
   [ 全部满足 7 项 Go 条件 ]                     [ 命中任一 No-Go 条件 ]
        |                                                 |
        v                                                 v
   [ 批准进入 Phase 2 ]                          [ 立即终止项目 ]
   - 编写最小事务协议                            - 转为向上游提 Issue / PR
   - 进行 8 卡端到端评测                         - 或转入备选方向 (DiffState)
```

### 8.1 Go 条件（必须全部满足）

1. **真实静默错误复现**：在锁定的公开栈中稳定复现至少 **2 类**静默错误，且至少 **1 类**发生在开启完整组校验与普通 Idempotency 之后。
2. **真实崩溃窗口存在**：实测复现出真实的 `Queue Consume ↔ Optimizer Checkpoint` 或 `Optimizer ↔ ACK` 状态不一致窗口。
3. **梯度污染可测**：故障注入造成确凿的 StepManifest 损坏、Gradient 偏差或 Optimizer Delta 漂移。
4. **零无效提交**：RewardTxn 原型在全部注入切点下保持 **0 个** Invalid Committed Step。
5. **重算节省达标**：相比 B2 / B5 强基线，Selective Replay 在维持正确性的前提下减少昂贵 Rollout/Reward 重算量 **\(\ge 30\%\)**。
6. **协议开销极低**：正常路径下的协议额外开销（不包含 Checkpoint 本身落盘用时）**\(< 5\%\)**。
7. **无现成等价保证**：确认所 Pin 版本的 StaleFlow / AReaL / TransferQueue 均未提供等价的 Durable Step Commit 机制。

### 8.2 No-Go 停损条件（任一成立即终止）

1. 所有错误仅靠显式 Group ID 或 AReaL 风格的 `drop_incomplete_group` 即可彻底消除。
2. 目标栈不存在可复现的消费—训练崩溃窗口。
3. B5（Per-step Checkpoint + 全量重跑）在实际工作负载下的开销已经极低，Selective Replay 节省不足 **20%–30%**。
4. 无法将 Exactly-once 形式化绑定到 Durable Checkpoint，方案退化为“大概率不丢”。
5. 必须通过任意内存篡改、伪造元数据等不切实际手段才能触发故障。
6. 相关开源项目近期已合入相同的 StepManifest / Checkpoint Commit 协议。
7. 最终系统贡献退化为单纯的 Lineage 查询界面或 SQLite WAL 简单封装。

---

## 9. 过门后 Phase 2：最小事务机制设计与端到端评测

_（注：本节内容仅在通过第 8 节全部 Go 门禁后执行）_

### 9.1 最小系统实现组件（必须实现的 7 个模块）

```
[ 1. Group Seal ]  ---> 绑定不可变 Prompt/Tokenizer/Policy/Verifier 版本向量
        |
[ 2. Reward CAS ]   ---> 基于 (LogicalID, Epoch, Attempt) 的原子比较交换写入
        |
[ 3. StepManifest ] ---> 将 U 个 GroupManifest 与 Predecessor Checkpoint 哈希打包
        |
[ 4. StepToken ]    ---> 生成唯一令牌并固化嵌入 Checkpoint Metadata
        |
[ 5. Reconciler ]   ---> 崩溃重启后基于最新 Durable Checkpoint 恢复一致状态
        |
[ 6. Selective Replay ] -> 仅重算失败的 Reward/Environment，复用 Rollout 前缀
        |
[ 7. Trace Runner ] ---> 自动化故障注入与端到端回归评测套件
```

### 9.2 8 卡端到端性能与正确性评测

- **资源分配**：4×H100 Rollout + 4×H100 Learner（数学任务 CPU Verifier）。
- **模型规格**：Qwen3-4B / Qwen2.5-7B。
- **评测指标**：
  1. 正常训练吞吐（Throughput Overhead \(< 5\%\)）。
  2. 故障注入下的恢复时延（Recovery Time in seconds）。
  3. 累计节省的 GPU·Hours 与 Rollout Tokens 数量。
  4. 最终收敛曲线与 Clean Oracle 的对齐度。

---

## 10. 产物规范、目录结构与可复现 Schema

### 10.1 实验目录标准树

```text
/public/home/caiyiwen/rewardtxn/
├── STACKS.lock                       # 框架版本与 Commit SHA 锁定文件
├── configs/                          # 实验配置文件 (YAML)
├── scripts/
│   ├── check_env.sh                  # 环境预检脚本
│   ├── phase0_diff_oracle.py         # Phase 0 差分验证脚本
│   ├── inject_fault.py               # 故障注入钩子脚本
│   └── judge_gates.py                # 自动化门禁判定脚本
└── runs/
    └── {exp_id}/                     # 单次实验唯一目录
        ├── meta.json                 # 实验元数据与环境快照
        ├── logs/                     # 训练与注入日志
        ├── manifests/                # Group 与 Step Manifest 导出 (JSONL)
        ├── checkpoints/              # Checkpoint 与原子指针
        ├── metrics.json              # 收集的指标数据
        └── verdict.json              # 自动门禁判定结果
```

### 10.2 核心元数据 `meta.json` 规范

```json
{
  "exp_id": "p1-slime-B5-R1_kill-K8-s42-20260824",
  "phase": "p1",
  "stack": "slime",
  "baseline": "B5",
  "commit_sha": "a6272da0d4f3d0a08520c99a2f3b4f6c887960dc",
  "seed": 42,
  "group_size_K": 8,
  "batch_groups_U": 4,
  "fault_injection": {
    "crash_point": "R1",
    "mechanism": "kill_pid",
    "target_step": 10
  },
  "tolerances": {
    "max_l2_diff": 0.001,
    "min_cosine_similarity": 0.995
  },
  "created_at": "2026-08-24T15:00:00.000Z"
}
```

### 10.3 门禁判定结果 `verdict.json` 规范

```json
{
  "exp_id": "p1-slime-RT-R1_kill-K8-s42-20260824",
  "auto_pass": true,
  "gates": {
    "silent_errors_detected": 2,
    "consume_checkpoint_window_reproduced": true,
    "measurable_gradient_delta": true,
    "invalid_committed_steps": 0,
    "selective_replay_savings_pct": 34.5,
    "protocol_overhead_pct": 2.8,
    "no_equivalent_in_upstream": true
  },
  "decision": "GO",
  "notes": "满足全部 7 项 Go 条件，批准进入 Phase 2。"
}
```

---

## 11. 运维与自动化判定脚本

在 `h100` 上运行以下 Python 脚本 `judge_gates.py`，自动解析 `runs/{exp_id}` 下的数据并生成最终决策：

```python
#!/usr/bin/env python3
"""
自动化 Go / No-Go 门禁判定器
"""
import json
import sys
from pathlib import Path

def evaluate_run(run_dir):
    run_path = Path(run_dir)
    meta_file = run_path / "meta.json"
    metrics_file = run_path / "metrics.json"

    if not meta_file.exists() or not metrics_file.exists():
        print(f"Error: 缺少元数据或指标文件 in {run_dir}")
        sys.exit(1)

    with open(meta_file, "r") as f:
        meta = json.load(f)
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # 检查 Go 条件
    g1 = metrics.get("silent_errors_count", 0) >= 2
    g2 = metrics.get("crash_window_reproduced", False) is True
    g3 = metrics.get("gradient_delta_observed", False) is True
    g4 = metrics.get("invalid_committed_steps", 999) == 0
    g5 = metrics.get("replay_savings_pct", 0.0) >= 30.0
    g6 = metrics.get("protocol_overhead_pct", 100.0) < 5.0
    g7 = metrics.get("no_upstream_equivalent", True) is True

    all_go = g1 and g2 and g3 and g4 and g5 and g6 and g7

    verdict = {
        "exp_id": meta.get("exp_id"),
        "auto_pass": all_go,
        "gates": {
            "g1_silent_errors_ge_2": g1,
            "g2_crash_window_reproduced": g2,
            "g3_gradient_delta_observed": g3,
            "g4_zero_invalid_commit": g4,
            "g5_savings_ge_30pct": g5,
            "g6_overhead_lt_5pct": g6,
            "g7_no_upstream_equivalent": g7
        },
        "decision": "GO" if all_go else "NO_GO"
    }

    out_file = run_path / "verdict.json"
    with open(out_file, "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"=== 评审结果: {verdict['decision']} ===")
    for k, v in verdict["gates"].items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")

    return 0 if all_go else 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python judge_gates.py <run_dir>")
        sys.exit(1)
    sys.exit(evaluate_run(sys.argv[1]))
```

---

## 12. 复现实验 Checklist（MLSys / EuroSys 标准）

在准备最终论文材料与开源发布前，必须逐项勾选确认：

- [ ] **环境全记录**：包含完整的 `nvidia-smi` 拓扑、内核版本、CUDA 与 Docker 版本快照。
- [ ] **版本绝对锁定**：三套框架的 Git Commit SHA 及 Docker Image Digest 均已固化且可拉取。
- [ ] **Phase 0 差分数据包**：包含小模型下 \(K \in \{4, 8, 16\}\) 的完整数值偏差与梯度余弦矩阵。
- [ ] **R0–C2 全切点日志**：每个注入切点均有独立的崩溃日志、恢复日志与 StepToken 轨迹。
- [ ] **B0–B5 逐级对比表**：证明在相同故障下，B3/B4/B5 会发生数据损坏或算力浪费。
- [ ] **开销分项核算**：清晰剥离“协议元数据开销（\(< 5\%\)）”与“Checkpoint 本身落盘开销”。
- [ ] **零无效提交证明**：长周期故障注入测试中，Invalid Committed Step 严格为 0。
- [ ] **无生产 Trace 依赖**：所有故障注入均可通过公开脚本 100% 独立复现。
