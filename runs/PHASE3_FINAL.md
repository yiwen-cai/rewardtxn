# Phase 3 最终结项归档

日期: 2026-08-29

最终裁决: **PASS — Phase 3 已完成，无待完成的 Phase 3 必做实验**

阶段版本: `phase3a` → `phase3b` → `phase3c`

最终归档版本: `phase3-final`

## 1. 结项目标与最终状态

Phase 3 在 Phase 2 最小事务协议之上闭合三个层次：

1. **3A 消费侧正确性**：ABORTED 混版本组在训练消费前统一为权威 v1 reward；
2. **3B 可用性**：训练进程死亡后自动生成恢复计划、自动重启并从 checkpoint 接续；
3. **3C 规模化与回归**：全协议回归、长程收敛、多进程 CAS、模型升级和资源保护。

最终阶段门禁为 **9/9 PASS**，一键协议回归为 **9/9 PASS**。Phase 3 的完成
定义全部满足；8 卡 4+4 与更大模型测试始终是门禁外增强项，不属于未完成的 Phase 3
工作。

## 2. 阶段门禁总表

| 阶段 | 门禁 | 结果 | 关键证据 |
|---|---:|---|---|
| 3A | G3A1–G3A3 | **3/3 PASS** | 注入组 20/20 ABORTED+autofix；训练消费 reward 对 v1 权威值 0 不一致；干净组 AUTO_FIX on/off 配对 8/8 一致；同模式吞吐下降 2.78% |
| 3B | G3B1–G3B3 | **3/3 PASS** | kill -9 后无人干预恢复；已提交 step 0–9 各仅一次；恢复 202s（优化配置复验 148s） |
| 3C | G3C1–G3C3 | **3/3 PASS** | 回归 9/9；100 步逐步 loss MAE 0.006466<0.05；4 进程 CAS 恰好一次 |

权威门禁文件：

- `runs/PHASE3_GATE3A.json`
- `runs/PHASE3_GATE3B.json`
- `runs/PHASE3_GATE3C.json`
- `runs/PHASE3_REGRESSION.json`

## 3. 完成定义逐项核对

| # | Phase 3 完成定义 | 归档证据 | 状态 |
|---|---|---|---|
| 1 | 3A/3B/3C 全部门禁 PASS | 三个 GATE JSON，均 3/3 | ✅ |
| 2 | 无人干预端到端故障恢复 | `p3b-slime-kill-autorecover2-*`；kill→plan→restart→step10–19 | ✅ |
| 3 | 真实训练消费侧 0 混算 | 3A 注入运行；20/20 组修正，样本级 0 mismatch | ✅ |
| 4 | R1/R2/R3/Q0/L0/L2 全切点回归一条命令可复现 | `bash scripts/phase3_regress.sh`，含显式六切点矩阵，9/9 PASS | ✅ |
| 5 | 阶段文档、门禁与 tag 完整 | `PHASE3_PLAN.md`、3A/B/C 报告、3 个 tag | ✅ |
| 6 | 最终归档可验证 | 本报告、archive manifest、convergence evidence、`phase3-final` | ✅ |

## 4. 关键结果

### 4.1 正确性

- 1.5B skew 窗口组：**20/20 ABORTED + autofix**；
- 1.5B 归档 reward：4,296 条，`verifier=v1` 之外记录为 **0**；
- 0.5B 100 步运行：2,751 组全部 SEALED，10,368 条 reward 全部为 v1；
- 干净组 AUTO_FIX on/off 对同一组 8/8 返回值一致；
- R1/R2/R3/Q0/L0/L2、Replay 正确性、CAS 单/多进程幂等均进入 9/9 回归。

### 4.2 可用性

- 实弹路径：iter9 落盘后 kill -9 → 自动识别 committed=9 →
  `--load + --override-opt-param-scheduler` 自动重启 → 完成 step10–19；
- 已提交 step 0–9 不重训；
- 首次门禁恢复时延 202s，内存/IO 优化配置复验为 148s，均小于 5min；
- 自动恢复最多重试 3 次，避免无限恢复循环。

### 4.3 长程、规模与资源

- 新配置 0.5B Seal 100 步 vs 历史同 Seal 配置 100 步：逐步 loss
  平均绝对差 **0.006466 < 0.05**；
- 新运行 100 步 loss 均值 0.006333、整体 population std 0.005636，
  后 50 步均值 0.003331；
- 1.5B 20 步复验证明 Seal/CAS/AUTO_FIX/checkpoint 语义在模型升级后保持一致；
- CAS 4 进程并发同组只保留一份权威记录；
- Ray object store 16GiB、SGLang concurrency 64、SQLite CAS、checkpoint 滚动保留、
  启动资源门禁和训练期间健康采样均已在线验证。

逐步 loss 序列、统计量及源日志哈希归档在
`runs/PHASE3_CONVERGENCE_EVIDENCE.json`。

## 5. 归档结构与保留策略

### 5.1 Portable evidence（进入 Git）

以下证据必须能在 checkout 后独立审阅或运行：

- 计划、3A/B/C 报告、本报告；
- Gate JSON、回归 JSON、收敛序列与恢复历史；
- 协议、恢复、六切点回归、资源门禁和归档脚本及其 CPU 依赖锁定；
- Qwen2.5-1.5B `rotary-base` 单行兼容 patch；
- canonical runs 的 meta/rewards/seals/manifests 与训练日志；
- 恢复审计报告。

每个 portable 文件的字节数和 SHA-256 位于
`runs/PHASE3_ARCHIVE_MANIFEST.json`。其中 `archive_base_commit` 明确表示生成归档前的
源基线提交；最终归档提交由 `phase3-final` tag 绑定，避免 manifest 自指哈希循环。

### 5.2 Local large artifacts（不进入 Git）

checkpoint 权重和模型资产体积约数十 GiB，继续由 `.gitignore` 排除。manifest 对每个
canonical run 记录 checkpoint：

- 是否存在；
- iteration 目录；
- latest iteration；
- 文件数、总字节数；
- 基于“相对路径+文件大小”的 inventory SHA-256。

该摘要用于本机归档盘点，不冒充 checkpoint 内容哈希。Phase 3 的 CPU 回归已改为使用
归档 StepToken 构造轻量 fixture，**不依赖被 Git 排除的大 checkpoint**。

## 6. 未完成/替代运行的处置

以下运行未形成有效门禁证据，保留为本地诊断数据并由 `.gitignore` 明确排除：

| 运行 | 有效步数 | 处置原因 |
|---|---:|---|
| `smoke-...232347` | — | 冒烟配置修复前失败，已由 `...232940` 替代 |
| `p3c-slime-long-clean-*` | 7 | 训练失败 |
| `p3c-slime-long-clean2-*` | 9 | 训练中断 |
| `p3c-slime-long-clean-dm-*` | 70 | 训练中断 |
| `p3c-slime-long-seal-dm-*` | 10 | CUDA invalid argument，已由完整 100 步 Seal 替代 |

它们不参与任何 PASS 门禁统计。

## 7. 口径说明与已知边界

1. **G3C2 长程证据**是 0.5B 新配置 Seal 100 步与历史同 Seal 配置 100 步的对齐；
   1.5B 是独立的 20 步协议升级复验。归档不将后者表述为 100 步长程实验。
2. 0.5B 新长程运行的 `meta.json` 中 `params.rm=custom_rm_path` 是当时 metadata
   默认值遗留；同运行的 `seals.jsonl`、`rewards.jsonl` 和启动日志证明 Seal 链路实际
   启用。运行后入口已在提交 `454a8ea` 修复为从 `RTX_CUSTOM_RM` 派生。
3. 8 卡 4+4、Qwen3-4B/7B 和多 seed 配对长程属于论文/规模增强项，未纳入 Phase 3
   强制门禁。
4. RewardTxn 协议和训练循环未修改 third_party 源码；Qwen2.5-1.5B 模型配置存在一处
   `rotary-base 10000→1000000` 兼容修正，已独立归档为
   `patches/slime-qwen2.5-1.5b-rotary-base.patch`。
5. 单样本 RM 路径无法撤回已返回 reward；消费侧正确性依赖 slime 原生 `--group-rm`，
   已在 3A 真实训练路径验证。
6. checkpoint 前崩溃走冷启动和协议重放；本轮实弹恢复门禁聚焦存在 iter9 checkpoint
   的 resume 路径。

以上边界均已显式归档，不改变 Phase 3 完成裁决。

## 8. 验证与恢复命令

```bash
# GPU 无关的协议回归依赖（已有环境可跳过安装）
python3 -m pip install -r requirements-phase3-archive.txt
bash scripts/phase3_regress.sh

# 重建归档索引与校验本地源证据
python3 scripts/phase3_archive.py write
python3 scripts/phase3_archive.py verify --require-local-sources

# 最终 tag 建立后校验版本与干净工作区
python3 scripts/phase3_archive.py verify --require-final-tag --require-clean

# 恢复结项版本
git checkout phase3-final
```

## 9. 后续交接

**Phase 3 无剩余必做实验。** 后续若继续，应单独建立 Phase 4/论文增强计划，不应再以
“补 Phase 3”为名扩张已完成的验收范围。候选方向仅包括：8 卡大模型复测、多 seed
长程配对、低噪声吞吐基准、框架上游化与生产监控。
