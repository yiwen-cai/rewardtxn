# Phase 2A 阶段报告 — Group Seal + Reward CAS（数据面）

日期: 2026-08-27 | 门禁: **PASS (3/3)** | 回退点: `git checkout phase1-complete`（当前已提交 `phase2a`）

## 设计

在 slime `--custom-rm-path` 包装层实现（零 third_party 改动），`scripts/phase2_seal_rm.py`：
- **Group Seal 状态机**: 逐样本登记 (group_index → {verifiers, logical_ids, count})；
  组完成 (count==K=8) 时检查版本一致性 → 全一致 **SEALED** / 版本混合 **ABORTED**，
  审计流写 `seals.jsonl`（GroupManifest 雏形，2B/2C 直接输入）
- **Reward CAS**: rewards.jsonl 按 logical_id=(group_index,index,rollout_id) 幂等写入，
  重复提交 → 拒绝 + `cas_rejects.jsonl` 审计
- 复用 Day 2 注入框架（skew 注入使 Seal 检测可验证）

## 门禁验证

| 门禁 | 实验 | 结果 |
|---|---|---|
| **G2A1** 注入组 100% 检测 | skew+seal 20 步 (窗口 g20-39) | **20/20 ABORTED**（versions=[v1,v2], count=8, unique=8）；其余组 SEALED |
| **G2A2** 干净组零误报 | none+seal 20 步 | **1245/1245 SEALED, ABORTED=0** |
| **G2A3** 协议开销 <5% | 字节 + 端到端 | Seal 元数据 0.73%/步（模拟器一致口径, 373B vs 51200B）< 5%；20 步总时长 11.3min vs 无 seal 基线 14.5min（波动内无系统差异）；纯逻辑 21µs/样本 |
| CAS 单测 | 本地幂等验证 | 重复 logical_id 拒绝 + 审计；异 index/rollout 放行 |

## 关键发现

1. **检测率 100% + 零误报**：Seal 在无注入下完全透明，注入下全量检出——把 Day 2
   "完全静默"变成"显式可审计"
2. **组数 > 训练消费数**（564-1245 组 vs 训练 80 组）：slime 预取 buffer 使 rollout
   生成量远大于消费量——这放大了"ABORTED 组需要重放"的成本面，Selective Replay
   价值更高（2C 量化）
3. **设计边界（分阶段诚实记录）**：2A 完成检测/审计/CAS。逐样本 RM 调用下，
   ABORTED 组前几条 reward 已返回 slime 无法回改；"0 混算进入训练"的消费侧 gate
   需要 2B StepToken（绑定 checkpoint）+ 2C Reconciler/Selective Replay 完成

## 回退与版本
- `git tag phase2a`（本阶段全部代码/脚本/文档/门禁 JSON）
- 回退: `git checkout phase2a` / `git checkout phase1-complete`（阶段间独立）
- 失败演练: 本阶段两实验各遇一次训练侧 NaN（既有随机性问题, 与 Seal 无关, 重跑成功）

## 产物
```
scripts/phase2_seal_rm.py      # Seal+CAS 包装层 (RTX_SEAL=1)
scripts/phase2_run.sh          # 阶段运行入口 (meta 带窗口/phase=p2)
scripts/phase2_gate2a.py       # CAS 单测 + seal 分析 + gate2a.json
runs/p2a-slime-skew-seal-K8-s42-20260826/   # G2A1 实验 (20步 succeeded)
runs/p2a-slime-none-seal-K8-s42-20260826/   # G2A2/G2A3 实验 (20步 succeeded)
runs/PHASE2_GATE2A.json        # 门禁 JSON
```
