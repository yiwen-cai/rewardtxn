# HANDOFF — RewardTxn 论文实验 P0 交接文档

> 交接时间：2026-08-30（第二轮会话结束）
> 交接人：caiyiwen（PI 会话）
> 接收人：下一会话 / 协作者
> 关联文档：`RewardTxn 论文级补充实验方案.md`（§15 + 附录 A–E）、`runs/PHASE4_P0.md`、
> `REWARDTXN_EXPERIMENT_DELIVERY.md`（Phase 1–3 交付）、`RewardTxn H100 自包含实验指导.md`

---

## 1. 一句话现状

论文实验方案已补全为自包含分阶段方案（§15 + 附录 A–E），P0 工具链骨架**已完成并提交**
（`ed3d655`），自审发现的 6 个 fixture/oracle 缺陷**已修复并验证**（本次交接提交）；
**唯一关键遗留：E2 全量 30,000 次 trace 需用新 fixture 重跑并入库**（中间报告当前在
paper_gates 下 FAIL 属预期）。

## 2. Git 状态

```
f7c423a docs: 补充实验方案补全为自包含分阶段方案（§15 + 附录 A–E）
ed3d655 phase4-p0: 论文实验骨架与预注册（schema/prereg/生成器/oracle/trace）
<本次交接提交> 自审修复 + 交接文档（见 git log 最新）
```

**重要事实（易踩坑）**：`ed3d655` 里入库的 `runs/TRACE_REPORT_PAPER.json` 实际是
**360 注入（per-cut 30）的确定性测试产物**，不是 30k 结果——提交时报告文件被最后一次
确定性测试覆盖，30k 全量结果从未入库。该报告在 `paper_gates.py` 下 FAIL（G2/G3 n 不足），
不会被误认为正式证据。**不要把它当证据引用，也不要删掉**——重跑全量后覆盖提交即可。

## 3. 已完成并验证

### 3.1 P0 工具链（ed3d655，自审后修订）

| 产物 | 路径 | 状态 |
|---|---|---|
| events.jsonl 行 schema | configs/paper_event.schema.json | 完成 |
| 预注册模板 ×9 | prereg/ | 完成（pending_pilot） |
| fault schedules（E7×5 seed + E8×3 soak） | prereg/fault_schedules/ | 完成，确定性，validate 8/8 |
| GSM8K 评测集划分 | prereg/eval_splits/gsm8k_eval500_seed42.json | 完成（7,473=6,973+500） |
| trace oracle（附录 C.2） | scripts/trace_oracle.py | 完成，单测通过 |
| run 契约校验器（§10） | scripts/validate_run_artifacts.py | 完成（paper/legacy 双模式） |
| E2 12 切点 trace runner | scripts/paper_trace_runner.py | **修复后待全量重跑** |
| E2 门禁判定 | scripts/paper_gates.py | 完成 |

### 3.2 自审发现并修复的 6 个问题（重要，勿回退）

1. **恒真 fixture**：Q0/Q1/L0/L2/L3 原本 `ok=True` 写死或构造即真（50 次随机化判定不变）。
   已改为"状态机产生事件日志 + 日志解释器（VERIFIERS）从日志反推状态判定"。
   验证：空转检测（各切点唯一结果 ≥3）+ 7 项破坏性测试（删恢复/错决策/假提交/重复
   apply/token 篡改/hash 篡改/双重恢复计划）全部检出。
2. **Q0 语义**：补上"已消费已提交（不重投）vs 已消费未提交（必须重投）"的区分。
3. **R1–R3 随机化 K**：k ∈ {4,8,16} 按迭代随机（方案要求的 group 大小维度）。
4. **C1/C2 循环自洽**：权威映射不再从日志构造，fixture 返回 (事件日志, 外部权威映射)，
   oracle 独立判定，(b)/(c) 类违规真实可测。
5. **R2/R3 值级发散**：引入 `\boxed{42.0}` 样本（v1=1, v2=0），AUTO_FIX 的值修正被真实考验。
6. **e8 schedule schema**：补 steps/k/u 字段。

### 3.3 自审中发现但保留的已知边界

- Q0/Q1/L0/L2/L3/C1/C2 是**协议模型 fixture**（非真实进程），与真实进程实验分表报告——
  方案 §1 原则如此设计，不是缺陷；
- 模型 fixture 判定"恒过"是正确语义（协议模型实现正确时本就不该失败），其价值在
  覆盖参数空间 + 锁定不变量；统计强度来自 R1–R5 真实代码路径 + 后续真实进程实验。

## 4. 交接时的工作区状态

- 未提交：`scripts/paper_trace_runner.py`（修复）、`scripts/fault_schedule_gen.py`（e8 字段）、
  `prereg/fault_schedules/e8-soak-*.json`（重生成）、`runs/TRACE_REPORT_PAPER.json` 与
  `runs/TRACE_PAPER_EVENTS_SAMPLE.jsonl`（300/切点中间结果）、`runs/PHASE4_P0.md`（如实修订）、
  **本文档**——本次交接提交一并入库；
- 中断的全量 30k 运行残留 tmp 目录 `/tmp/rtx-paper-trace-*`，可清理；
- 全量 30k 重跑后**必须**用 `paper_gates.py` 确认 PASS 再提交报告。

## 5. 下一步（按优先级）

1. **重跑全量 E2 trace**（第一件事，纯 CPU）：
   ```bash
   python3 scripts/paper_trace_runner.py          # 12×2,500=30,000，预计 12–55 分钟
   python3 scripts/paper_gates.py --trace-report runs/TRACE_REPORT_PAPER.json   # 必须 PASS
   git add runs/TRACE_REPORT_PAPER.json runs/TRACE_PAPER_EVENTS_SAMPLE.jsonl && git commit
   ```
2. **paper-e0 tag 冻结**（需用户点头）：
   `git tag paper-e0`（从当前 HEAD）+ 扩展 `STACKS.lock` 记录镜像 digest：
   slime v0.3.1 = `420e89724ba1`、areal-runtime v2.0.0-sglang = `c0573bb8412a`
   （⚠️ 勿用 slime `latest` 旧镜像 `8b75cc58197e`）。
3. **冻结 prereg**：`prereg/tost_margin.json` 填单一 margin、`prereg_index.json` 填
   frozen_at/freeze_commit（建议等 GPU pilot 出方差后再定 margin 值）。
4. **GPU pilot**（E3/E6/E7 核心 cell 短训 2–3 次，校准 §15.6 预算表）：
   当前 GPU0 被外部任务占用（util 100%），可用完整 4 卡槽 = **NUMA1：GPU 3,4,6,7**；
   1.5B 必须 `RTX_EXTRA_MODEL_ARGS="--rotary-base 1000000"`。
5. **新脚本骨架**（附录 A 标"新增"，不阻塞主线）：`areal_repro` 入口、
   `schedule_player.py`、`replay_boundary.py`、`soak_runner.sh`。

## 6. 环境快照（2026-08-30 实测）

| 项 | 值 |
|---|---|
| 系统 Python | 3.8.10（新脚本已加 `from __future__ import annotations`，新代码勿用运行时求值的 `list[...]` 注解） |
| /public 可用 | 988G（86% 已用，高于 200G 门禁） |
| GPU | GPU0 被外部任务占用（727MiB/100%），GPU1–7 空闲 |
| 镜像 | slime v0.3.1（420e89724ba1）、areal-runtime v2.0.0-sglang（c0573bb8412a）均在 |
| 数据集 | models/datasets/gsm8k/dapo-gsm8k-train.jsonl（7,473 行） |

## 7. 速查命令

```bash
# 全量 trace + 门禁（交接后第一件事）
python3 scripts/paper_trace_runner.py && python3 scripts/paper_gates.py --trace-report runs/TRACE_REPORT_PAPER.json

# 冒烟（<2 分钟）
python3 scripts/paper_trace_runner.py --per-cut 200

# 重新生成 fault schedules / 评测集划分（确定性，产物不变）
python3 scripts/fault_schedule_gen.py e7 && python3 scripts/fault_schedule_gen.py e8 && python3 scripts/fault_schedule_gen.py validate
python3 scripts/gen_gsm8k_split.py

# run 契约校验（每批新实验必跑）
python3 scripts/validate_run_artifacts.py runs/<exp_id> --paper --json
```

## 8. 需要接收人/用户决策的事项

1. 是否现在创建 paper-e0 tag + STACKS.lock 冻结（第 4 节第 2 项）；
2. TOST margin 冻结值（pilot 后定）；
3. 是否按 NUMA1（3,4,6,7）启动 GPU pilot；
4. 全量 30k 重跑结果是否需要当天入库（约 12–55 分钟，纯 CPU，不占 GPU）。
