# Phase 4 P0：论文实验骨架与预注册（P0 阶段文档）

> 阶段状态：**P0 骨架完成**（2026-08-30）。本阶段产出方案 §15.1 所列的冻结前全部
> 工具链骨架：event schema、prereg 模板、fault schedule 生成器、评测集划分、
> trace oracle、run 契约校验器、E2 12 切点 trace runner、门禁判定器。
> 尚未执行：paper-e0 tag 冻结、真实 GPU pilot（等待代码审查与 GPU 槽位）。

## 1. 交付清单

| 产物 | 路径 | 验证 |
|---|---|---|
| events.jsonl 行 schema | configs/paper_event.schema.json | 手写校验器同源实现 |
| 预注册模板（9 文件） | prereg/*.json + README | JSON 可解析，字段齐全 |
| fault schedule 生成器 | scripts/fault_schedule_gen.py | e7 5 seeds + e8 3 soak 全部校验通过 |
| fault schedules | prereg/fault_schedules/（8 个 JSON） | 确定性（重复生成 hash 一致） |
| GSM8K 评测集划分 | prereg/eval_splits/gsm8k_eval500_seed42.json | 7,473 = 6,973 + 500，split_hash 固定 |
| trace oracle（附录 C.2） | scripts/trace_oracle.py | 5 类违规全检出；CP/Wilson/rule-of-three 数学自检 |
| run 产物契约校验器（§10） | scripts/validate_run_artifacts.py | legacy run 如实 FAIL；paper run PASS；CLI 可用 |
| E2 trace runner（12 切点） | scripts/paper_trace_runner.py | 30,000 注入 0 失败；确定性可复现 |
| E2 门禁判定 | scripts/paper_gates.py | G1–G4 全 PASS |

## 2. E2 trace 全量结果（本次执行，非正式结论）

- 12 切点 × 2,500 次 = **30,000 次确定性随机化注入，0 invalid commit**；
- 聚合 95% 失败率上界（rule of three）= **1.0×10⁻⁴**，与方案 RQ2 门禁一致；
- 每切点 Clopper–Pearson 上界 ≈1.20×10⁻³、Wilson ≈1.53×10⁻³（k=0, n=2,500）；
- R1–R5 驱动**真实** phase2_seal_rm 代码路径（SQLite CAS/Seal/AUTO_FIX/多进程），
  Q0/Q1/L0/L2/L3/C1/C2 为确定性协议模型 fixture（与真实进程实验分表报告）；
- 报告：runs/TRACE_REPORT_PAPER.json；事件样本：runs/TRACE_PAPER_EVENTS_SAMPLE.jsonl。

**耗时实测（计入预算）**：全量 30,000 次 ≈ **55 分钟**（单进程；R5 多进程 spawn 与
R1–R3 真实代码批量为主要开销）。附录 D 风险 4 缓解成立：pilot 校准 + 分层并行可将
正式重跑压到 <30 分钟。

## 3. 使用方法

```bash
# 重新生成 fault schedules（确定性，重复执行产物不变）
python3 scripts/fault_schedule_gen.py e7 && python3 scripts/fault_schedule_gen.py e8
python3 scripts/fault_schedule_gen.py validate

# 重新生成评测集划分
python3 scripts/gen_gsm8k_split.py

# E2 trace（正式 30,000 / 冒烟 200/切点）
python3 scripts/paper_trace_runner.py
python3 scripts/paper_trace_runner.py --per-cut 200

# 门禁判定
python3 scripts/paper_gates.py --trace-report runs/TRACE_REPORT_PAPER.json

# run 契约校验（新实验每批必跑）
python3 scripts/validate_run_artifacts.py runs/<exp_id> --paper --json
```

## 4. P0 剩余事项（待代码审查与 GPU 槽位）

1. 创建 paper-e0 tag（从 phase3-final），扩展 STACKS.lock 记录镜像 digest
   （slime v0.3.1 = 420e89724ba1、areal-runtime v2.0.0-sglang = c0573bb8412a）；
2. 冻结 prereg：tost_margin 单一值、prereg_index frozen_at/freeze_commit；
3. GPU pilot：E3/E6/E7 核心 cell 短训 2–3 次（当前可用槽位 NUMA1 = GPU 3,4,6,7，
   GPU0 被外部任务占用），校准 §15.6 预算表；
4. 新脚本（附录 A 标"新增"）：areal_repro 入口、schedule_player.py、
   replay_boundary.py、soak_runner.sh（骨架可后续补齐）；
5. phase2_trace_runner.py 保持不动（既有 TRACE_REPORT.json 流水线），
   E2 12 切点由 paper_trace_runner.py 承载。

## 5. 与方案的一致性

- 附录 B.2 schema：生成器输出字段与文档示例一致（schedule_id/seed/stack/model/
  steps/events[]，step→group 换算 = step×U..step×U+U-1）；
- 附录 C.2 oracle：4 条 invalid-commit 定义全部实现于 trace_oracle.py；
- 附录 C.3 评测集：GSM8K 7,473 → 6,973 + 500（seed 42，split_hash 固定）；
- §10 数据契约：validate_run_artifacts.py 实现 9 类产物与 events 行校验；
- §15.1 P0 门禁：CPU regression 未重跑（本阶段不涉及 third_party 改动），
  新工具链全部自测通过。
