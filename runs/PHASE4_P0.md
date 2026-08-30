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

## 2. E2 trace 执行情况（2026-08-30，状态如实标注）

**第一轮（旧 fixture，已废弃）**：12 切点 × 2,500 = 30,000 注入 0 失败，聚合上界
1.0×10⁻⁴（耗时 54 分钟）。该轮 Q0/Q1/L0/L2/L3 为恒真 fixture（见 §6 自审），
结果已被新 fixture 取代；**30k 结果未入库**（提交时报告文件被最后一次确定性
测试覆盖为 360 注入版本），不作为正式证据。

**第二轮（新 fixture，当前代码）**：自审修复后仅完成中间验证——
300/切点 = 3,600 注入 0 失败（86s，确定性可复现）；**全量 30,000 注入被中断**
（用户打断），待重跑。中间报告 runs/TRACE_REPORT_PAPER.json 在 paper_gates 下
**FAIL（G2/G3 n 不足）属预期**，不会被误认为正式证据。

**耗时参考（计入预算）**：300/切点 86s → 线性外推全量约 12 分钟；
第一轮实测 55 分钟（含旧 fixture 的 R5 多进程开销），正式重跑以实测为准。
附录 D 风险 4 缓解成立：pilot 校准 + 分层并行可压到 <30 分钟。

## 3. 使用方法

```bash
# 重新生成 fault schedules（确定性，重复执行产物不变）
python3 scripts/fault_schedule_gen.py e7 && python3 scripts/fault_schedule_gen.py e8
python3 scripts/fault_schedule_gen.py validate

# 重新生成评测集划分
python3 scripts/gen_gsm8k_split.py

# E2 trace（正式 30,000 / 冒烟 --per-cut 200）
python3 scripts/paper_trace_runner.py
python3 scripts/paper_trace_runner.py --per-cut 200

# 门禁判定
python3 scripts/paper_gates.py --trace-report runs/TRACE_REPORT_PAPER.json

# run 契约校验（新实验每批必跑）
python3 scripts/validate_run_artifacts.py runs/<exp_id> --paper --json
```

## 4. P0 剩余事项（待代码审查与 GPU 槽位）

1. **重跑全量 E2 trace 并入库**：`python3 scripts/paper_trace_runner.py`（新 fixture，
   预计 12–55 分钟）→ `paper_gates.py` PASS 后提交 runs/TRACE_REPORT_PAPER.json；
2. 创建 paper-e0 tag（从 phase3-final），扩展 STACKS.lock 记录镜像 digest
   （slime v0.3.1 = 420e89724ba1、areal-runtime v2.0.0-sglang = c0573bb8412a）；
3. 冻结 prereg：tost_margin 单一值、prereg_index frozen_at/freeze_commit；
4. GPU pilot：E3/E6/E7 核心 cell 短训 2–3 次（当前可用槽位 NUMA1 = GPU 3,4,6,7，
   GPU0 被外部任务占用），校准 §15.6 预算表；
5. 新脚本（附录 A 标"新增"）：areal_repro 入口、schedule_player.py、
   replay_boundary.py、soak_runner.sh（骨架可后续补齐）；
6. phase2_trace_runner.py 保持不动（既有 TRACE_REPORT.json 流水线），
   E2 12 切点由 paper_trace_runner.py 承载。

## 5. 自审发现与修复（2026-08-30，交接前完成）

| # | 问题 | 修复 | 验证 |
|---|---|---|---|
| 1 | Q0/Q1/L0/L2/L3 为恒真 fixture（ok 写死或构造即真，50 次随机化判定不变） | 改为"状态机产生事件日志 + 日志解释器判定"，oracle 从日志反推状态 | 空转检测：各切点 50 次随机化唯一结果 ≥3；破坏性测试（删恢复/错决策/假提交）全部检出 |
| 2 | Q0 语义错误：未区分"已消费已提交（不重投）"与"已消费未提交（必须重投）" | 引入 committed 集合，恢复只重投未提交部分 | lost/over/dup 三向校验 |
| 3 | R1–R3 未随机化 K（方案要求 group 大小维度） | k ∈ {4,8,16} 按迭代随机 | 各 K 下 seal/autofix 断言通过 |
| 4 | e8 schedule 缺 steps/k/u 字段（违反附录 B.2 schema） | 补字段，validate 全绿 | 生成器 validate 8/8 OK |
| 5 | C1/C2 的权威映射从日志自身构造（循环自洽，c 类违规测不到） | fixture 返回 (事件日志, 外部权威映射)，oracle 独立判定 | 篡改 token/hash 被检出 |
| 6 | R2/R3 的 v1/v2 值相同（GOOD 样本上 v1=v2=1），AUTO_FIX 值修正未真正考验 | 引入发散样本 \boxed{42.0}（v1=1, v2=0），后半个组用发散样本 | autofix_values 逐位置对拍通过 |
| 7 | 已提交的 TRACE_REPORT_PAPER.json 实为 360 注入版本（提交时被确定性测试覆盖），30k 结果未入库 | 本阶段文档如实标注；待重跑全量后提交 | 中间报告在 paper_gates 下 FAIL（n 不足）属预期 |

## 6. 与方案的一致性

- 附录 B.2 schema：生成器输出字段与文档示例一致（schedule_id/seed/stack/model/
  steps/events[]，step→group 换算 = step×U..step×U+U-1）；
- 附录 C.2 oracle：4 条 invalid-commit 定义全部实现于 trace_oracle.py；
- 附录 C.3 评测集：GSM8K 7,473 → 6,973 + 500（seed 42，split_hash 固定）；
- §10 数据契约：validate_run_artifacts.py 实现 9 类产物与 events 行校验；
- §15.1 P0 门禁：CPU regression 未重跑（本阶段不涉及 third_party 改动），
  新工具链全部自测通过。
