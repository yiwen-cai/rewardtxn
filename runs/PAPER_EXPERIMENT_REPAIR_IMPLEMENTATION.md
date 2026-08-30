# Implementation Notes: RewardTxn 论文实验修复

## 执行合同
- 已批准计划：修复 P0 证据链与复现冻结，完成 pilot 后冻结 `paper-e0`，再运行正式 E2 30k 与真实进程证据。
- 成功标准：
  - E2 每次注入隔离，覆盖预注册调度维度，统一 oracle 能检出注册违规；
  - 门禁从原始产物重算并拒绝不自洽报告；冒烟与正式产物不互相覆盖；
  - 正式 run 满足事件、配置、元数据、verdict、hash manifest 契约并有自动测试；
  - 实际源码、镜像、模型、数据、seed、GPU 元数据可追溯；
  - prereg、统计脚本、数值阈值和预算在正式结果前冻结；
  - CPU 回归通过，pilot 完成，`paper-e0` 冻结；随后 E2 30k 和真实进程门禁通过。
- 明确不做：不重写 Phase 1–3 历史结论；不删除历史失败/冒烟产物；不推送远端；不做与论文证据链无关的重构。
- 当前状态：in_progress

## 任务与所有权
- [ ] T1 E2 runner/oracle/gate/产物契约修复 — agent: runner_impl — owns: `scripts/paper_trace_runner.py`, `scripts/trace_oracle.py`, `scripts/paper_gates.py`, `configs/paper_event.schema.json`, 相关新测试 — depends_on: 无
- [ ] T2 实验入口与复现冻结审计/实现 — agent: repro_impl — owns: `scripts/phase2_run.sh`, `scripts/day2_slime_train.sh`, `scripts/resource_gate.py`, `STACKS.lock`, 相关新脚本/测试 — depends_on: 无
- [x] T3 统计脚本与 prereg 收敛 — agent: stats_impl — owns: 新统计脚本/测试、`prereg/*.json`（不含 fault schedules） — depends_on: pilot 数据仅可保留待填字段
- [ ] T4 主代理整合、执行笔记、回归、pilot/P0/P1 执行与最终验收 — agent: root — owns: 本笔记、跨任务整合、runs 文档与最终产物 — depends_on: T1–T3
- [x] T4a fault schedule 与 GSM8K split 身份修复 — agent: root — owns: `scripts/fault_schedule_gen.py`, `scripts/gen_gsm8k_split.py`, `prereg/fault_schedules/`, `prereg/eval_splits/`, `tests/paper/test_schedule_and_split.py` — depends_on: 无
- [x] T4b 严格 run artifact 契约 — agent: root — owns: `scripts/validate_run_artifacts.py`, `tests/paper/test_artifact_contract.py` — depends_on: T1 输出接口约定
- [x] T4c 干净第三方源码准备 — agent: root — owns: `scripts/prepare_paper_source.py`, `tests/paper/test_prepare_source.py` — depends_on: 无

## 关键决策
### D1 正式 30k 延后到 P0 冻结之后
- 时间：2026-08-30
- 触发原因：HANDOFF 顺序与论文方案 P0/P1 冻结纪律冲突。
- 选择与理由：修复与 pilot 先行，预注册/统计/环境冻结为 `paper-e0` 后再产生正式 E2 结果。
- 被放弃的选择：立即重跑 30k 并在结果之后冻结。
- 影响范围：E2 结果时序、tag、prereg、交接文档。
- 是否偏离计划：否。

### D2 Phase 1–3 保留为历史/探索证据
- 时间：2026-08-30
- 触发原因：新发现集中在 P0 paper runner 与新证据契约，未证明历史真实进程结论错误。
- 选择与理由：不删除、不重写历史；正式论文核心 cell 从冻结版本复跑。
- 被放弃的选择：无差别废弃全部历史实验。
- 影响范围：历史 runs、正式 E1/E2 复跑范围。
- 是否偏离计划：否。

### D3 E8 使用真正的 Poisson 到达过程
- 时间：2026-08-30
- 触发原因：旧实现固定事件数后均匀采样，仅等价于给定 N 条件下的 Poisson 到达，不能称完整 Poisson 过程。
- 选择与理由：用确定性 schedule seed 驱动指数分布的事件间隔，并记录实际事件数与启动/收尾保护窗。
- 被放弃的选择：固定每个 8h run 恰好 16 个事件。
- 影响范围：3 份 E8 schedule 事件数改为 21/15/16；E8 仍保持均值每小时 2 次。
- 是否偏离计划：否。

### D4 正式实验挂载 pinned commit 的独立源码树
- 时间：2026-08-30
- 触发原因：实际 `third_party/slime` 含用户未提交配置修改，外层 tag 无法冻结该工作树。
- 选择与理由：从 pinned commit 准备独立 clean clone，记录全树内容哈希；保留 Git 元数据供正式入口校验，同时既不触碰用户修改，也不让 dirty 文件进入正式运行。
- 被放弃的选择：清理/提交用户的 nested-repo 修改；仅记录 dirty=true 后继续正式运行。
- 影响范围：正式 Docker bind mount、输入 manifest、复现说明。
- 是否偏离计划：否。

## 关键改动
- `runs/PAPER_EXPERIMENT_REPAIR_IMPLEMENTATION.md`：建立持续执行与交接记录；验证：最终逐文件审计。
- `scripts/fault_schedule_gen.py`：E7 seed 生成不再受批次顺序影响；E8 改为指数间隔 Poisson，增强 schema 自检。
- `prereg/fault_schedules/e7-s*.json`、`e8-soak-*.json`：按修复后生成器重建；验证：8/8 validate PASS。
- `scripts/gen_gsm8k_split.py`、`prereg/eval_splits/gsm8k_eval500_seed42.json`：split hash 绑定源 JSONL SHA-256，新增 verify。
- `tests/paper/test_schedule_and_split.py`：覆盖 seed 批次独立性、Poisson 确定性和源数据哈希绑定。
- `scripts/validate_run_artifacts.py`：严格验证 JSON/JSONL、事件 schema、config/schedule hash、退出状态和 artifact SHA-256，不把文件存在等同于契约通过。
- `tests/paper/test_artifact_contract.py`：覆盖完整 run、篡改 hash 和非法事件。
- `scripts/prepare_paper_source.py`：从 pinned Git commit 生成可验证的只读实验源码树，不含当前 checkout 的用户改动。
- `tests/paper/test_prepare_source.py`：证明 dirty checkout 不会进入归档树，且篡改会被检出。
- `scripts/paper_statistics.py`：实现配对/block bootstrap、TOST、非劣、Cliff's delta、Wilcoxon、Holm 与 pilot 精度规划。
- `scripts/validate_prereg.py`、`prereg/*.json`：正式启动 fail-closed；固定分析单位和结果无关规则，保留真实 pilot 待填项。
- `tests/paper/test_statistics_*.py`、`test_prereg_*.py`：统计数学与冻结时序测试。

## 验证记录
- 2026-08-30：修复前基线 `master@3262513`，主工作区 clean；当前 paper gate 因 360 次样本量不足而 FAIL。
- 2026-08-30：已并行派发 T1 runner/oracle/gate、T2 复现入口/冻结、T3 统计/prereg；文件所有权互不重叠。
- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.paper.test_schedule_and_split` → 3/3 PASS。
- `python3 scripts/fault_schedule_gen.py validate` → 8/8 schedule PASS。
- `python3 scripts/gen_gsm8k_split.py --verify` → PASS，source SHA-256 `39518db04de7132a…`。
- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.paper.test_artifact_contract` → 3/3 PASS。
- 宿主 `nvidia-smi`（提权只读）→ GPU 0–7 均 4 MiB/81559 MiB、util 0%；NUMA1 四卡组 3/4/6/7 可用。
- 宿主 Docker 只读核实：slime RepoDigest `sha256:2feaad36…`、image ID `sha256:420e8972…`；AReaL 本地 tag 为 `areal-project/areal-runtime:v2.0.0-sglang`，仅有 image ID `sha256:c0573bb8…`、无 RepoDigest。
- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest tests.paper.test_prepare_source` → 1/1 PASS。
- 实际 pinned Slime clean clone `/tmp/rewardtxn-paper-slime-a6272da0-v3` → detached `a6272da0`、工作树 clean、563 entries（含 1 symlink）、tree SHA-256 `e9063132c5cbf765…`，verify PASS。
- 统计专项 10/10、prereg 专项 7/7 PASS；`validate_prereg.py --allow-pending` PASS 并如实列出 23 项，正式模式预期 FAIL。

## 阻塞与后续
- GPU 与 Docker daemon 需使用已批准的只读/执行权限访问；当前已确认 GPU 空闲及镜像身份，代码修复完成后可进入 pilot 准备。
- `.codex/notes/` 为只读挂载，执行笔记改存 `runs/`。
