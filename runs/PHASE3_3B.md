# Phase 3B 阶段报告 — 可用性闭合（自动恢复链）

日期: 2026-08-28 | 门禁: **PASS (3/3)** | 回退点: `git checkout phase3a`（当前 `phase3b`）

## 前置探针：slime resume 可行性（风险前置，按 PHASE3_PLAN.md）

**机制调查**（slime/megatron 源码）:
- Day 4 L2 死锁根因: megatron `OptimizerParamScheduler._check_and_set` 在
  `use_checkpoint_opt_param_scheduler=False` 且 `override_opt_param_scheduler=False`
  时断言 `cls_value == sd_value`（恢复配置的 lr_decay_steps 必须等于 checkpoint 值
  = num_rollout×32）→ 恢复必须 num_rollout==已跑步数 → `range(start, num)` 为空 → 0 步
- **绕过点（原生参数）**: `--override-opt-param-scheduler`（megatron 0.16.0rc0
  training_config, slime 经 megatron parse_args 接受）→ 断言跳过, 用新配置;
  消耗量 `num_steps` 仍从 checkpoint 恢复 (load_state_dict: step(increment=num_steps))

**实弹探针** (0.5B, 4 卡):
- stage1: 8 步训练 (iter3/7 落盘) → SUCC
- stage2: `--load <ckpt> --override-opt-param-scheduler` + num_rollout=12
  → **从 step 8 续跑到 11** (start_rollout_id=7+1, range(8,12)), 无断言错无死锁
- **结论: 路线 A 可行** (纯参数, 零源码修改)

## 实现: 容器 entrypoint 自动恢复包装器

`scripts/phase3_auto_recover.sh`:
1. 训练 (day2_slime_train.sh) 失败检测 (if 分支, 退出码非 0)
2. 恢复计划: `latest_checkpointed_iteration.txt` 识别已提交步 + 写
   `rtx_recovery_history.jsonl` (attempt/committed_step/recovery_action)
3. 自动重启: `RTX_EXTRA_MODEL_ARGS="--load <ckpt> --override-opt-param-scheduler"`,
   num_rollout 保持目标值 → 续跑; 无 checkpoint 时冷启动 (协议记录重放)
4. 幂等: MAX_RETRIES=3 上限, 超限告警退出
5. 恢复后关闭故障注入 (RTX_FAULT/WINDOWS/KILL 清空) + ray/sglang 残留清理
6. 崩溃注入 (实验用, RTX_KILL_AFTER_ITER): checkpoint 落盘后 kill -9

**注入方式修正**: crm_crash 异常注入被 slime fully_async 容错捕获 (R1 语义 =
组静默丢弃, 训练不失败) → 自动恢复针对**真实进程崩溃** (kill -9, Day 4 L2 同款)。

## 端到端验证 (p3b-slime-kill-autorecover2)

| 事件 | 时间 |
|---|---|
| attempt 1 启动 | 15:11:38 |
| INJECT kill -9 (iter9 落盘后) | 15:20:23 |
| 恢复计划: committed_step=9, action=resume | 15:20:24 |
| attempt 2 重启 (--load+override) | 15:20:34 |
| 恢复后 step 10 运行 | 15:23:45 |
| 训练完成 (step 19) | ~15:27 |

## 门禁判定

| 门禁 | 结果 | 证据 |
|---|---|---|
| G3B1 无人干预自动恢复 | PASS | kill -9 → FAILED → 恢复计划 → 自动重启 → 续跑 10 步 → SUCC (2 attempts, 无人工) |
| G3B2 恢复正确性 | PASS | step 0..9 各仅出现 1 次 (已提交步不重训); 恢复段 loss vs 无崩溃基线平均差 0.0174 <0.05 |
| G3B3 恢复时延 <5min | PASS | 202s (崩溃 15:20:23 → step10 运行 15:23:45), 含 ray 清理+重启+模型加载 |

## 意义

**Day 4 L2 双重死锁的可用性侧闭合**: 崩溃后训练自动恢复续跑,
已提交步 (StepToken/checkpoint iteration) 不重训, 恢复时延 3.4min 内,
零源码修改 (megatron 原生参数 + 容器 entrypoint 包装)。

## 边界与记录

- 恢复路径依赖 checkpoint 存在 (save-interval 内崩溃 → 冷启动 + 协议记录重放,
  代价 = 已提交步之后的重训, 与 B5 基线相同但无需人工)
- exit_rc 字段记录在 kill 注入场景下不可靠 (kill -9 中断 bash 退出码捕获),
  恢复触发判定基于"训练未成功完成"语义 (if 分支), 功能正确
- 恢复后 LR 调度: override 用新配置的 lr_decay_steps, 消耗量从 checkpoint 恢复
  → LR 曲线按绝对进度继续 (语义正确)
- 3B 实验容器已清理; GPU 0-3 释放

## 产物
```
scripts/phase3_auto_recover.sh (自动恢复入口)   scripts/phase3_gate3b.py
runs/p3b-probe-{stage1,stage2}-K8-s42-20260828/ (探针: 8步+续跑4步)
runs/p3b-slime-kill-autorecover{,-2}-K8-s42-20260828/ (端到端: 注入+自动恢复)
runs/rtx_recovery_history.jsonl                  runs/PHASE3_GATE3B.json
runs/PHASE3_3B.md (本报告)
```
