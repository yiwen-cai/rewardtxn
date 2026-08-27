# Day 2 方案修复记录 (2026-08-25)

按修正方案完成 Day 2 基础设施搭建与验证。共修复 4 个问题。

## 修复 1: 注入点与架构不匹配 (文档 R1/R2 在 slime 中无对应物)

- **事实**: deepscaler RM 是同步函数, 在 RolloutManager actor 内联执行; 无独立 Reward Worker 进程; 无 at-least-once 重试窗口
- **方案**: 利用 slime 官方 `--custom-rm-path` 参数写注入模块 `scripts/day2_custom_rm.py`
  - 兼容单样本 (fully-async 路径) 与 batch 两种调用签名
  - R3_skew: 按 group_index 窗口 + 组内位置分流 v1/v2 (v2=strict 双 grader 判定变体)
  - R1_crm_crash: 窗口内抛 RuntimeError
  - R2_dup: 弱化版 (陈旧重试结果混入), 完整语义留 Day 3
- **验证**: 窗口内每组恰好前 4 条 v1 + 后 4 条 v2, 窗口外零注入

## 修复 2: 注入效果不可测 -> 逐样本 instrumentation

- **方案**: custom RM 内逐样本落盘 `rewards.jsonl` (group_index/index/rollout_id/reward/verifier/injected/fault)
- **附带收益**: 解决 Day 1 Lineage `provenance=derived` 问题, Day 2 起为真实逐样本数据

## 修复 3: reward 信号为 0 (三个叠加根因)

1. **GSM8K 数据缺 role 字段**: Qwen chat template 按 role 匹配, 无 role 消息被跳过 -> 模型只见 system 输出摆烂回复 "Hello!"
   - 修复: `scripts/prepare_gsm8k.py` 转换时加 `"role": "user"` (DAPO 原数据有 role, 之前转换遗漏)
2. **0.5B 模型能力不足**: DAPO 竞赛题答对率 0, 输出含乱码
   - 修复: 按文档 4.2 换 Qwen2.5-1.5B-Instruct (ModelScope 下载 + torch_dist 转换)
3. **slime deepscaler RM bug**: 原版要求 response 含 `</think>` 或 `###Response` 分隔符 (DeepSeek 风格), Qwen 输出无此标记 -> 恒 0
   - 修复: `_v1_reward` 兼容版 (无分隔符时直接对完整 response 提取答案)
   - 附带修复: slime `qwen2.5-1.5B.sh` 上游脚本 rotary-base=10000 错误 (实际 1000000), 转换与训练均用 `--rotary-base 1000000` 覆盖

## 修复 4: 运行环境问题

- `--ipc=host` 导致容器共享宿主 /dev/shm, 外部 ray 任务占满 tmpfs 后容器内 ray object store OOM -> 移除, 容器独立 64G shm
- custom_rm_path 需以 `模块.函数` 形式传参且 /workspace/scripts 加入 ray runtime PYTHONPATH
- GPU 0-3 被外部用户训练占用 -> 运行脚本支持 RTX_GPUS/RTX_NUM_GPUS 动态选择

## 验证结果 (2 卡 smoke, 1.5B + GSM8K)

- 对照组 v1 判对率: **50.8%** (reward 信号正常)
- skew 注入组: v1 子组 30%, v2 子组 0% (strict), 组均值偏移 0.50->0.25
- 逐样本 rewards.jsonl 完整记录, 注入/对照可严格区分
- 训练全程无报错无感知 -> 正是"静默污染"证据

## 正式 Day 2 实验参数 (下一步)

```
模型: Qwen2.5-1.5B-Instruct (RTX_MODEL_DIR / RTX_MODEL_CONFIG=qwen2.5-1.5B.sh)
数据: GSM8K 7473 条 (RTX_DATA_PATH=/root/datasets/gsm8k/dapo-gsm8k-train.jsonl)
额外参数: RTX_EXTRA_MODEL_ARGS="--rotary-base 1000000"
注入: skew / crm_crash / dup (窗口按 group_index, step = group_index//4)
运行: RTX_GPUS='"device=..."' bash scripts/day2_run.sh <fault> <start> <end> 20
```
