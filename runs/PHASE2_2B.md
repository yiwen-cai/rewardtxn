# Phase 2B 阶段报告 — StepManifest + StepToken（持久化面）

日期: 2026-08-27 | 门禁: **PASS (3/3)** | 回退点: `git checkout phase2a`（当前 `phase2b`）

## 设计

`scripts/phase2_manifest.py`（零 third_party 改动）：
- **StepToken** = `RTX-ST-<iter>-<weights_hash[:16]>`：权重内容采样哈希绑定
  （头部+尾部 4KB/文件 → 检测同 size 内容替换，对 mtime 迁移鲁棒）
- **StepManifest** = sidecar `step_token_<iter>.json`：{token, iter, weights_hash,
  prev_token, prev_iter, ts}——链式 prev 引用形成持久化提交链
- **存储位置**：`SAVE_DIR` 上级 `manifests/`（checkpoint 目录由容器 root 所有，
  宿主无写权限；token 内容哈希绑定 checkpoint = 等同嵌入）
- **watch 模式**：轮询 `latest_checkpointed_iteration.txt`，新 checkpoint 自动
  补写 sidecar（兼容 megatron `iter_0000003` 命名）

## 门禁验证

| 门禁 | 证据 | 结果 |
|---|---|---|
| **G2B1** save 时 manifest+token 落盘 | 真实训练 20 步 (save-interval 4): iter 3,7,11,15,19 全部生成 sidecar，prev 链完整 | PASS |
| **G2B2** kill 后识别已提交步 | Day4 L2 真实 kill 场景遗留 checkpoint: audit → 已提交步 **[3,7]**，恢复决策从 iter 7 继续（不重放） | PASS |
| **G2B3** token 幂等/唯一 | 单测: 同内容 token 稳定 / 同 size 异内容哈希不同 / mtime 迁移鲁棒 / NO_TOKEN 缺失识别 | PASS |

## 关键发现

1. **崩溃后"已提交步"可精确识别**：Day 4 双重死锁场景的遗留 checkpoint 现在有
   明确的提交语义——协议层知道 iter 7 是最后提交步（token 绑定内容，防篡改）
2. **checkpoint 目录权限边界**：容器 root 创建、宿主不可写 → sidecar 外部存储 +
   内容哈希绑定（等同嵌入语义）
3. **恢复死锁（Day4）是框架层问题**：协议层识别已交付；"从 iter 7 重建并继续训练"
   的框架级包装由 2C Reconciler 实现（通过包装层绕开 num_rollout 匹配死锁）

## 回退与版本
- `git tag phase2b`；回退: `git checkout phase2a` / `phase2b`
- 本阶段一次环境失败（GPU7 外部抢占 → SGLang init 失败；CUDA invalid argument），
  均重试成功，与协议逻辑无关

## 产物
```
scripts/phase2_manifest.py            # watch/audit/token + 命名兼容 + 内容采样哈希
runs/p2b-slime-ckpt-K8-s42-20260827/manifests/   # 5 个 StepToken sidecar (真实训练)
runs/p1-slime-L2-K8-s42-20260825/manifests/      # kill 场景 sidecar [3,7]
runs/PHASE2_GATE2B.json               # 门禁 JSON
```
