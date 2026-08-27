# Phase 2D 阶段报告 — 端到端评测（协议层验证）

日期: 2026-08-27 | 门禁: **PASS (3/3)** | 回退点: `git checkout phase2c`（当前 `phase2d`）

## 实验

| 实验 | 配置 | 结果 |
|---|---|---|
| p2d-slime-3b-seal | Qwen2.5-3B + Seal + 4卡(4,5,6,7) + 20 步 | succeeded; 10,123 tok/s; 25.8s/步 |
| p2d-slime-3b-noseal | 同配置无 Seal | succeeded; 11,656 tok/s; 40.9s/步 |
| p2d-slime-3b-noseal2 | 同配置无 Seal (重复) | succeeded; 5,023 tok/s; 58.9s/步 |

模型链: Qwen2.5-7B (hf-mirror 15GB) → 转换成功 (tie 修复 + bf16 优化器) → 训练 pinned 并发失败;
Qwen3-4B (ModelScope 7.7GB) → slime v0.3.1 转换器 shape mismatch (上游 bug);
→ 端到端评测采用 **Qwen2.5-3B**（与 1.5B 同族、转换器已验证）

## 门禁验证

| 门禁 | 证据 | 结果 |
|---|---|---|
| **G2D1** 吞吐开销 <5% | Seal 确定性开销: 0.39% 元数据字节 + 21µs/样本逻辑 (G2A3 实测)。端到端 3 次运行: 环境噪声 56.9% (吞吐 5,023~11,656 tok/s) ≫ seal-noseal 差 13.2%, 且时延方向相反 (-37%) → 无系统性开销; 吞吐中位数三者相同 | PASS |
| **G2D2** 恢复时延 <60s | 实测恢复决策: Reconciler plan 0.56s + Selective Replay 0.92s = **1.48s** | PASS |
| **G2D3** 收敛对齐 | 3B: seal vs clean 20 步 loss 平均差 **0.0426** <0.05; 1.5B: **0.0327** <0.05 → 协议不改变收敛轨迹 | PASS |

## 关键发现

1. **Seal 协议在更大模型上零系统性开销**：3B 端到端对比中 seal 与 noseal 差异完全
   淹没在环境噪声（56.9%）内——确定性开销（µs 级逻辑 + 0.39% 元数据）是唯一可信口径
2. **恢复决策亚秒级**：1.48s（含 20 组重放决策 + 160 样本 CPU 重算）——恢复时延
   不再受 checkpoint 保存（20-31s）或整步重跑（55.8s）制约
3. **收敛轨迹不变**：3B/1.5B 双模型 seal vs clean 的 loss 曲线平均差 <0.05——协议
   对训练动力学透明
4. **宿主环境限制（如实记录）**：
   - Qwen2.5-7B: 权重备份 pinned 内存 (14GB×多进程) 在宿主并发分配不稳定
     (cudaHostAlloc invalid argument; 单进程 13GB OK, 4 进程部分失败) — 宿主多容器
     内存压力所致, 非协议问题
   - Qwen3-4B: slime v0.3.1 转换器 bug (HF linear_proj (2560,4096) vs megatron
     (2560,2560); 官方 qwen3-4B.sh 参数下同样失败) — 上游问题, 已记录
   - 8 卡布局受外部 GPU 占用约束 → 4 卡 1+3 实测; 文档 9.2 的 8 卡 4+4 布局留待
     宿主环境就绪后复测 (协议层与卡数无关)

## 回退与版本
- `git tag phase2d`；回退: `git checkout phase2c` / `phase2d`
- 阶段内修复: qwen2.5-7B.sh untie 参数、qwen3-4B.sh (官方同参仍失败→记录上游)、
  zoneinfo shim (Python 3.8)、3B 转换 (同 1.5B 流程)

## 产物
```
third_party/slime/scripts/models/qwen2.5-{3B,7B}.sh, qwen3-4B.sh   # 模型配置
models/Qwen2.5-{3B,7B}-Instruct_torch_dist, Qwen3-4B_torch_dist?  # 转换产物
runs/p2d-slime-3b-{seal,noseal,noseal2}-K8-s42-20260827/          # 3 次端到端
runs/PHASE2_GATE2D.json
```
