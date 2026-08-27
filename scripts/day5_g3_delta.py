#!/usr/bin/env python3
"""
Day 5 g3 受控实验: 真实注入数据 -> Optimizer Delta 漂移可测

方法: 用 Day 2 R3_skew 实验的逐样本 reward 数据 (同一样本 v1 vs 注入值),
构造同一组样本的干净/污染 Advantage, 模拟策略梯度, 对比权重更新 Delta
与余弦相似度。同数据、仅 reward 不同 -> 归因干净 (rollout 随机性被消除)。
"""
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch


def group_advantage(rewards, epsilon=1e-6):
    """GRPO 组归一化 Advantage (与 slime 语义一致)"""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + epsilon)


def load_real_samples(run_dir: str, n_groups: int = 8, window=(20, 39)):
    """从 rewards.jsonl 加载注入窗口内的完整组 (每组 8 条, 含 v1 与注入值)"""
    recs = [json.loads(l) for l in open(Path(run_dir) / "rewards.jsonl")]
    groups = defaultdict(list)
    for r in recs:
        if window[0] <= r["group_index"] <= window[1]:
            groups[r["group_index"]].append(r)
    out = []
    for gi in sorted(groups):
        g = groups[gi]
        if len(g) == 8:
            out.append(g)
        if len(out) >= n_groups:
            break
    return out


def simulate_delta(groups, seed=42, lr=0.1, d_feat=16):
    """对每组: 干净 advantage (全 v1) vs 污染 advantage (后 4 条用注入值),
    线性策略梯度 -> 权重 Delta 对比"""
    torch.manual_seed(seed)
    deltas_clean, deltas_fault = [], []
    for g in groups:
        r_clean = torch.tensor([x["reward"] if x["verifier"] == "v1" else x["reward"] for x in g])
        # 干净组: 全部按 v1 计算 (用同组 v1 样本的 reward; 注入样本若用 v1 会是什么值?
        # 无法重算, 用"注入样本在对照组的 v1 判对率期望"近似 -> 用组内 v1 均值替代)
        v1_vals = [x["reward"] for x in g if x["verifier"] == "v1"]
        v1_mean = statistics.mean(v1_vals) if v1_vals else 0.0
        r_clean = torch.tensor([x["reward"] if x["verifier"] == "v1" else v1_mean for x in g])
        r_fault = torch.tensor([x["reward"] for x in g])

        adv_clean = group_advantage(r_clean)
        adv_fault = group_advantage(r_fault)

        feats = torch.randn(len(g), d_feat)
        w = torch.randn(d_feat)
        # 梯度 = -mean(adv * feat) (简化策略梯度)
        grad_clean = -(feats * adv_clean.unsqueeze(1)).mean(0)
        grad_fault = -(feats * adv_fault.unsqueeze(1)).mean(0)
        deltas_clean.append(lr * grad_clean)
        deltas_fault.append(lr * grad_fault)

    d_clean = torch.stack(deltas_clean)
    d_fault = torch.stack(deltas_fault)
    l2 = torch.norm(d_clean - d_fault).item()
    cos = torch.nn.functional.cosine_similarity(d_clean.view(1, -1), d_fault.view(1, -1)).item()
    return l2, cos, len(groups)


def main(run_dir: str):
    groups = load_real_samples(run_dir)
    print(f"真实注入组: {len(groups)} 组 (每组 8 条, 后 4 条为注入值)")
    l2, cos, n = simulate_delta(groups)
    print(f"权重更新 Delta L2 偏差: {l2:.6f}")
    print(f"梯度方向余弦相似度: {cos:.6f}")
    result = {
        "experiment": "g3_controlled_delta",
        "source": run_dir,
        "n_groups": n,
        "delta_l2": l2,
        "cosine_similarity": cos,
        "gradient_delta_observed": l2 > 1e-4 and cos < 0.999,
        "note": "同一样本、仅 reward 版本混入 -> Delta 漂移可归因",
    }
    out = Path(run_dir) / "g3_controlled_delta.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"written: {out}")


if __name__ == "__main__":
    main("runs/p1-slime-skew-K8-s42-20260825")
