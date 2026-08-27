#!/usr/bin/env python3
"""
Phase 0: 确定性数学与梯度污染差分 Oracle (文档 6.2 节内嵌脚本)
验证: 单样本扰动在组归一化下的误差扩散
"""
import torch
import torch.nn as nn
import torch.optim as optim

def compute_group_advantage(rewards, epsilon=1e-6):
    """GRPO 组归一化 Advantage 计算"""
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + epsilon)

def run_oracle_test():
    torch.manual_seed(42)
    # 模拟简单的线性策略网络
    model_clean = nn.Linear(16, 1, bias=False)
    model_fault = nn.Linear(16, 1, bias=False)
    model_fault.load_state_dict(model_clean.state_dict())

    opt_clean = optim.SGD(model_clean.parameters(), lr=0.1)
    opt_fault = optim.SGD(model_fault.parameters(), lr=0.1)

    # 构造 K=4 的一组特征输入
    inputs = torch.randn(4, 16)

    # 1. 干净奖励组 [0.0, 0.0, 1.0, 1.0] -> 均值 0.5, 方差 0.577
    r_clean = torch.tensor([0.0, 0.0, 1.0, 1.0])
    adv_clean = compute_group_advantage(r_clean)

    # 2. 注入故障: 重复消费导致 0 被替换为 1 -> [0.0, 1.0, 1.0, 1.0]
    r_fault = torch.tensor([0.0, 1.0, 1.0, 1.0])
    adv_fault = compute_group_advantage(r_fault)

    print(f"Clean Advantage: {adv_clean.tolist()}")
    print(f"Fault Advantage: {adv_fault.tolist()}")

    # 模拟策略损失 Loss = - (log_prob * Advantage)
    logits_clean = model_clean(inputs).squeeze(-1)
    loss_clean = -(logits_clean * adv_clean).mean()
    opt_clean.zero_grad()
    loss_clean.backward()
    opt_clean.step()

    logits_fault = model_fault(inputs).squeeze(-1)
    loss_fault = -(logits_fault * adv_fault).mean()
    opt_fault.zero_grad()
    loss_fault.backward()
    opt_fault.step()

    # 计算参数更新 Delta
    delta_clean = model_clean.weight.data
    delta_fault = model_fault.weight.data
    l2_diff = torch.norm(delta_clean - delta_fault).item()
    cos_sim = torch.cosine_similarity(delta_clean.view(-1), delta_fault.view(-1), dim=0).item()

    print(f"\n=== Phase 0 差分结果 ===")
    print(f"权重更新 L2 偏差: {l2_diff:.6f}")
    print(f"梯度方向余弦相似度: {cos_sim:.6f}")

    # 判定
    if l2_diff > 1e-4 and cos_sim < 0.999:
        print("\n>> PASS: 成功证明组内单样本故障导致不可逆的 Optimizer 参数污染！")
        return True
    else:
        print("\n>> FAIL: 误差未造成显著梯度偏差，请检查超参设定。")
        return False

if __name__ == "__main__":
    run_oracle_test()
