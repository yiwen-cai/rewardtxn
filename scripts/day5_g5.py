#!/usr/bin/env python3
"""
Day 5 g5: Selective Replay 重算节省核算 (对比 B2/B5)

单位: Rollout GPU·s (重算昂贵部分 = rollout; Reward 为 CPU rule-based)
数据来源: Day 2 注入 (R1/R2/R3) + Day 4 B5 实测 (checkpoint 开销/重跑成本)
"""
import json

# ---- 实测输入 ----
STEP_ROLLOUT_S = 31.3      # Day 2 perf: rollout_time_mean
STEP_TRAIN_S = 30.8        # Day 4: step_time 均值
B5_SAVE_S = [31.0, 20.0]   # Day 4: checkpoint 保存耗时
REWARD_RECOMPUTE_S = 0.1   # CPU rule-based reward 重算 (毫秒~秒级)
GROUP_REPLAY_OVERHEAD_S = 1.0  # Selective Replay 保守计入调度/重验开销/组

# 场景定义: (名称, 丢失组数, 每步组数, 说明)
SCENARIOS = [
    ("R1_crash_32samples", 4, 4, "4 组 32 样本 (RM 崩溃/整组丢弃)"),
    ("R3_skew_20groups", 20, 4, "20 组 80 样本 (跨版本混算 ABORTED)"),
    ("R2_dup_20groups", 20, 4, "20 组 80 样本 (陈旧重试混入 ABORTED)"),
    ("Q0_lost_32samples", 4, 4, "32 样本 (队列消费后崩溃)"),
]


def compute():
    rows = []
    for name, n_groups, groups_per_step, desc in SCENARIOS:
        n_steps = n_groups / groups_per_step
        # B2: 丢组 -> 整组 rollout 重生成 (100% 浪费)
        b2_cost = n_steps * STEP_ROLLOUT_S
        # B5: 整步重跑 (rollout 重生成 + 训练) + checkpoint 保存开销
        b5_cost = n_steps * (STEP_ROLLOUT_S + STEP_TRAIN_S) + n_steps * (sum(B5_SAVE_S) / len(B5_SAVE_S))
        # Selective Replay: 仅重算 reward + 保守调度开销 (复用 rollout 前缀)
        sr_cost = n_groups * (REWARD_RECOMPUTE_S + GROUP_REPLAY_OVERHEAD_S)
        rows.append({
            "scenario": name, "desc": desc,
            "b2_cost_gpu_s": round(b2_cost, 1),
            "b5_cost_gpu_s": round(b5_cost, 1),
            "selective_replay_cost_s": round(sr_cost, 1),
            "savings_vs_b2_pct": round((1 - sr_cost / b2_cost) * 100, 1),
            "savings_vs_b5_pct": round((1 - sr_cost / b5_cost) * 100, 1),
        })
    # 汇总 (全部场景加权平均)
    tot_b2 = sum(r["b2_cost_gpu_s"] for r in rows)
    tot_b5 = sum(r["b5_cost_gpu_s"] for r in rows)
    tot_sr = sum(r["selective_replay_cost_s"] for r in rows)
    summary = {
        "scenarios": rows,
        "total_savings_vs_b2_pct": round((1 - tot_sr / tot_b2) * 100, 1),
        "total_savings_vs_b5_pct": round((1 - tot_sr / tot_b5) * 100, 1),
        "g5_savings_ge_30pct": (1 - tot_sr / tot_b2) * 100 >= 30.0,
        "assumptions": {
            "rollout_cost_per_step_s": STEP_ROLLOUT_S,
            "train_cost_per_step_s": STEP_TRAIN_S,
            "checkpoint_save_cost_s": B5_SAVE_S,
            "reward_recompute_s": REWARD_RECOMPUTE_S,
            "replay_overhead_per_group_s": GROUP_REPLAY_OVERHEAD_S,
            "note": "Selective Replay 复用 Rollout 前缀, 仅重算失败 Reward (CPU)",
        },
    }
    json.dump(summary, open("runs/DAY5_G5.json", "w"), indent=2, ensure_ascii=False)
    for r in rows:
        print(f"{r['scenario']:24s} B2={r['b2_cost_gpu_s']:7.1f}s B5={r['b5_cost_gpu_s']:7.1f}s "
              f"SR={r['selective_replay_cost_s']:6.1f}s 节省vsB2={r['savings_vs_b2_pct']:5.1f}% "
              f"vsB5={r['savings_vs_b5_pct']:5.1f}%")
    print(f"总计: vs B2 {summary['total_savings_vs_b2_pct']}%, vs B5 {summary['total_savings_vs_b5_pct']}%")
    print(f"g5 达标: {summary['g5_savings_ge_30pct']}")


if __name__ == "__main__":
    compute()
