#!/usr/bin/env python3
"""
Day 1 Lineage 提取器: 从训练日志解析 Step 级指标并构建最小 Lineage
用法: python scripts/day1_lineage.py runs/{exp_id}

产物:
  manifests/steps.jsonl   - 每步: step, step_token(=step), 时延/吞吐/是否已 checkpoint
  manifests/groups.jsonl  - 每步 U=4 组 × K=8 轨迹的确定性描述 (provenance=derived:
                            日志仅含聚合指标, 单组信息由配置静态推导)
  metrics.json            - throughput_base, step_latency 等 (Day 5 judge_gates 输入)
"""
import ast
import json
import re
import statistics
import sys
from pathlib import Path

STEP_RE = re.compile(r"step (\d+): (\{.*\})")
PERF_RE = re.compile(r"perf (\d+): (\{.*\})")
CKPT_RE = re.compile(r"saving checkpoint at iteration\s+(\d+)")
ROLLOUT_RE = re.compile(r"rollout (\d+): (\{.*\})")


def _parse_dict(s: str):
    """日志中的 Python dict repr (单引号) -> dict"""
    return ast.literal_eval(s)


def main(run_dir: str):
    run_path = Path(run_dir)
    log_file = run_path / "logs" / "train.log"
    if not log_file.exists():
        print(f"Error: {log_file} 不存在")
        sys.exit(1)

    with open(meta_file := run_path / "meta.json") as f:
        meta = json.load(f)
    K = meta["group_size_K"]
    U = meta["batch_groups_U"]

    steps, perfs, ckpts, rollouts = {}, [], [], []
    for line in log_file.open(errors="replace"):
        m = STEP_RE.search(line)
        if m:
            try:
                steps[int(m.group(1))] = _parse_dict(m.group(2))
            except Exception:
                pass
        m = PERF_RE.search(line)
        if m:
            try:
                perfs.append((int(m.group(1)), _parse_dict(m.group(2))))
            except Exception:
                pass
        m = CKPT_RE.search(line)
        if m:
            ckpts.append(int(m.group(1)))
        m = ROLLOUT_RE.search(line)
        if m:
            try:
                rollouts.append((int(m.group(1)), _parse_dict(m.group(2))))
            except Exception:
                pass

    if not perfs:
        print("Error: 日志中未解析到 perf 指标, 训练可能未启动或已失败")
        sys.exit(1)

    # ---------- steps.jsonl ----------
    manif_dir = run_path / "manifests"
    manif_dir.mkdir(exist_ok=True)
    n_ckpt = set(ckpts)
    # 仅保留 actor 训练侧 perf (含 perf/step_time 键); rollout 侧指标单独收集
    actor_perfs = sorted((s, p) for s, p in perfs if "perf/step_time" in p)
    rollout_perfs = sorted((s, p) for s, p in perfs if "perf/rollout_time" in p)
    with open(manif_dir / "steps.jsonl", "w") as f:
        for step, p in actor_perfs:
            rec = {
                "step": step,
                "step_token": step,  # Day1 占位: step 序号即 StepToken
                "step_time_s": p.get("perf/step_time"),
                "actor_train_time_s": p.get("perf/actor_train_time"),
                "wait_time_s": p.get("perf/train_wait_time"),
                "update_weights_time_s": p.get("perf/update_weights_time"),
                "actor_train_tok_per_s": p.get("perf/actor_train_tok_per_s"),
                "actor_train_tflops": p.get("perf/actor_train_tflops"),
                "checkpointed": step in n_ckpt,
                "checkpoint_iter": step if step in n_ckpt else None,
            }
            f.write(json.dumps(rec) + "\n")

    # ---------- groups.jsonl (derived lineage) ----------
    with open(manif_dir / "groups.jsonl", "w") as f:
        for step, _ in actor_perfs:
            for g in range(U):
                base = (step * U + g) * K
                rec = {
                    "group_id": f"s{step}g{g}",
                    "step": step,
                    "group_contract": {"K": K, "n_samples": K},
                    "trajectory_ids": [f"t{base + i}" for i in range(K)],
                    "reward_plan_digest": "deepscaler-rb-v1",  # 确定性 rule-based RM
                    "provenance": "derived",  # 日志仅聚合, 单组映射为静态推导
                }
                f.write(json.dumps(rec) + "\n")

    # ---------- metrics.json ----------
    step_times = [p.get("perf/step_time") for _, p in actor_perfs if p.get("perf/step_time")]
    tok_rates = [p.get("perf/actor_train_tok_per_s") for _, p in actor_perfs
                 if p.get("perf/actor_train_tok_per_s")]
    wait_ratios = [p.get("perf/wait_time_ratio") for _, p in actor_perfs
                   if p.get("perf/wait_time_ratio")]
    rollout_times = [p.get("perf/rollout_time") for _, p in rollout_perfs
                     if p.get("perf/rollout_time")]
    rollout_tok_per_gpu = [p.get("perf/tokens_per_gpu_per_sec") for _, p in rollout_perfs
                           if p.get("perf/tokens_per_gpu_per_sec")]
    metrics = {
        "exp_id": meta["exp_id"],
        "n_steps": len(actor_perfs),
        "throughput_base_tok_per_s": round(statistics.mean(tok_rates), 2) if tok_rates else None,
        "step_latency_mean_s": round(statistics.mean(step_times), 3) if step_times else None,
        "step_latency_median_s": round(statistics.median(step_times), 3) if step_times else None,
        "step_latency_max_s": round(max(step_times), 3) if step_times else None,
        "wait_time_ratio_mean": round(statistics.mean(wait_ratios), 4) if wait_ratios else None,
        "rollout_time_mean_s": round(statistics.mean(rollout_times), 3) if rollout_times else None,
        "rollout_tokens_per_gpu_per_s": round(statistics.mean(rollout_tok_per_gpu), 2)
        if rollout_tok_per_gpu else None,
        "checkpoint_count": len(ckpts),
        "checkpoint_iters": sorted(ckpts),
        "group_size_K": K,
        "batch_groups_U": U,
        # Day 5 门禁输入字段 (Day 1 无故障基线, 留空/0)
        "silent_errors_count": 0,
        "crash_window_reproduced": False,
        "gradient_delta_observed": False,
        "invalid_committed_steps": 0,
        "replay_savings_pct": 0.0,
        "protocol_overhead_pct": 0.0,
        "no_upstream_equivalent": True,
    }
    with open(run_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"steps: {len(actor_perfs)}, checkpoints: {sorted(ckpts)}, groups: {len(actor_perfs)*U}")
    print(f"throughput_base: {metrics['throughput_base_tok_per_s']} tok/s, "
          f"step_latency_mean: {metrics['step_latency_mean_s']} s")
    print(f"written: {manif_dir}/steps.jsonl, {manif_dir}/groups.jsonl, {run_path}/metrics.json")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python day1_lineage.py runs/{exp_id}")
        sys.exit(1)
    main(sys.argv[1])
