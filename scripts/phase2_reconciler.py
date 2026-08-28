#!/usr/bin/env python3
"""
Phase 2C: Reconciler + Selective Replay

Reconciler: 崩溃后恢复决策 (基于 StepToken sidecar + seals.jsonl + rewards.jsonl)
  plan <save_dir> <run_dir> [resume_iter]
    输出 recovery_plan.json:
      - committed_iters       已提交步 (StepToken 完整) -> 不重放
      - resume_iter           恢复起点 (最后已提交步)
      - replay_groups         需重放组 (ABORTED 混算组 / 崩溃不完整组)
      - replay_samples        需重算样本 (缺失 reward 或非权威版本)
      - reissue_tasks         需重新投递任务 (Q0 队列场景, 从外部注入)
      - selective: true       仅重算 Reward, 复用 Rollout

Selective Replay:
  replay <run_dir> <plan.json>
    对 replay_samples 用权威 verifier (v1) 重算 reward (CPU),
    输出 replay_result.json: {samples, correct_rewards, cpu_cost_s, est_savings}
    对比基准: B5 整步重跑 = 30.8s/步 + checkpoint 20-31s (Day4 实测)
"""
import json
import os
import sys
import time
from pathlib import Path

import importlib.util
_spec = importlib.util.spec_from_file_location("p2m", "scripts/phase2_manifest.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

AUTHORITATIVE_VERIFIER = "v1"
B5_STEP_S = 30.8      # Day4 实测 step_time 均值
B5_CKPT_S = 25.0      # Day4 实测保存开销 (20-31s 取中值)
B5_FULL_STEP_S = B5_STEP_S + B5_CKPT_S


def load_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open()]


def build_plan(save_dir: str, run_dir: str, resume_iter=None) -> dict:
    save_dir = Path(save_dir)
    run_dir = Path(run_dir)
    audit = _m.audit(save_dir)
    committed = audit["committed_iters"]
    # 组状态: 最后一次 seal 记录为准 (可能存在预取 buffer 冗余组)
    seals = load_jsonl(run_dir / "seals.jsonl")
    rewards = load_jsonl(run_dir / "rewards.jsonl")
    by_group = {}
    for s in seals:
        by_group[s["group_index"]] = s
    # ABORTED 组 (混算) + 崩溃不完整组 (reward=None 样本存在)
    aborted_groups = [g for g, s in by_group.items() if s["status"] == "ABORTED"]
    group_counts = {}
    for r in rewards:
        group_counts[r["group_index"]] = group_counts.get(r["group_index"], 0) + 1
    crash_groups = {r["group_index"] for r in rewards if r.get("reward") is None}
    # 崩溃组: 存在 reward=None 样本的组 (无论组内是否满 8 条) 需恢复
    incomplete = sorted(g for g in crash_groups if g not in aborted_groups)
    # 重算样本: ABORTED 组全部 (混版本 -> 权威重算) + 崩溃组缺失 reward 样本
    replay_samples = []
    for r in rewards:
        if r["group_index"] in aborted_groups:
            replay_samples.append({"group_index": r["group_index"], "index": r["index"],
                                   "rollout_id": r["rollout_id"],
                                   "response": r.get("response", ""), "label": r.get("label", ""),
                                   "reason": "mixed-version"})
        elif r["group_index"] in incomplete and r.get("reward") is None:
            replay_samples.append({"group_index": r["group_index"], "index": r["index"],
                                   "rollout_id": r["rollout_id"],
                                   "response": r.get("response", ""), "label": r.get("label", ""),
                                   "reason": "crash-missing"})
    replay_groups = sorted(aborted_groups + incomplete)
    resume = resume_iter if resume_iter is not None else (committed[-1] if committed else None)
    plan = {
        "committed_iters": committed,
        "resume_iter": resume,
        "replay_groups": replay_groups,
        "replay_samples": len(replay_samples),
        "aborted_groups": aborted_groups,
        "incomplete_groups": incomplete,
        "selective": True,
        "note": "仅重算 Reward (CPU), 复用 Rollout 前缀; 已提交步不重放",
    }
    (run_dir / "recovery_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
    return plan


def selective_replay(run_dir: str, plan_path=None) -> dict:
    run_dir = Path(run_dir)
    plan = json.loads((run_dir / "recovery_plan.json").read_text() if plan_path is None else Path(plan_path).read_text())
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")) if __file__ else None
    rewards = load_jsonl(run_dir / "rewards.jsonl")
    # 需要重算的样本 (ABORTED 组内所有非权威版本样本 + 缺失样本)
    replay_map = {}
    aborted_set = set(plan["aborted_groups"])
    for r in rewards:
        if r["group_index"] in aborted_set:
            lid = (r["group_index"], r["index"], r["rollout_id"])
            replay_map[lid] = r
        elif r.get("reward") is None:  # 崩溃缺失样本
            lid = (r["group_index"], r["index"], r["rollout_id"])
            replay_map[lid] = r
    t0 = time.perf_counter()
    replayed = []
    for lid, r in replay_map.items():
        resp = r.get("response") or ""
        label = r.get("label") or ""
        # 权威 verifier v1 (与 day2/phase2a 相同实现语义)
        from phase2_seal_rm import _v1_reward
        rw = float(_v1_reward(resp, label))
        replayed.append({"group_index": r["group_index"], "index": r["index"],
                         "rollout_id": r["rollout_id"], "old_verifier": r.get("verifier"),
                         "old_reward": r.get("reward"), "new_reward": rw, "authoritative": True})
    cpu_cost = time.perf_counter() - t0
    # 节省核算: B5 整步重跑 vs Selective Replay (CPU reward 重算)
    n_groups = len(plan["replay_groups"])
    b5_cost = n_groups / 4.0 * B5_FULL_STEP_S          # 组占比换算整步成本 (U=4)
    savings = 1 - cpu_cost / b5_cost if b5_cost > 0 else 0
    res = {
        "replayed_samples": len(replayed),
        "cpu_reward_recompute_s": round(cpu_cost, 3),
        "b5_full_restep_est_s": round(b5_cost, 1),
        "savings_vs_b5": round(savings * 100, 1),
        "samples": replayed,
    }
    (run_dir / "replay_result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def write_audit_report(run_dir, plan: dict, replay: dict = None) -> Path:
    """3C: 恢复审计报告 (Markdown, 人类可读): 已提交步/重放组/重算样本/修正清单"""
    run_dir = Path(run_dir)
    lines = [
        "# 恢复审计报告 (Recovery Audit Report)", "",
        f"- 生成时间: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"- 恢复起点 (resume_iter): {plan.get('resume_iter', 'none')}",
        f"- 已提交步 (StepToken): {plan.get('committed_iters', [])}",
        f"- 需重放组 (ABORTED/崩溃): {len(plan.get('replay_groups', []))}",
        f"- 需重算样本: {plan.get('replay_samples', 0)}",
        f"- 重新投递任务 (Q0): {len(plan.get('reissue_tasks', []))}",
        "", "## 已提交步 (不重放)", "",
    ]
    for it in plan.get("committed_iters", []):
        lines.append(f"- iter {it} (checkpoint 内容哈希绑定, StepToken 幂等)")
    lines += ["", "## 重放组清单", "", "| 组 | 状态 | 版本 | 样本 |", "|---|---|---|---|"]
    for g in plan.get("aborted_groups", []):
        lines.append(f"| {g} | ABORTED | 混合 | 8 |")
    for g in plan.get("replay_groups", []):
        if g not in plan.get("aborted_groups", []):
            lines.append(f"| {g} | 崩溃/不完整 | - | - |")
    lines += ["", "## 重算样本", "", f"共 {plan.get('replay_samples', 0)} 个样本 (权威 verifier v1, CPU 重算)"]
    if replay:
        lines += ["", "## 重算结果", "",
                  f"- 重算样本: {replay.get('replayed_samples', 0)}",
                  f"- CPU 重算耗时: {replay.get('cpu_reward_recompute_s', 0):.2f}s",
                  f"- B5 整步重跑预估: {replay.get('b5_full_restep_est_s', 0):.1f}s",
                  f"- 节省 vs B5: {replay.get('savings_vs_b5', 0):.1f}%"]
    out = run_dir / "recovery_audit_report.md"
    out.write_text("\n".join(lines) + "\n")
    return out


def main():
    mode = sys.argv[1]
    if mode == "plan":
        r = build_plan(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else None)
        print(json.dumps({k: v for k, v in r.items() if k != "note"}, indent=1, ensure_ascii=False))
    elif mode == "replay":
        r = selective_replay(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        print(json.dumps({"replayed_samples": r["replayed_samples"], "cpu_s": r["cpu_reward_recompute_s"],
                          "b5_est_s": r["b5_full_restep_est_s"], "savings_vs_b5_pct": r["savings_vs_b5"]}, indent=1))
    elif mode == "report":
        run_dir = sys.argv[2]
        plan = json.loads((Path(run_dir) / "recovery_plan.json").read_text())
        rep = None
        rp = Path(run_dir) / "replay_result.json"
        if rp.exists():
            rep = json.loads(rp.read_text())
        out = write_audit_report(run_dir, plan, rep)
        print(f"report written: {out}")


if __name__ == "__main__":
    main()
