#!/usr/bin/env python3
"""Phase 2A 门禁验证工具: CAS 单测 + seals/rewards 分析"""
import json
import os
import sys
import tempfile
from pathlib import Path

K = 8


def cas_unit_test():
    """Reward CAS 幂等单测: 同一 logical_id 二次写入拒绝; 不同 logical_id 放行"""
    import importlib.util
    # mock slime (宿主无容器环境, CAS 逻辑不依赖 RM 数学库)
    import types, sys
    fake = types.ModuleType("slime")
    fake_hub = types.ModuleType("slime.rollout.rm_hub")
    fake_math = types.ModuleType("slime.rollout.rm_hub.math_utils")
    for fn in ("extract_answer", "grade_answer_mathd", "grade_answer_sympy"):
        setattr(fake_math, fn, lambda *a, **k: 0)
    fake_hub.math_utils = fake_math
    fake.rollout = types.ModuleType("slime.rollout")
    fake.rollout.rm_hub = fake_hub
    pkg = types.ModuleType("slime.rollout")
    pkg.rm_hub = fake_hub
    fake.rollout = pkg
    sys.modules["slime"] = fake
    sys.modules["slime.rollout"] = pkg
    sys.modules["slime.rollout.rm_hub"] = fake_hub
    sys.modules["slime.rollout.rm_hub.math_utils"] = fake_math
    spec = importlib.util.spec_from_file_location("p2seal", "scripts/phase2_seal_rm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.RUN_DIR = tempfile.mkdtemp()
    mod._seen_logical.clear()
    class S:
        def __init__(s, gi, idx, rid):
            s.group_index, s.index, s.rollout_id = gi, idx, rid
    # 第一次写入
    ok1 = mod._cas_write({"group_index": 3, "index": 5, "rollout_id": "r1", "reward": 1.0})
    # 同 logical_id 重复 -> 拒绝
    ok2 = mod._cas_write({"group_index": 3, "index": 5, "rollout_id": "r1", "reward": 0.0})
    # 不同 index -> 放行
    ok3 = mod._cas_write({"group_index": 3, "index": 6, "rollout_id": "r1", "reward": 0.0})
    # 不同 rollout_id -> 放行
    ok4 = mod._cas_write({"group_index": 3, "index": 5, "rollout_id": "r2", "reward": 0.0})
    rejects = list(Path(mod.RUN_DIR, "cas_rejects.jsonl").open()) if os.path.exists(os.path.join(mod.RUN_DIR, "cas_rejects.jsonl")) else []
    assert ok1 and ok3 and ok4 and not ok2, (ok1, ok2, ok3, ok4)
    assert len(rejects) == 1 and "duplicate-logical-id" in rejects[0], rejects
    print(f"[CAS] 幂等单测 PASS: 首次写入 {ok1}, 重复拒绝 {not ok2}, 异 index {ok3}, 异 rollout {ok4}, rejects={len(rejects)}")
    return True


def seal_analysis(run_dir: str):
    """分析 seals.jsonl + rewards.jsonl, 输出 2A 门禁结论 JSON"""
    d = Path(run_dir)
    seals = [json.loads(l) for l in (d / "seals.jsonl").open()] if (d / "seals.jsonl").exists() else []
    rewards = [json.loads(l) for l in (d / "rewards.jsonl").open()] if (d / "rewards.jsonl").exists() else []
    aborted = [s for s in seals if s["status"] == "ABORTED"]
    sealed = [s for s in seals if s["status"] == "SEALED"]
    # 注入窗口识别 (meta 里 fault/start/end)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    fault = meta.get("fault_injection", {}).get("mechanism", "?")
    win = meta.get("fault_injection", {})
    start, end = win.get("window_group_start", -1), win.get("window_group_end", -1)
    exp_aborted = (end - start + 1) if start >= 0 else 0
    # 混算拦截验证: ABORTED 组在 rewards 中的版本分布
    mixed_in_rewards = 0
    aborted_groups = {s["group_index"] for s in aborted}
    for s in aborted_groups:
        vers = {r["verifier"] for r in rewards if r["group_index"] == s}
        if len(vers) > 1:
            mixed_in_rewards += 1
    res = {
        "run": run_dir, "fault": fault,
        "seals_total": len(seals), "aborted": len(aborted), "sealed": len(sealed),
        "expected_aborted": exp_aborted,
        "aborted_groups_mixed_in_rewards": mixed_in_rewards,
        "reward_records": len(rewards),
        "g2a1_detection": len(aborted) >= exp_aborted if fault != "none" else None,
        "g2a2_zero_false_positive": len(aborted) == 0 if fault == "none" else None,
    }
    (d / "gate2a.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"[Seal] {run_dir}: fault={fault} seals={len(seals)} "
          f"ABORTED={len(aborted)} SEALED={len(sealed)} 期望ABORTED={exp_aborted} "
          f"混算组={mixed_in_rewards}")
    return res


if __name__ == "__main__":
    cas_unit_test()
    if len(sys.argv) > 1:
        seal_analysis(sys.argv[1])
