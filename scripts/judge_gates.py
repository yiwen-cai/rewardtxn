#!/usr/bin/env python3
"""
自动化 Go / No-Go 门禁判定器 (文档 11 节内嵌脚本)
"""
import json
import sys
from pathlib import Path

def evaluate_run(run_dir):
    run_path = Path(run_dir)
    meta_file = run_path / "meta.json"
    metrics_file = run_path / "metrics.json"

    if not meta_file.exists() or not metrics_file.exists():
        print(f"Error: 缺少元数据或指标文件 in {run_dir}")
        sys.exit(1)

    with open(meta_file, "r") as f:
        meta = json.load(f)
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    # 检查 Go 条件
    g1 = metrics.get("silent_errors_count", 0) >= 2
    g2 = metrics.get("crash_window_reproduced", False) is True
    g3 = metrics.get("gradient_delta_observed", False) is True
    g4 = metrics.get("invalid_committed_steps", 999) == 0
    g5 = metrics.get("replay_savings_pct", 0.0) >= 30.0
    g6 = metrics.get("protocol_overhead_pct", 100.0) < 5.0
    g7 = metrics.get("no_upstream_equivalent", True) is True

    all_go = g1 and g2 and g3 and g4 and g5 and g6 and g7

    verdict = {
        "exp_id": meta.get("exp_id"),
        "auto_pass": all_go,
        "gates": {
            "g1_silent_errors_ge_2": g1,
            "g2_crash_window_reproduced": g2,
            "g3_gradient_delta_observed": g3,
            "g4_zero_invalid_commit": g4,
            "g5_savings_ge_30pct": g5,
            "g6_overhead_lt_5pct": g6,
            "g7_no_upstream_equivalent": g7
        },
        "decision": "GO" if all_go else "NO_GO"
    }

    out_file = run_path / "verdict.json"
    with open(out_file, "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"=== 评审结果: {verdict['decision']} ===")
    for k, v in verdict["gates"].items():
        print(f"  - {k}: {'PASS' if v else 'FAIL'}")

    return 0 if all_go else 1

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python judge_gates.py <run_dir>")
        sys.exit(1)
    sys.exit(evaluate_run(sys.argv[1]))
