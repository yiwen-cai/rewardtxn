#!/usr/bin/env python3
"""
P0: paper 门禁判定 — 对 TRACE_REPORT_PAPER.json 等报告执行 E2/E3 门禁

E2 门禁（§7）:
  G1: 全部切点 failures == 0
  G2: 每切点 n >= 1,500（RQ2 分层下限）
  G3: 总注入 >= 30,000 且聚合 rule-of-three 上界已报告（≈10^-4 量级）
  G4: 每切点上界（Clopper–Pearson / Wilson）如实报告

用法:
  python3 scripts/paper_gates.py --trace-report runs/TRACE_REPORT_PAPER.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CUT_POINTS = ["R1", "R2", "R3", "R4", "R5", "Q0", "Q1", "L0", "L2", "L3", "C1", "C2"]
MIN_PER_CUT = 1500
MIN_TOTAL = 30000


def evaluate_trace(report: dict) -> dict:
    gates = {}
    cuts = report.get("cutpoints", {})
    missing = [c for c in CUT_POINTS if c not in cuts]
    gates["G1_zero_failures"] = all(cuts[c]["failures"] == 0 for c in cuts) and not missing
    gates["G2_min_per_cut"] = all(cuts[c]["n"] >= MIN_PER_CUT for c in cuts) and not missing
    total = report.get("total", {})
    gates["G3_total_30000"] = total.get("n", 0) >= MIN_TOTAL
    gates["G3_rule_of_three_reported"] = total.get("rule_of_three_upper") is not None
    gates["G4_upper_bounds_reported"] = all(
        cuts[c].get("cp_upper") is not None and cuts[c].get("wilson_upper") is not None for c in cuts
    ) and not missing
    verdict = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "summary": {
            "total_n": total.get("n", 0),
            "total_failures": total.get("failures", 0),
            "rule_of_three_upper": total.get("rule_of_three_upper"),
            "missing_cuts": missing,
        },
    }
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-report", type=Path, required=True)
    args = ap.parse_args()

    report = json.loads(args.trace_report.read_text())
    verdict = evaluate_trace(report)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    sys.exit(0 if verdict["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
