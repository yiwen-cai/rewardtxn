#!/usr/bin/env python3
"""Summarize the frozen minimal FT formal matrix from per-pair records.

Reads minimal_evidence/formal-*-pair.json (only formal_pair_verified pairs count;
stopped/technically-invalid attempts are listed but excluded) and each run's cost.json.
Applies the frozen rule: exact two-sided McNemar only if all 10 F2 pairs are safe and decidable.
"""
import argparse
import glob
import json
import math
import os
import statistics

SAFE = {"correct_recovered", "safe_discard", "no_fault_verified"}


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(200):
            mid = (lo + hi) / 2
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) >= alpha / 2)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= alpha / 2)
    return [lower, upper]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_cdf(min(b, c), n, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="docs/experiments/rewardtxn-ft-20260916/minimal_evidence")
    ap.add_argument("--out", default="docs/experiments/rewardtxn-ft-20260916/FORMAL_RESULTS_20260925.json")
    args = ap.parse_args()

    pairs, excluded = [], []
    for path in sorted(glob.glob(os.path.join(args.evidence, "formal-*-pair.json"))):
        d = json.load(open(path))
        if d.get("status") != "formal_pair_verified":
            excluded.append({"name": d["name"], "status": d.get("status"), "error": d.get("error")})
            continue
        runs = {}
        for r in d["runs"]:
            cost = json.load(open(os.path.join(r["evidence"], "cost.json")))
            src = r.get("source", {})
            runs[r["arm"]] = {
                "classification": r["classification"],
                "reused_rows": src.get("same_execution_source_rows"),
                "target_rows": src.get("target_rows"),
                "all_reused": bool(src.get("all_32_reused")),
                "generated_tokens_after_fault": src.get("generated_tokens_after_fault"),
                "wall_seconds": cost["wall_seconds"],
                "gpu_hours": cost["allocated_gpu_hours"],
                "peak_incremental_bytes": r["disk_peak"]["peak_incremental_bytes"],
                "storage": r.get("storage"),
            }
        pairs.append({"name": d["name"], "scenario": d["scenario"], "seed": d["seed"],
                      "order": d["order"], "freeze_sha256": d["freeze_sha256"], "runs": runs})

    f2 = [p for p in pairs if p["scenario"] == "F2"]
    all_safe = len(f2) == 10 and all(r["classification"] in SAFE for p in f2 for r in p["runs"].values())
    a_succ = sum(p["runs"]["A"]["all_reused"] for p in f2)
    r_succ = sum(p["runs"]["R"]["all_reused"] for p in f2)
    b = sum(p["runs"]["R"]["all_reused"] and not p["runs"]["A"]["all_reused"] for p in f2)
    c = sum(p["runs"]["A"]["all_reused"] and not p["runs"]["R"]["all_reused"] for p in f2)

    def med(scn, arm, key):
        v = [p["runs"][arm][key] for p in pairs if p["scenario"] == scn and p["runs"][arm][key] is not None]
        return {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v)} if v else None

    out = {
        "scope": "frozen minimal FT matrix (AReaL branch, 10 updates/run); formal pairs only",
        "pairs": pairs,
        "excluded_attempts": excluded,
        "f2": {
            "pairs": len(f2),
            "all_safe_and_decidable": all_safe,
            "A_success": a_succ, "R_success": r_succ,
            "A_exact95": clopper_pearson(a_succ, len(f2)),
            "R_exact95": clopper_pearson(r_succ, len(f2)),
            "discordant_R_only": b, "discordant_A_only": c,
            "mcnemar_exact_two_sided_p": mcnemar_exact(b, c) if all_safe else None,
            "alpha": 0.05,
        },
        "cost": {scn: {arm: {k: med(scn, arm, k) for k in
                             ("wall_seconds", "gpu_hours", "peak_incremental_bytes", "generated_tokens_after_fault")}
                       for arm in ("A", "R")} for scn in ("F2", "no_fault")},
        "cost_note": "descriptive only; no-fault cost is not subtracted to infer fault benefit (frozen rule)",
    }
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({"f2": out["f2"], "excluded": excluded}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
