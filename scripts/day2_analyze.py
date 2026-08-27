#!/usr/bin/env python3
"""Day 2 注入效果分析器
用法: python scripts/day2_analyze.py runs/{exp_id}
输出: runs/{exp_id}/analysis.json + 控制台摘要
"""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


def main(run_dir: str):
    rp = Path(run_dir)
    meta = json.load(open(rp / "meta.json"))
    recs = [json.loads(l) for l in open(rp / "rewards.jsonl")]
    fi = meta.get("fault_injection", {})
    # 窗口从 target_step 解析
    import re
    m = re.search(r"\[(\d+),(\d+)\]", str(fi.get("target_step", "")))
    start, end = (int(m.group(1)), int(m.group(2))) if m else (-1, -2)

    win = [r for r in recs if start <= r["group_index"] <= end]
    out = [r for r in recs if not (start <= r["group_index"] <= end)]

    def acc(rs):
        return (sum(1 for r in rs if r["reward"] == 1) / len(rs)) if rs else None

    res = {
        "exp_id": meta["exp_id"],
        "fault": fi.get("crash_point"),
        "window": [start, end],
        "n_total": len(recs),
        "n_window": len(win),
        "n_control": len(out),
        "control_acc": acc(out),
        "window_acc": acc(win),
        "window_v1_acc": acc([r for r in win if r["verifier"] == "v1"]),
        "window_v2_acc": acc([r for r in win if r["verifier"] == "v2"]),
        "injected_count": sum(1 for r in recs if r.get("injected")),
    }

    # 组级统计量偏移 (窗口内, 混入前后对比)
    if res["fault"] in ("R3", "R2"):
        group_deltas = []
        for gi in sorted(set(r["group_index"] for r in win)):
            g = [r for r in win if r["group_index"] == gi]
            v1 = [r for r in g if r["verifier"] == "v1"]
            v2 = [r for r in g if r["verifier"] != "v1"]  # 注入子组 (v2 / retry-stale)
            if v1 and v2:
                # 假设纯 v1: 组均值 = v1 子组的 v1 分值 + v2 子组若用 v1 的分值(用 v1 分布期望替代)
                # 简化: 用 v1 子组均值作为整组 v1-期望
                delta = acc(g) - acc(v1)
                group_deltas.append({"group": gi, "v1_mean": acc(v1), "actual_mean": acc(g), "delta": delta})
        res["group_deltas"] = group_deltas
        res["group_delta_mean"] = round(statistics.mean(d["delta"] for d in group_deltas), 4) if group_deltas else None

    # 训练侧: step loss 对比 (注入窗口步 vs 其他步)
    steps = []
    import re as _re
    step_re = _re.compile(r"step (\d+): (\{.*\})")
    for line in (rp / "logs" / "train.log").open(errors="replace"):
        m = step_re.search(line)
        if m:
            try:
                import ast
                steps.append((int(m.group(1)), ast.literal_eval(m.group(2))))
            except Exception:
                pass
    res["n_steps"] = len(steps)

    json.dump(res, open(rp / "analysis.json", "w"), indent=2, ensure_ascii=False)
    print(f"=== {meta['exp_id']} (fault={res['fault']}) ===")
    print(f"样本: 注入窗口 {res['n_window']} / 对照 {res['n_control']} / 总 {res['n_total']}")
    def pct(v):
        return f"{v*100:.1f}%" if v is not None else "N/A"

    print(f"判对率: 对照(v1) {pct(res['control_acc'])} | 窗口内 v1 {pct(res['window_v1_acc'])} | 注入子组 {pct(res['window_v2_acc'])}")
    if res.get("group_delta_mean") is not None:
        print(f"组统计量偏移均值: {res['group_delta_mean']} (混入 v2 后组均值下降)")
    print(f"injected 样本数: {res['injected_count']}")
    print(f"训练步数: {res['n_steps']}")


if __name__ == "__main__":
    main(sys.argv[1])
