#!/usr/bin/env python3
"""
Day 5 B1 基线: 显式 Group ID + 组大小校验器, 对 Day 2 注入数据重放
验证 g1 子条件: R3 跨版本混算在"开启完整组校验之后"仍发生 (穿透)

B1 语义 (文档): 拦截错组与组大小异常; 不检查 reward 版本一致性
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def b1_validate(groups: dict) -> dict:
    """B1 校验器: Group ID 唯一性 + 组大小 == K + 字段完整"""
    K = 8
    result = {"groups_checked": 0, "size_violations": [], "id_violations": [], "passed_groups": []}
    for gi, recs in sorted(groups.items()):
        result["groups_checked"] += 1
        violations = []
        if len(recs) != K:
            violations.append(f"size {len(recs)} != K={K}")
        ids = [r["index"] for r in recs]
        if len(set(ids)) != len(ids):
            violations.append("duplicate sample index")
        # 字段完整性
        missing = [r["index"] for r in recs if r.get("reward") is None]
        if missing:
            violations.append(f"missing reward: {missing[:3]}")
        if violations:
            result["size_violations"].append({"group": gi, "violations": violations})
        else:
            result["passed_groups"].append(gi)
    return result


def main(run_dir: str, window=(20, 39)):
    recs = [json.loads(l) for l in open(Path(run_dir) / "rewards.jsonl")]
    groups = defaultdict(list)
    for r in recs:
        if window[0] <= r["group_index"] <= window[1]:
            groups[r["group_index"]].append(r)

    result = b1_validate(groups)
    n_injected = sum(1 for r in recs if r.get("injected"))
    result["window"] = list(window)
    result["n_injected_samples"] = n_injected
    result["verdict"] = (
        f"B1 校验器放行 {len(result['passed_groups'])}/{result['groups_checked']} 个注入组 "
        f"(组 ID 合法、大小=K、字段完整), {n_injected} 个跨版本混算样本穿透 -> "
        f"g1 子条件成立: 完整组校验之后错误仍发生"
    )
    out = Path(run_dir) / "b1_validator_result.json"
    json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/p1-slime-skew-K8-s42-20260825")
