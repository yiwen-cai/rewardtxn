#!/usr/bin/env python3
"""
P0: trace oracle — 附录 C.2 invalid committed step 判定规则

invalid committed step 定义为以下任一:
  (a) 混版本组（revision 不一致）进入 committed GroupManifest/StepManifest;
  (b) StepToken 相对权威 step 序列出现重复或缺失;
  (c) checkpoint 内容哈希与 StepToken 绑定不符;
  (d) 错误 reward（与权威 v1 重算对拍不一致）进入 committed gradient。

本模块提供:
  - classify_event_log(events, authoritative) -> 违规列表 (类型 a-d + 事件索引);
  - wilson_upper / clopper_pearson_upper / rule_of_three: 失败率 95% 上界;
  - aggregate(cut_results): 每切点失败率上界 + 聚合 rule-of-three 报告。

事件行格式遵循 configs/paper_event.schema.json（type/ts/exp_id 必填）。

用法（库）+ CLI:
  python3 scripts/trace_oracle.py --events events.jsonl --authoritative-steps 0..99
  python3 scripts/trace_oracle.py --aggregate report.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 违规判定
# --------------------------------------------------------------------------

def classify_event_log(events: list[dict], authoritative_steps: list[int]) -> list[dict]:
    """对事件日志做 (a)-(d) 判定。返回违规事件列表（空 = 无违规）。

    authoritative_steps: 权威已提交 step 序列。
    简单形式: [step, ...]；完整形式: [{"step": 0, "step_token": "t0", "checkpoint_hash": "h0"}, ...]

    角色语义:
      - checkpoint 事件（payload.role 缺省/commit）= 持久化提交声明；同 token 重复声明 = (b) 重复;
      - learner 事件 payload.op == "apply" = 实际应用；同 token 应用 >1 次 = (b) 重复 apply;
      - ack / recovery 事件不是新的提交声明，不触发重复判定。
    """
    viol = []
    auth = {s["step"]: s for s in authoritative_steps} if authoritative_steps and isinstance(authoritative_steps[0], dict) else {}
    auth_by_token = {s["step_token"]: s for s in auth.values()}
    auth_tokens = set(auth_by_token)
    claimed_tokens: dict[str, dict] = {}  # step_token -> checkpoint 事件
    apply_counts: dict[str, int] = {}

    for i, ev in enumerate(events):
        t = ev.get("type")
        if t == "group":
            versions = ev.get("payload", {}).get("versions") or ev.get("versions")
            status = ev.get("payload", {}).get("status") or ev.get("status")
            if status in ("COMMITTED", "PREPARED") and isinstance(versions, (list, set)) and len(set(versions)) > 1:
                viol.append({"type": "a_mixed_revision_committed", "event": i, "ev": ev})
        elif t == "checkpoint":
            role = ev.get("payload", {}).get("role", "commit")
            tok = ev.get("step_token")
            h = ev.get("checkpoint_hash")
            if role == "commit" and tok:
                if tok in claimed_tokens:
                    viol.append({"type": "b_duplicate_step_token", "event": i, "ev": ev})
                claimed_tokens[tok] = ev
                if auth and tok in auth_tokens and h:
                    want = auth_by_token[tok]["checkpoint_hash"]
                    if want and h != want:
                        viol.append({"type": "c_token_hash_mismatch", "event": i, "ev": ev})
        elif t == "learner" and ev.get("payload", {}).get("op") == "apply":
            tok = ev.get("payload", {}).get("step_token")
            if tok:
                apply_counts[tok] = apply_counts.get(tok, 0) + 1
        elif t == "reward":
            committed = ev.get("payload", {}).get("committed", ev.get("committed", False))
            if committed:
                r = ev.get("payload", {}).get("reward", ev.get("reward"))
                r_auth = ev.get("payload", {}).get("authoritative_reward")
                if r_auth is not None and r != r_auth:
                    viol.append({"type": "d_wrong_reward_committed", "event": i, "ev": ev})

    # (b) 重复 apply：同一 token 被应用超过一次
    for tok, c in sorted(apply_counts.items()):
        if c > 1:
            viol.append({"type": "b_duplicate_apply", "step_token": tok, "count": c})

    # (b) 缺失：权威 token 从未出现提交声明（liveness 型）
    if auth:
        for tok in sorted(auth_tokens - set(claimed_tokens)):
            viol.append({"type": "b_missing_step_token", "step_token": tok})

    return viol


def summarize_violations(viols: list[dict]) -> dict:
    from collections import Counter
    c = Counter(v["type"] for v in viols)
    return {"violations": len(viols), "by_type": dict(c)}


# --------------------------------------------------------------------------
# 失败率上界（95%）
# --------------------------------------------------------------------------

def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval 上界（k 失败 / n 次）。"""
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return min(1.0, center + half)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Clopper–Pearson 精确上界。k=0 时退化为 1 - alpha^(1/n)。"""
    if n <= 0:
        return 0.0
    if k == 0:
        return 1 - alpha ** (1.0 / n)
    if k >= n:
        return 1.0
    # 二分求解 Beta 分位数上界
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = min((lo + hi) / 2, 1 - 1e-12)  # 保护 log1p(-pp) 域
        # P(X <= k | n, p=mid) >= alpha/2 时上界 >= mid
        import math as _m
        # 用正则化不完全 Beta 的近似（大 n 时 Wilson 已足够；此处保留精确实现路径）
        from math import lgamma, exp
        def _binom_cdf(kk, nn, pp):
            s = 0.0
            for j in range(kk + 1):
                lg = lgamma(nn + 1) - lgamma(j + 1) - lgamma(nn - j + 1) + j * _m.log(pp) + (nn - j) * _m.log1p(-pp)
                s += exp(lg)
            return s
        if _binom_cdf(k, n, mid) >= alpha / 2:
            lo = mid
        else:
            hi = mid
    return lo


def rule_of_three(n: int) -> float:
    """0 失败时 95% 上界 ≈ 3/n。"""
    return 3.0 / n if n > 0 else 0.0


def aggregate(cut_results: dict[str, dict]) -> dict:
    """cut_results: {cut: {"n": int, "failures": int}} -> 聚合报告。

    每切点给 Clopper–Pearson 上界；全部零失败时按 rule of three 给聚合上界。
    """
    report = {"cutpoints": {}, "total": {"n": 0, "failures": 0}}
    for cut, r in sorted(cut_results.items()):
        n, k = r["n"], r["failures"]
        report["cutpoints"][cut] = {
            "n": n, "failures": k,
            "wilson_upper": round(wilson_upper(k, n), 8),
            "cp_upper": round(clopper_pearson_upper(k, n), 8),
        }
        report["total"]["n"] += n
        report["total"]["failures"] += k
    t = report["total"]
    if t["failures"] == 0 and t["n"] > 0:
        report["total"]["rule_of_three_upper"] = round(rule_of_three(t["n"]), 8)
    else:
        report["total"]["rule_of_three_upper"] = None
    report["status"] = "PASS" if t["failures"] == 0 else "FAIL"
    return report


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, help="events.jsonl 文件")
    ap.add_argument("--authoritative-steps", type=str, default="",
                    help="权威 step 序列，如 0..99 或 step:token:hash 每行一个")
    ap.add_argument("--aggregate", type=Path, help="切点结果 JSON 聚合")
    args = ap.parse_args()

    if args.aggregate:
        cut_results = json.loads(args.aggregate.read_text())
        rep = aggregate(cut_results)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        sys.exit(0 if rep["status"] == "PASS" else 1)

    if not args.events:
        ap.error("需要 --events 或 --aggregate")
    events = [json.loads(l) for l in args.events.open(encoding="utf-8") if l.strip()]
    auth = []
    if args.authoritative_steps:
        for line in args.authoritative_steps.strip().splitlines():
            if ":" in line:
                step, tok, h = line.split(":")
                auth.append({"step": int(step), "step_token": tok, "checkpoint_hash": h})
            else:
                auth.append(int(line))
    viols = classify_event_log(events, auth)
    rep = summarize_violations(viols)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    sys.exit(0 if rep["violations"] == 0 else 1)


if __name__ == "__main__":
    main()
