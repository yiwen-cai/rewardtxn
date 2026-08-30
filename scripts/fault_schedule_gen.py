#!/usr/bin/env python3
"""
P0: fault schedule 生成器 — 附录 B.2 schema 的确定性实现

E7: 对 5 个训练 seed (17/29/42/73/101) 各生成一份 500 步故障序列。
    同 seed 的 Faulted Default / Faulted B5 / Faulted RewardTxn 播放同一份 schedule。
E8: 生成 3 次 ×8 小时 soak 的 Poisson 故障序列（平均每 30 分钟 1 次）。

确定性规则:
  - 所有随机性来自 schedule_seed（与训练 seed 解耦，避免种子复用混淆）；
  - 事件 step 落在 [first_event_step, steps - last_event_step] 内且互不重复；
  - group 索引换算固定为 step = group_index // U，即 groups = [step*U, step*U+U-1]；
  - 输出前做 schema 自校验，写入 prereg/fault_schedules/。

用法:
  python3 scripts/fault_schedule_gen.py e7 --seeds 17,29,42,73,101 --steps 500 --k 8 --u 4
  python3 scripts/fault_schedule_gen.py e8 --runs 3 --hours 8 --rate-per-hour 2.0
  python3 scripts/fault_schedule_gen.py validate   # 校验已有全部 schedule
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "prereg" / "fault_schedules"

CUTS = ("R1", "R2", "R3", "R4", "R5", "Q0", "L3", "C1", "C2")


def _params_for(cut: str, rng: random.Random, u: int, step: int) -> dict:
    """按切点生成参数化事件负载（确定性）。"""
    groups = [step * u + i for i in range(u)]
    if cut == "R1":
        return {"groups": groups, "mechanism": "crm_crash", "crash_frac": 0.5}
    if cut == "R2":
        return {"groups": groups, "retry": "stale", "verifier": "v1", "stale_verifier": "retry-stale"}
    if cut == "R3":
        return {"groups": groups, "verifier": "v2", "digest": "d2"}
    if cut == "R4":
        return {"groups": groups[:1], "same_id": True, "second_digest": "d-x", "expect": "blocked"}
    if cut == "R5":
        return {"groups": groups[:1], "workers": 2, "expect": "single_winner"}
    if cut == "Q0":
        return {"groups": groups, "after": "get_meta", "expect": "reclaim_or_manifest"}
    if cut == "L3":
        return {"groups": groups, "after": "optimizer_return", "before": "checkpoint", "expect": "not_committed"}
    if cut == "C1":
        return {"groups": groups, "after": "checkpoint_durable", "before": "commit_pointer", "expect": "unique_verdict"}
    if cut == "C2":
        return {"groups": groups, "ack_drop": True, "expect": "no_duplicate_apply"}
    raise ValueError(cut)


def gen_e7(seeds, steps: int, k: int, u: int, n_events: int = 9) -> list[dict]:
    schedules = []
    margin = max(10, steps // 20)
    for seed in seeds:
        sched_seed = int(hashlib.sha256(f"e7:{seed}".encode()).hexdigest()[:8], 16)
        rng = random.Random(sched_seed)
        step_pool = list(range(margin, steps - margin))
        rng.shuffle(step_pool)
        chosen = sorted(step_pool[:n_events])
        # Never mutate the module-level cut catalog.  A schedule for seed S must
        # be identical whether generated alone or as part of a seed batch.
        cuts = list(CUTS)
        rng.shuffle(cuts)
        events = []
        for step, cut in zip(chosen, cuts[:n_events]):
            events.append({
                "step": int(step),
                "cut": cut,
                "params": _params_for(cut, rng, u, int(step)),
            })
        events.sort(key=lambda e: e["step"])
        schedules.append({
            "schedule_id": f"e7-s{seed}-faulted",
            "seed": seed,
            "schedule_seed": sched_seed,
            "stack": "slime",
            "model": "Qwen2.5-1.5B-Instruct",
            "steps": steps,
            "k": k,
            "u": u,
            "generator": "scripts/fault_schedule_gen.py",
            "events": events,
        })
    return schedules


def gen_e8(runs: int, hours: float, rate_per_hour: float) -> list[dict]:
    if runs <= 0 or hours <= 0 or rate_per_hour <= 0:
        raise ValueError("runs/hours/rate_per_hour must be positive")
    schedules = []
    cuts_cycle = ["R3", "Q0", "C2", "R2", "L3", "C1", "R5", "R1"]
    for r in range(runs):
        schedule_seed = int(hashlib.sha256(f"e8:{r}".encode()).hexdigest()[:8], 16)
        rng = random.Random(schedule_seed)
        total_sec = int(hours * 3600)
        guard_sec = min(600, total_sec // 10)
        rate_per_sec = rate_per_hour / 3600.0

        # Homogeneous Poisson process: exponential inter-arrival times.  The
        # previous implementation fixed N and sampled uniform offsets, which is
        # only a Poisson process conditional on N and incorrectly removed count
        # variability.  Guard windows avoid faults during startup/teardown.
        offsets = []
        t = float(guard_sec)
        while True:
            t += rng.expovariate(rate_per_sec)
            if t >= total_sec - guard_sec:
                break
            offsets.append(round(t, 3))
        events = []
        for i, off in enumerate(offsets):
            cut = cuts_cycle[i % len(cuts_cycle)]
            step = min(499, int((off / total_sec) * 500))  # soak 以 wall-clock 为准
            events.append({
                "wall_clock_sec": off,
                "wall_clock_min": round(off / 60.0, 3),
                "step": step,
                "cut": cut,
                "params": _params_for(cut, rng, 4, step),
            })
        schedules.append({
            "schedule_id": f"e8-soak-run{r+1}",
            "seed": r + 1,
            "schedule_seed": schedule_seed,
            "stack": "slime",
            "model": "Qwen2.5-1.5B-Instruct",
            "steps": 500,
            "k": 8,
            "u": 4,
            "kind": "soak",
            "hours": hours,
            "rate_per_hour": rate_per_hour,
            "fault_process": "poisson",
            "arrival_model": "exponential_interarrival",
            "startup_teardown_guard_sec": guard_sec,
            "realized_event_count": len(events),
            "generator": "scripts/fault_schedule_gen.py",
            "events": events,
        })
    return schedules


def validate(s: dict) -> list[str]:
    errs = []
    if not s.get("schedule_id"):
        errs.append("missing schedule_id")
    evs = s.get("events", [])
    steps = s.get("steps")
    u = s.get("u", 4)
    seen_event_keys = set()
    previous_wall_sec = -math.inf
    for e in evs:
        if "step" in e and steps is not None and not (0 <= e["step"] < steps):
            errs.append(f"step out of range: {e['step']}")
        gs = e.get("params", {}).get("groups", [])
        for g in gs:
            if g // u != e.get("step"):
                errs.append(f"group {g} inconsistent with step {e.get('step')} (u={u})")
        if e.get("cut") not in CUTS:
            errs.append(f"unknown cut {e.get('cut')}")
        key = (e.get("step"), e.get("cut"), e.get("wall_clock_sec"))
        if key in seen_event_keys:
            errs.append(f"duplicate event: {key}")
        seen_event_keys.add(key)
        if s.get("kind") == "soak":
            wall_sec = e.get("wall_clock_sec")
            if not isinstance(wall_sec, (int, float)) or wall_sec <= previous_wall_sec:
                errs.append("soak wall_clock_sec must be strictly increasing numbers")
            else:
                previous_wall_sec = wall_sec
    if s.get("kind") == "soak":
        if s.get("fault_process") != "poisson" or s.get("arrival_model") != "exponential_interarrival":
            errs.append("soak schedule must use exponential-interarrival Poisson process")
        if not isinstance(s.get("schedule_seed"), int):
            errs.append("soak schedule missing integer schedule_seed")
        if s.get("realized_event_count") != len(evs):
            errs.append("realized_event_count mismatch")
    if not s.get("generator", "").startswith("scripts/"):
        errs.append("missing generator attribution")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p7 = sub.add_parser("e7")
    p7.add_argument("--seeds", default="17,29,42,73,101")
    p7.add_argument("--steps", type=int, default=500)
    p7.add_argument("--k", type=int, default=8)
    p7.add_argument("--u", type=int, default=4)
    p8 = sub.add_parser("e8")
    p8.add_argument("--runs", type=int, default=3)
    p8.add_argument("--hours", type=float, default=8.0)
    p8.add_argument("--rate-per-hour", type=float, default=2.0)
    sub.add_parser("validate")
    args = ap.parse_args()

    if args.cmd == "validate":
        out, bad = {}, []
        for p in sorted(OUT_DIR.glob("*.json")):
            s = json.loads(p.read_text())
            errs = validate(s)
            out[p.name] = {"ok": not errs, "errors": errs, "events": len(s["events"])}
            if errs:
                bad.append(p.name)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        sys.exit(1 if bad else 0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.cmd == "e7":
        scheds = gen_e7([int(s) for s in args.seeds.split(",")], args.steps, args.k, args.u)
    else:
        scheds = gen_e8(args.runs, args.hours, args.rate_per_hour)

    for s in scheds:
        errs = validate(s)
        if errs:
            print(f"VALIDATION FAILED {s['schedule_id']}: {errs}", file=sys.stderr)
            sys.exit(2)
        p = OUT_DIR / f"{s['schedule_id']}.json"
        p.write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {p.relative_to(BASE)} ({len(s['events'])} events)")


if __name__ == "__main__":
    main()
