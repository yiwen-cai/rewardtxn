#!/usr/bin/env python3
"""Decompose per-run wall time of formal A/R runs from existing evidence (CPU only).

Phases (seconds):
  launch_to_trainer : launcher start -> trainer process start (rollout servers up)
  trainer_to_first  : trainer start -> first batch_taken (init + first rollout batch)
  loop              : first batch_taken -> last recover_handler_dump_returned
  loop_tail         : last dump_returned -> launcher exit
  outside_launcher  : cost.wall_seconds - launcher span (container setup, acceptance, cleanup)
Per step (medians over steps): wait_batch (prev dump_returned -> batch_taken),
  train (batch_taken -> optimizer_end), save (dump_start -> dump_returned), and for R the
  RewardTxn transaction events update_prepared/committed relative to optimizer/save.
"""
import argparse
import datetime as dt
import glob
import json
import os
import re
import statistics
from zoneinfo import ZoneInfo

TS = re.compile(r"(\d{8}-\d{2}:\d{2}:\d{2}\.\d{3})")


def launcher_times(path):
    stamps, trainer = [], None
    for line in open(path, errors="replace"):
        m = TS.search(line)
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y%m%d-%H:%M:%S.%f").replace(tzinfo=ZoneInfo("UTC")).timestamp()
        stamps.append(t)
        if trainer is None and "Found" in line and "rollout servers" in line:
            trainer = t
    return stamps[0], trainer, stamps[-1]


def load_jsonl(pattern):
    out = []
    for f in glob.glob(pattern):
        out += [json.loads(l) for l in open(f) if l.strip()]
    return out


def med(v):
    return round(statistics.median(v), 2) if v else None


def profile(run):
    pilot = load_jsonl(os.path.join(run, "observer-pilot", "*.jsonl"))
    trainer_ev = sorted((e for e in pilot if e.get("source_role") == "trainer"), key=lambda e: e["monotonic_ns"])
    by = {}
    for e in trainer_ev:
        by.setdefault(e["event"], []).append(e)
    # wall<->monotonic offset from observer events (same host clock)
    off = trainer_ev[0]["wall_time_ns"] - trainer_ev[0]["monotonic_ns"]
    w = lambda ns: (int(ns) + off) / 1e9

    l0, ltr, l1 = launcher_times(os.path.join(run, "launcher.log"))
    wall = json.load(open(os.path.join(run, "cost.json")))["wall_seconds"]
    first = w(by["batch_taken"][0]["monotonic_ns"])
    last = w(by["recover_handler_dump_returned"][-1]["monotonic_ns"])
    res = {
        "run": os.path.basename(run),
        "wall": round(wall, 1),
        "launch_to_trainer": round(ltr - l0, 1),
        "trainer_to_first": round(first - ltr, 1),
        "loop": round(last - first, 1),
        "loop_tail": round(l1 - last, 1),
        "outside_launcher": round(wall - (l1 - l0), 1),
    }
    t = lambda k, i: w(by[k][i]["monotonic_ns"])
    n = len(by["batch_taken"])
    res["step_wait_batch"] = med([t("batch_taken", i) - t("recover_handler_dump_returned", i - 1) for i in range(1, n)])
    res["step_train"] = med([t("optimizer_end", i) - t("batch_taken", i) for i in range(n)])
    res["step_opt_to_save"] = med([t("recover_handler_dump_start", i) - t("optimizer_end", i) for i in range(n)])
    res["step_save"] = med([t("recover_handler_dump_returned", i) - t("recover_handler_dump_start", i) for i in range(n)])
    rtx = os.path.join(run, "rewardtxn", "events.jsonl")
    if os.path.exists(rtx):
        rev = {}
        for e in load_jsonl(rtx):
            rev.setdefault(e["event"], []).append(w(e["monotonic_ns"]))
        seq = ["update_prepared", "optimizer_applied", "scheduler_applied", "async_scheduled", "async_finalized", "committed"]
        for a, b in zip(seq, seq[1:]):
            res[f"r_{a}->{b}"] = med([y - x for x, y in zip(rev[a], rev[b])])
        res["r_committed->next_prepared"] = med([y - x for x, y in zip(rev["committed"], rev["update_prepared"][1:])])
        res["r_batch_taken->prepared"] = med([p - t("batch_taken", i) for i, p in enumerate(rev["update_prepared"][:n])])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="docs/experiments/rewardtxn-ft-20260916/minimal_evidence")
    ap.add_argument("--pattern", default="formal-nofault-*-[ar]")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = [profile(r) for r in sorted(glob.glob(os.path.join(args.evidence, args.pattern))) if os.path.isdir(r)]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
