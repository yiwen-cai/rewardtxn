#!/usr/bin/env python3
"""E2 deterministic trace runner with auditable, isolated trial artifacts.

Formal runs use 2,500 trials per cut.  Smoke runs are explicitly separated and
can never overwrite the formal artifact path::

    python3 scripts/paper_trace_runner.py
    python3 scripts/paper_trace_runner.py --mode smoke --per-cut 3
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import phase2_seal_rm as m
import trace_oracle

CUT_POINTS = ["R1", "R2", "R3", "R4", "R5", "Q0", "Q1", "L0", "L2", "L3", "C1", "C2"]
REAL_CODE_CUTS = {"R1", "R2", "R3", "R4", "R5"}
MODEL_CUTS = set(CUT_POINTS) - REAL_CODE_CUTS
FORMAL_PER_CUT = 2500
EXP_ID = "paper-e2-trace"
GOOD = "Let me solve.\n</think>\nThe answer is 42.\n###Response\n\\boxed{42}"
DIVERGENT = "Let me solve.\n</think>\nThe answer is 42.0.\n###Response\n\\boxed{42.0}"

SCHEDULE_LEVELS = {
    "kill_time": [0.1, 0.35, 0.65, 0.9],
    "ack_loss": [False, True],
    "attempt_order": ["forward", "reverse", "interleaved"],
    "revision": ["v1", "v2", "v3"],
    "group_size": [4, 8, 16],
    "checkpoint_delay": [0, 1, 4],
}
RELEVANT_DIMENSIONS = {
    "R1": ["kill_time", "group_size"],
    "R2": ["revision", "group_size"],
    "R3": ["revision", "group_size"],
    "R4": ["attempt_order", "revision"],
    "R5": ["attempt_order", "revision"],
    "Q0": ["kill_time", "attempt_order", "revision", "group_size"],
    "Q1": ["kill_time", "attempt_order", "group_size"],
    "L0": ["kill_time", "checkpoint_delay"],
    "L2": ["kill_time", "attempt_order", "group_size", "checkpoint_delay"],
    "L3": ["kill_time", "checkpoint_delay"],
    "C1": ["kill_time", "revision", "checkpoint_delay"],
    "C2": ["kill_time", "ack_loss", "attempt_order", "revision", "checkpoint_delay"],
}


class TrialFailure(RuntimeError):
    def __init__(self, record):
        super().__init__("invalid trial %s/%s" % (record["cut"], record["trial"]))
        self.record = record


class _S:
    def __init__(self, group_index, index, rollout_id, response, label):
        self.group_index = group_index
        self.index = index
        self.rollout_id = rollout_id
        self.response = response
        self.label = label


def _utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def _git_output(*args):
    result = subprocess.run(["git"] + list(args), cwd=str(BASE), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.returncode, result.stdout.strip()


def _git_identity():
    return_code, commit = _git_output("rev-parse", "HEAD")
    if return_code:
        return "UNKNOWN", False
    # Formal provenance is bound to tracked source plus explicit source hashes.
    # Run artifacts themselves are normally untracked, so counting untracked
    # output would make a completed formal run impossible to rerun.
    status_code, status = _git_output("status", "--porcelain", "--untracked-files=no")
    return commit, status_code == 0 and not status


def _dimension_value(seed, cut, trial, dimension):
    levels = SCHEDULE_LEVELS[dimension]
    material = "%s:%s:%s" % (seed, cut, dimension)
    local = random.Random(int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16))
    permutation = list(levels)
    local.shuffle(permutation)
    return permutation[trial % len(permutation)]


def build_schedule(seed, per_cut):
    trials = []
    for cut in CUT_POINTS:
        for trial in range(per_cut):
            dimensions = {name: _dimension_value(seed, cut, trial, name)
                          for name in SCHEDULE_LEVELS}
            core = {"cut": cut, "trial": trial, "dimensions": dimensions,
                    "relevant_dimensions": RELEVANT_DIMENSIONS[cut]}
            trials.append({**core, "schedule_id": _sha_bytes(_canonical(core))[:20]})
    return trials


def schedule_coverage(trials):
    coverage = {}
    for row in trials:
        cut = row["cut"]
        coverage.setdefault(cut, {})
        for dimension in row["relevant_dimensions"]:
            key = json.dumps(row["dimensions"][dimension], sort_keys=True)
            coverage[cut].setdefault(dimension, {})[key] = coverage[cut].setdefault(dimension, {}).get(key, 0) + 1
    return coverage


def _schedule_values(row):
    return dict(row["dimensions"])


def _prepare_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        superseded = path.with_name("%s.superseded.%s.%s" % (path.name, stamp, os.getpid()))
        path.rename(superseded)
    path.mkdir(parents=True)
    (path / "manifests").mkdir()


def _reset_trial(run_dir, fault="none", group_size=8, autofix=False, revision="v1"):
    os.environ["RTX_RUN_DIR"] = str(run_dir)
    os.environ["RTX_CAS_INDEX_DIR"] = str(run_dir / "cas")
    (run_dir / "cas").mkdir(parents=True, exist_ok=True)
    m.RUN_DIR = str(run_dir)
    m.SEAL, m.GROUP_RM, m.AUTO_FIX = True, True, autofix
    m.K, m.FAULT = group_size, fault
    m.V2_MODE = "strict" if revision in ("v1", "v3") else "relaxed"
    m._WINDOWS = {}
    m._seal_state.clear()
    m._seen_logical.clear()
    m._pending_samples.clear()
    m._reset_cas_connection()


def _batch(group_index, group_size, mixed=False):
    half = group_size // 2
    return [_S(group_index, index, index,
               DIVERGENT if mixed and index >= half else GOOD, "42")
            for index in range(group_size)]


def _read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _r_events(cut, trial_dir, group_index, schedule, worker_results=None):
    events = []
    for record in _read_jsonl(trial_dir / "rewards.jsonl"):
        if record.get("group_index") != group_index:
            continue
        committed = cut in ("R2", "R3", "R4", "R5") and record.get("reward") is not None
        events.append({"type": "reward", "ts": float(record.get("ts", 0.0)), "step": schedule["trial"],
                       "logical_id": record.get("logical_id"),
                       "payload": {**record, "op": "cas_write", "committed": committed}})
    for record in _read_jsonl(trial_dir / "cas_rejects.jsonl"):
        if record.get("group_index") != group_index:
            continue
        events.append({"type": "reward", "ts": float(record.get("ts", 0.0)), "step": schedule["trial"],
                       "logical_id": record.get("logical_id"),
                       "payload": {**record, "op": "cas_reject", "committed": False}})
    for record in _read_jsonl(trial_dir / "seals.jsonl"):
        if record.get("group_index") != group_index:
            continue
        events.append({"type": "group", "ts": float(record.get("ts", 0.0)), "step": schedule["trial"],
                       "group_ids": [group_index], "payload": record})
    for result in worker_results or []:
        events.append({"type": "learner", "ts": time.time(), "step": schedule["trial"],
                       "payload": {"op": "recovery_worker_exit", **result}})
    events.sort(key=lambda event: event["ts"])
    events.append({"type": "fault", "ts": (events[-1]["ts"] + 1e-6 if events else time.time()),
                   "step": schedule["trial"], "payload": {"cut": cut,
                   "kill_time": schedule["dimensions"]["kill_time"]}})
    return events


def _reward_authority(events):
    return [{"logical_id": event["logical_id"],
             "authoritative_reward": float(m._v1_reward(event["payload"].get("response", ""),
                                                         event["payload"].get("label", "")))}
            for event in events if event.get("type") == "reward"
            and event.get("payload", {}).get("op") == "cas_write"
            and event.get("payload", {}).get("committed")]


def fixture_r1(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    group_size = values["group_size"]
    _reset_trial(trial_dir, group_size=group_size, revision=values["revision"])
    samples = _batch(group_index, group_size)
    completed = min(group_size - 1, int(values["kill_time"] * group_size))
    for sample in samples[:completed]:
        asyncio.run(m._rm_one(sample))
    m._WINDOWS = {"crm_crash": [group_index, group_index]}
    try:
        asyncio.run(m._rm_one(samples[completed]))
    except RuntimeError:
        pass
    events = _r_events("R1", trial_dir, group_index, schedule)
    return events, {"steps": [], "rewards": []}


def _fixture_mixed(cut, schedule, trial_dir, group_index):
    values = _schedule_values(schedule)
    group_size = values["group_size"]
    _reset_trial(trial_dir, group_size=group_size, autofix=True, revision=values["revision"])
    m._WINDOWS = {("dup" if cut == "R2" else "skew"): [group_index, group_index]}
    asyncio.run(m._rm_batch_group(_batch(group_index, group_size, mixed=True)))
    events = _r_events(cut, trial_dir, group_index, schedule)
    return events, {"steps": [], "rewards": _reward_authority(events)}


def fixture_r2(schedule, trial_dir, group_index, worker_timeout):
    return _fixture_mixed("R2", schedule, trial_dir, group_index)


def fixture_r3(schedule, trial_dir, group_index, worker_timeout):
    return _fixture_mixed("R3", schedule, trial_dir, group_index)


def fixture_r4(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    _reset_trial(trial_dir, revision=values["revision"])
    logical_id = "r4:%s:0" % group_index
    first = {"group_index": group_index, "index": 0, "rollout_id": 0,
             "reward": 1.0, "verifier": values["revision"], "digest": "d-authority",
             "logical_id": logical_id, "response": GOOD, "label": "42"}
    # Both attempts carry the same externally correct reward but different
    # payload digests.  This lets the schedule vary real arrival order without
    # changing the oracle answer; the invariant under test is first-writer CAS,
    # not a scheduler-dependent definition of reward correctness.
    second = {**first, "digest": "d-conflict", "response": DIVERGENT}
    attempts = [first, second]
    if values["attempt_order"] == "reverse":
        attempts = [second, first]
    elif values["attempt_order"] == "interleaved":
        attempts[1]["attempt"] = "interleaved"
    m._cas_write(attempts[0])
    m._cas_write(attempts[1])
    events = _r_events("R4", trial_dir, group_index, schedule)
    return events, {"steps": [], "rewards": [{"logical_id": logical_id, "authoritative_reward": 1.0}]}


def _r5_worker(run_dir, logical_id, group_index, revision):
    os.environ["RTX_RUN_DIR"] = run_dir
    os.environ["RTX_CAS_INDEX_DIR"] = str(Path(run_dir) / "cas")
    import phase2_seal_rm as worker_module
    worker_module.RUN_DIR = run_dir
    worker_module._seen_logical.clear()
    worker_module._reset_cas_connection()
    worker_module._cas_write({"group_index": group_index, "index": 0, "rollout_id": 0,
                              "reward": 1.0, "verifier": revision, "logical_id": logical_id,
                              "response": GOOD, "label": "42"})


def fixture_r5(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    _reset_trial(trial_dir, revision=values["revision"])
    logical_id = "r5:%s:0" % group_index
    processes = [mp.Process(target=_r5_worker,
                            args=(str(trial_dir), logical_id, group_index, values["revision"]))
                 for _ in range(2)]
    start_order = list(range(2))
    if values["attempt_order"] == "reverse":
        start_order.reverse()
    for index in start_order:
        processes[index].start()
    deadline = time.monotonic() + worker_timeout
    worker_results = []
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(remaining)
        timed_out = process.is_alive()
        if timed_out:
            process.terminate()
            process.join(1.0)
        worker_results.append({"pid": process.pid, "exit_code": process.exitcode,
                               "timed_out": timed_out})
    events = _r_events("R5", trial_dir, group_index, schedule, worker_results)
    return events, {"steps": [], "rewards": [{"logical_id": logical_id, "authoritative_reward": 1.0}]}


def _ev(typ, step, ts, **extra):
    return {"type": typ, "ts": float(ts), "step": step, **extra}


def _ordered(values, order):
    values = list(values)
    if order == "reverse":
        return list(reversed(values))
    if order == "interleaved":
        return values[::2] + values[1::2]
    return values


def fixture_q0(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    ordered = _ordered(range(group_index, group_index + values["group_size"]), values["attempt_order"])
    consumed_count = max(1, int(len(ordered) * values["kill_time"]))
    consumed = ordered[:consumed_count]
    committed = set(consumed[:consumed_count // 2])
    events = [_ev("queue", schedule["trial"], i, group_ids=[gid],
                  payload={"op": "mark_consumed", "index": gid}) for i, gid in enumerate(consumed)]
    if committed:
        events.append(_ev("group", schedule["trial"], len(events), group_ids=sorted(committed),
                          payload={"status": "COMMITTED", "versions": [values["revision"]]}))
    events.append(_ev("fault", schedule["trial"], len(events),
                      payload={"cut": "Q0", "kill_time": values["kill_time"]}))
    need = [gid for gid in consumed if gid not in committed]
    if need:
        events.append(_ev("recovery", schedule["trial"], len(events), group_ids=need,
                          recovery_decision="reissue", payload={"attempt_order": values["attempt_order"]}))
    return events, {"steps": [], "rewards": []}


def fixture_q1(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    ordered = _ordered(range(group_index, group_index + values["group_size"]), values["attempt_order"])
    fetched = ordered[:max(1, int(len(ordered) * values["kill_time"]))]
    events = [_ev("queue", schedule["trial"], i, group_ids=[gid],
                  payload={"op": "get_data", "index": gid}) for i, gid in enumerate(fetched)]
    events.append(_ev("fault", schedule["trial"], len(events), payload={"cut": "Q1"}))
    events.append(_ev("recovery", schedule["trial"], len(events), group_ids=fetched,
                      recovery_decision="reissue", payload={"attempt_order": values["attempt_order"]}))
    return events, {"steps": [], "rewards": []}


def fixture_l0(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    in_memory = max(1, int(8 * values["kill_time"]))
    events = [_ev("learner", schedule["trial"], 0, payload={"op": "pre_manifest",
              "in_memory_steps": in_memory, "checkpoint_delay": values["checkpoint_delay"]}),
              _ev("fault", schedule["trial"], values["kill_time"], payload={"cut": "L0"}),
              _ev("recovery", schedule["trial"], 1 + values["checkpoint_delay"],
                  recovery_decision="cold_start")]
    return events, {"steps": [], "rewards": []}


def fixture_l2(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    ranks = values["group_size"]
    done = min(ranks - 1, int(ranks * values["kill_time"]))
    order = _ordered(range(ranks), values["attempt_order"])
    token = "pre-%s-%s" % (schedule["cut"], schedule["trial"])
    checkpoint_hash = "h-%s-d%s" % (token, values["checkpoint_delay"])
    events = [_ev("learner", schedule["trial"], 0,
                  payload={"op": "optimizer_start", "ranks": ranks})]
    for seq, rank in enumerate(order[:done], 1):
        events.append(_ev("learner", schedule["trial"], seq,
                          payload={"op": "rank_apply", "rank": rank, "durable": False}))
    events.append(_ev("fault", schedule["trial"], done + values["kill_time"], payload={"cut": "L2"}))
    events.append(_ev("recovery", schedule["trial"], done + 1, recovery_decision="rollback"))
    events.append(_ev("checkpoint", schedule["trial"], done + 1 + values["checkpoint_delay"],
                      checkpoint_hash=checkpoint_hash, step_token=token,
                      payload={"state": "pre_step", "role": "commit"}))
    authority = {"steps": [{"step": schedule["trial"], "step_token": token,
                             "checkpoint_hash": checkpoint_hash}], "rewards": []}
    return events, authority


def fixture_l3(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    steps = max(1, int(8 * values["kill_time"]))
    events = [_ev("learner", schedule["trial"], 0,
                  payload={"op": "optimizer_done", "in_memory": True,
                           "steps_in_memory": steps, "checkpoint_delay": values["checkpoint_delay"]}),
              _ev("fault", schedule["trial"], values["kill_time"], payload={"cut": "L3"}),
              _ev("recovery", schedule["trial"], 1 + values["checkpoint_delay"],
                  recovery_decision="rollback")]
    return events, {"steps": [], "rewards": []}


def _checkpoint_identity(schedule):
    values = _schedule_values(schedule)
    token = "%s-t%s-%s" % (schedule["cut"].lower(), schedule["trial"], values["revision"])
    checkpoint_hash = "sha256:%s:d%s" % (token, values["checkpoint_delay"])
    return token, checkpoint_hash


def fixture_c1(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    token, checkpoint_hash = _checkpoint_identity(schedule)
    delay = values["checkpoint_delay"]
    events = [_ev("checkpoint", schedule["trial"], delay, checkpoint_hash=checkpoint_hash,
                  step_token=token, payload={"durable": True, "role": "commit",
                  "revision": values["revision"]}),
              _ev("fault", schedule["trial"], delay + values["kill_time"],
                  payload={"cut": "C1", "commit_pointer": False}),
              _ev("recovery", schedule["trial"], delay + 1, recovery_decision="commit_ack",
                  step_token=token, checkpoint_hash=checkpoint_hash)]
    return events, {"steps": [{"step": schedule["trial"], "step_token": token,
                               "checkpoint_hash": checkpoint_hash}], "rewards": []}


def fixture_c2(schedule, trial_dir, group_index, worker_timeout):
    values = _schedule_values(schedule)
    token, checkpoint_hash = _checkpoint_identity(schedule)
    delay = values["checkpoint_delay"]
    attempts = _ordered([0, 1, 2], values["attempt_order"])
    events = [_ev("checkpoint", schedule["trial"], delay, checkpoint_hash=checkpoint_hash,
                  step_token=token, payload={"durable": True, "role": "commit",
                  "revision": values["revision"]}),
              _ev("ack", schedule["trial"], delay + values["kill_time"], step_token=token,
                  payload={"dropped": values["ack_loss"]}),
              _ev("fault", schedule["trial"], delay + values["kill_time"] + 0.01,
                  payload={"cut": "C2"})]
    for seq, attempt in enumerate(attempts):
        events.append(_ev("learner", schedule["trial"], delay + 0.1 + seq * 0.01,
                          payload={"op": "attempt_observed", "attempt": attempt}))
    events.extend([
        _ev("recovery", schedule["trial"], delay + 1, recovery_decision="commit_ack",
            step_token=token, checkpoint_hash=checkpoint_hash),
        _ev("learner", schedule["trial"], delay + 2,
            payload={"op": "apply", "step_token": token, "step": schedule["trial"]}),
    ])
    return events, {"steps": [{"step": schedule["trial"], "step_token": token,
                               "checkpoint_hash": checkpoint_hash}], "rewards": []}


FIXTURES = {
    "R1": fixture_r1, "R2": fixture_r2, "R3": fixture_r3, "R4": fixture_r4,
    "R5": fixture_r5, "Q0": fixture_q0, "Q1": fixture_q1, "L0": fixture_l0,
    "L2": fixture_l2, "L3": fixture_l3, "C1": fixture_c1, "C2": fixture_c2,
}


def _annotate(events, schedule, verdict=None):
    result = []
    for sequence, source in enumerate(events):
        event = dict(source)
        event.update(exp_id=EXP_ID, cut=schedule["cut"], trial=schedule["trial"],
                     schedule_id=schedule["schedule_id"], sequence=sequence)
        if verdict is not None:
            event["oracle_verdict"] = verdict
        result.append(event)
    return result


def _artifact_manifest(run_dir):
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "artifact_manifest.json" or path.name.endswith(".tmp"):
            continue
        files.append({"path": path.relative_to(run_dir).as_posix(), "bytes": path.stat().st_size,
                      "sha256": _sha_file(path)})
    return {"schema_version": "1.0", "files": files}


def _source_hashes():
    names = ["scripts/paper_trace_runner.py", "scripts/trace_oracle.py",
             "scripts/paper_gates.py", "scripts/phase2_seal_rm.py",
             "configs/paper_event.schema.json"]
    return {name: _sha_file(BASE / name) for name in names if (BASE / name).exists()}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--per-cut", type=int)
    parser.add_argument("--seed", type=int, default=0x5200E2)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--tmp", type=Path, help="trial scratch root")
    parser.add_argument("--worker-timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    if args.per_cut is None:
        args.per_cut = FORMAL_PER_CUT if args.mode == "formal" else 3
    if args.per_cut <= 0:
        parser.error("--per-cut must be positive")
    if args.mode == "formal" and args.per_cut != FORMAL_PER_CUT:
        parser.error("formal mode requires exactly 2500 trials per cut; use --mode smoke for smaller runs")
    if args.worker_timeout <= 0:
        parser.error("--worker-timeout must be positive")
    if args.out_dir is None:
        args.out_dir = BASE / "runs" / ("paper-e2-trace-formal" if args.mode == "formal"
                                        else "paper-e2-trace-smoke")
    elif not args.out_dir.is_absolute():
        args.out_dir = BASE / args.out_dir
    return args


def run(args):
    started = time.monotonic()
    created_at = _utc_now()
    _prepare_output(args.out_dir)
    _write_json(args.out_dir / "verdict.json", {"status": "RUNNING", "started_at": created_at})
    _write_json(args.out_dir / "exit_status.json",
                {"completed": False, "return_code": None, "finished_at": None})
    (args.out_dir / "stdout.log").write_text("", encoding="utf-8")
    (args.out_dir / "stderr.log").write_text("", encoding="utf-8")
    (args.out_dir / "events.jsonl").write_text("", encoding="utf-8")
    (args.out_dir / "verdicts.jsonl").write_text("", encoding="utf-8")
    (args.out_dir / "resource.jsonl").write_text("", encoding="utf-8")

    commit_sha, working_tree_clean = _git_identity()
    config = {
        "schema_version": "2.0", "mode": args.mode, "seed": args.seed,
        "per_cut": args.per_cut, "cuts": CUT_POINTS, "schedule_levels": SCHEDULE_LEVELS,
        "relevant_dimensions": RELEVANT_DIMENSIONS, "worker_timeout_seconds": args.worker_timeout,
        "out_dir": str(args.out_dir), "scratch_root_request": str(args.tmp) if args.tmp else None,
        "python_version": sys.version, "fail_fast_on_first_invalid_commit": True,
        "fixture_environment": {
            "RTX_SEAL": "1", "RTX_GROUP_RM": "1", "RTX_SEAL_AUTO_FIX": "cut-dependent",
            "RTX_FAULT": "schedule-dependent", "RTX_GROUP_SIZE": "schedule.group_size",
            "RTX_RUN_DIR": "fresh-directory-per-trial",
            "RTX_CAS_INDEX_DIR": "fresh-directory-per-trial/cas",
        },
    }
    _write_json(args.out_dir / "config.json", config)
    config_sha = _sha_file(args.out_dir / "config.json")
    schedule_rows = build_schedule(args.seed, args.per_cut)
    schedule_doc = {"schema_version": "2.0", "seed": args.seed, "trials": schedule_rows,
                    "coverage": schedule_coverage(schedule_rows)}
    _write_json(args.out_dir / "schedule.json", schedule_doc)
    schedule_sha = _sha_file(args.out_dir / "schedule.json")
    meta = {
        "exp_id": EXP_ID, "phase": "P1/E2", "stack": "phase2_seal_rm+protocol-model",
        "baseline": "RewardTxn", "commit_sha": commit_sha, "config_sha256": config_sha,
        "schedule_sha256": schedule_sha, "seed": args.seed,
        "group_size_K": 8, "group_size_schedule": SCHEDULE_LEVELS["group_size"],
        "batch_groups_U": 1,
        "fault_injection": {"cuts": CUT_POINTS, "schedule": "schedule.json"},
        "created_at": created_at, "working_tree_clean": working_tree_clean,
        "source_sha256": _source_hashes(),
    }
    _write_json(args.out_dir / "meta.json", meta)
    _write_json(args.out_dir / "manifests" / "fixture_manifest.json", {
        "manifest_type": "typed-fixture-evidence", "schema_version": "1.0",
        "real_code_cuts": sorted(REAL_CODE_CUTS), "model_fixture_cuts": sorted(MODEL_CUTS),
        "model_fixture_substitution": True,
        "scope_note": "Model fixtures are deterministic protocol evidence, not model/checkpoint artifacts.",
    })

    stdout_lines = []
    stderr_lines = []
    cut_results = {cut: {"n": 0, "failures": 0} for cut in CUT_POINTS}
    first_failure = None
    return_code = 1
    if args.tmp:
        args.tmp.mkdir(parents=True, exist_ok=True)
        scratch_root = Path(tempfile.mkdtemp(prefix="paper-trials-", dir=str(args.tmp)))
    else:
        scratch_root = Path(tempfile.mkdtemp(prefix="rtx-paper-trials-"))

    try:
        if args.mode == "formal" and not working_tree_clean:
            raise RuntimeError("formal run requires a clean git worktree")
        row_by_key = {(row["cut"], row["trial"]): row for row in schedule_rows}
        with (args.out_dir / "events.jsonl").open("a", encoding="utf-8") as events_handle, \
                (args.out_dir / "verdicts.jsonl").open("a", encoding="utf-8") as verdict_handle:
            for cut_index, cut in enumerate(CUT_POINTS):
                for trial in range(args.per_cut):
                    schedule = row_by_key[(cut, trial)]
                    trial_dir = scratch_root / cut / ("%06d" % trial)
                    trial_dir.mkdir(parents=True)
                    group_index = cut_index * 10_000_000 + trial
                    events, authoritative = FIXTURES[cut](schedule, trial_dir, group_index,
                                                          args.worker_timeout)
                    annotated = _annotate(events, schedule)
                    ok, detail = trace_oracle.verify_trial(cut, annotated, authoritative,
                                                           schedule["dimensions"])
                    annotated = _annotate(events, schedule, "PASS" if ok else "FAIL")
                    event_sha = _sha_bytes(_canonical(annotated))
                    for event in annotated:
                        _append_jsonl(events_handle, event)
                    record = {"cut": cut, "trial": trial, "schedule_id": schedule["schedule_id"],
                              "schedule": schedule["dimensions"], "event_count": len(annotated),
                              "events_sha256": event_sha, "authoritative": authoritative,
                              "status": "PASS" if ok else "FAIL", "detail": detail}
                    _append_jsonl(verdict_handle, record)
                    cut_results[cut]["n"] += 1
                    if not ok:
                        cut_results[cut]["failures"] += 1
                        first_failure = record
                        raise TrialFailure(record)
                message = "  %s: %s/%s PASS" % (cut, cut_results[cut]["n"], args.per_cut)
                stdout_lines.append(message)
                print(message, flush=True)
        return_code = 0
    except BaseException as exc:
        if isinstance(exc, TrialFailure):
            stderr_lines.append(str(exc))
        else:
            stderr_lines.append("%s: %s" % (type(exc).__name__, exc))
            stderr_lines.append(traceback.format_exc())

    report = trace_oracle.aggregate(cut_results)
    if return_code != 0:
        report["status"] = "FAIL"
    report.update({"schema_version": "2.0", "mode": args.mode, "schedule_seed": args.seed,
                   "commit_sha": commit_sha, "config_sha256": config_sha,
                   "schedule_sha256": schedule_sha, "first_failure": first_failure,
                   "real_code_cuts": sorted(REAL_CODE_CUTS), "model_cuts": sorted(MODEL_CUTS),
                   "generated_at": _utc_now()})
    _write_json(args.out_dir / "TRACE_REPORT_PAPER.json", report)
    elapsed = time.monotonic() - started
    _write_json(args.out_dir / "metrics.json", {
        "status": report["status"], "cutpoints": report["cutpoints"], "total": report["total"],
        "elapsed_seconds": round(elapsed, 6), "fail_fast": first_failure is not None,
    })
    with (args.out_dir / "resource.jsonl").open("a", encoding="utf-8") as resource_handle:
        _append_jsonl(resource_handle, {
            "ts": time.time(), "cpu_only": True, "elapsed_seconds": round(elapsed, 6),
            "scratch_root": str(scratch_root),
        })
    verdict = {"status": "PASS" if return_code == 0 and report["status"] == "PASS" else "FAIL",
               "fail_fast": first_failure is not None, "first_failure": first_failure,
               "completed_trials": report["total"]["n"]}
    _write_json(args.out_dir / "verdict.json", verdict)
    stdout_lines.append("=== E2 trace: %s (%s trials) ===" % (verdict["status"], report["total"]["n"]))
    stdout_lines.append("report: %s" % (args.out_dir / "TRACE_REPORT_PAPER.json"))
    (args.out_dir / "stdout.log").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")
    (args.out_dir / "stderr.log").write_text("\n".join(stderr_lines) + ("\n" if stderr_lines else ""),
                                                encoding="utf-8")
    _write_json(args.out_dir / "exit_status.json",
                {"completed": True, "return_code": return_code, "finished_at": _utc_now()})
    _write_json(args.out_dir / "artifact_manifest.json", _artifact_manifest(args.out_dir))
    for line in stdout_lines[-2:]:
        print(line, flush=True)
    if stderr_lines:
        print(stderr_lines[0], file=sys.stderr, flush=True)
    return return_code


def main(argv=None):
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
