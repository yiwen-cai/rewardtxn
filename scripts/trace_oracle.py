#!/usr/bin/env python3
"""Trace oracle for the paper fault-injection runs.

The oracle consumes *external* authoritative state. Event fields such as
``authoritative_reward`` are deliberately ignored: allowing the system under
test to state its own expected answer makes the oracle circular.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


def _normalise_authoritative(authoritative):
    if authoritative is None:
        return [], []
    if isinstance(authoritative, dict):
        return list(authoritative.get("steps", [])), list(authoritative.get("rewards", []))
    return list(authoritative), []


def _violation(kind, event=None, **extra):
    out = {"type": kind}
    if event is not None:
        out["event"] = event
    out.update(extra)
    return out


def classify_event_log(events: list[dict], authoritative) -> list[dict]:
    """Classify invalid commits in one trial.

    ``authoritative`` accepts the historical list form or the full form::

        {"steps": [{"step": 1, "step_token": "t1",
                     "checkpoint_hash": "h1"}],
         "rewards": [{"logical_id": "g:0:0",
                      "authoritative_reward": 1.0}]}

    Full authority enables checks for unknown, missing and duplicate tokens,
    steps and hashes, plus reward comparison against an external source.
    """
    viol = []
    raw_steps, raw_rewards = _normalise_authoritative(authoritative)

    auth_by_step = {}
    auth_by_token = {}
    for pos, raw in enumerate(raw_steps):
        item = {"step": raw} if isinstance(raw, int) else raw
        if not isinstance(item, dict) or not isinstance(item.get("step"), int):
            viol.append(_violation("authority_invalid_step", authority_index=pos))
            continue
        step = item["step"]
        if step in auth_by_step:
            viol.append(_violation("authority_duplicate_step", step=step))
        auth_by_step[step] = item
        tok = item.get("step_token")
        if tok is not None:
            if not isinstance(tok, str) or not tok:
                viol.append(_violation("authority_invalid_step_token", step=step))
            elif tok in auth_by_token:
                viol.append(_violation("authority_duplicate_step_token", step_token=tok))
            else:
                auth_by_token[tok] = item

    auth_rewards = {}
    for pos, item in enumerate(raw_rewards):
        if not isinstance(item, dict) or not item.get("logical_id"):
            viol.append(_violation("authority_invalid_reward", authority_index=pos))
            continue
        lid = str(item["logical_id"])
        if lid in auth_rewards:
            viol.append(_violation("authority_duplicate_reward", logical_id=lid))
        auth_rewards[lid] = item.get("authoritative_reward")

    claimed_tokens = {}
    claimed_steps = {}
    apply_counts = Counter()
    apply_first_event = {}
    committed_reward_counts = Counter()

    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            viol.append(_violation("malformed_event", event=i))
            continue
        typ = ev.get("type")
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}

        if typ == "group":
            status = payload.get("status", ev.get("status"))
            versions = payload.get("versions", ev.get("versions"))
            if status in ("COMMITTED", "SEALED"):
                if not isinstance(versions, (list, tuple, set)) or not versions:
                    viol.append(_violation("a_missing_revision_committed", i, ev=ev))
                elif any(v is None or v == "" for v in versions):
                    viol.append(_violation("a_unknown_revision_committed", i, ev=ev))
                elif len(set(versions)) > 1:
                    viol.append(_violation("a_mixed_revision_committed", i, ev=ev))

        elif typ == "checkpoint":
            role = payload.get("role", "commit")
            if role != "commit":
                continue
            tok = ev.get("step_token")
            step = ev.get("step")
            checkpoint_hash = ev.get("checkpoint_hash")
            if not isinstance(tok, str) or not tok:
                viol.append(_violation("b_missing_step_token", i, ev=ev))
            else:
                if tok in claimed_tokens:
                    viol.append(_violation("b_duplicate_step_token", i, step_token=tok, ev=ev))
                claimed_tokens[tok] = i
            if not isinstance(step, int) or step < 0:
                viol.append(_violation("b_missing_step", i, ev=ev))
            else:
                if step in claimed_steps:
                    viol.append(_violation("b_duplicate_step", i, step=step, ev=ev))
                claimed_steps[step] = i
            if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
                viol.append(_violation("c_missing_checkpoint_hash", i, ev=ev))

            if auth_by_step:
                if step not in auth_by_step:
                    viol.append(_violation("b_unknown_step", i, step=step, ev=ev))
                else:
                    want = auth_by_step[step]
                    want_tok = want.get("step_token")
                    want_hash = want.get("checkpoint_hash")
                    if want_tok is not None and tok != want_tok:
                        viol.append(_violation("b_step_token_mismatch", i, step=step, ev=ev))
                    if want_hash is not None and checkpoint_hash != want_hash:
                        viol.append(_violation("c_token_hash_mismatch", i, step=step, ev=ev))
            if auth_by_token and tok not in auth_by_token:
                viol.append(_violation("b_unknown_step_token", i, step_token=tok, ev=ev))

        elif typ == "learner" and payload.get("op") == "apply":
            tok = payload.get("step_token", ev.get("step_token"))
            step = payload.get("step", ev.get("step"))
            if not isinstance(tok, str) or not tok:
                viol.append(_violation("b_apply_missing_step_token", i, ev=ev))
                continue
            apply_counts[tok] += 1
            apply_first_event.setdefault(tok, i)
            if auth_by_token and tok not in auth_by_token:
                viol.append(_violation("b_apply_unknown_step_token", i, step_token=tok, ev=ev))
            elif tok in auth_by_token and step != auth_by_token[tok]["step"]:
                viol.append(_violation("b_apply_step_mismatch", i, step_token=tok, ev=ev))

        elif typ == "reward":
            committed = payload.get("committed", ev.get("committed", False))
            if not committed:
                continue
            lid = payload.get("logical_id", ev.get("logical_id"))
            reward = payload.get("reward", ev.get("reward"))
            if auth_rewards:
                if not lid:
                    viol.append(_violation("d_missing_reward_identity", i, ev=ev))
                elif str(lid) not in auth_rewards:
                    viol.append(_violation("d_unknown_reward_committed", i, logical_id=str(lid), ev=ev))
                else:
                    committed_reward_counts[str(lid)] += 1
                    if reward != auth_rewards[str(lid)]:
                        viol.append(_violation("d_wrong_reward_committed", i, logical_id=str(lid), ev=ev))

    for tok, count in sorted(apply_counts.items()):
        if count > 1:
            viol.append(_violation("b_duplicate_apply", apply_first_event[tok], step_token=tok, count=count))
        if tok not in claimed_tokens:
            viol.append(_violation("b_apply_without_commit", apply_first_event[tok], step_token=tok))

    for step, item in sorted(auth_by_step.items()):
        if step not in claimed_steps:
            viol.append(_violation("b_missing_step", step=step))
        tok = item.get("step_token")
        if tok is not None and tok not in claimed_tokens:
            viol.append(_violation("b_missing_step_token", step_token=tok, step=step))

    for logical_id in sorted(auth_rewards):
        count = committed_reward_counts[logical_id]
        if count == 0:
            viol.append(_violation("d_missing_authoritative_reward", logical_id=logical_id))
        elif count > 1:
            viol.append(_violation("d_duplicate_reward_commit", logical_id=logical_id, count=count))

    return viol


def _events_of(events, typ, op=None):
    result = []
    for ev in events:
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        if ev.get("type") == typ and (op is None or payload.get("op") == op):
            result.append(ev)
    return result


def _group_ids(events, typ, op=None):
    out = []
    for ev in _events_of(events, typ, op):
        group_ids = ev.get("group_ids", [])
        if isinstance(group_ids, list):
            out.extend(group_ids)
    return out


def verify_trial(cut: str, events: list[dict], authoritative, schedule=None) -> tuple[bool, dict]:
    """Apply the generic oracle and the cut-specific protocol assertion."""
    schedule = schedule or {}
    viol = classify_event_log(events, authoritative)
    fault = [e for e in _events_of(events, "fault") if e.get("payload", {}).get("cut") == cut]
    checks = {"fault_marker_once": len(fault) == 1}

    if cut == "R1":
        rewards = _events_of(events, "reward", "cas_write")
        groups = _events_of(events, "group")
        checks.update(crash_persisted=any(e.get("payload", {}).get("reward") is None for e in rewards),
                      no_group_commit=not groups)
    elif cut in ("R2", "R3"):
        groups = _events_of(events, "group")
        aborted = [e for e in groups if e.get("payload", {}).get("status") == "ABORTED"]
        committed_rewards = [e for e in _events_of(events, "reward", "cas_write")
                             if e.get("payload", {}).get("committed")]
        checks.update(one_aborted_group=len(aborted) == 1,
                      no_sealed_group=not any(e.get("payload", {}).get("status") == "SEALED" for e in groups),
                      authoritative_group_size=len(committed_rewards) == schedule.get("group_size"))
    elif cut == "R4":
        checks.update(one_cas_winner=len([e for e in _events_of(events, "reward", "cas_write")
                                          if e.get("payload", {}).get("committed")]) == 1,
                      one_conflict_reject=len(_events_of(events, "reward", "cas_reject")) == 1)
    elif cut == "R5":
        workers = _events_of(events, "learner", "recovery_worker_exit")
        checks.update(one_cas_winner=len([e for e in _events_of(events, "reward", "cas_write")
                                          if e.get("payload", {}).get("committed")]) == 1,
                      workers_finished=len(workers) == 2 and all(e.get("payload", {}).get("exit_code") == 0
                                                                 for e in workers),
                      no_worker_timeout=not any(e.get("payload", {}).get("timed_out") for e in workers))
    elif cut == "Q0":
        consumed = set(_group_ids(events, "queue", "mark_consumed"))
        committed = set(_group_ids(events, "group"))
        planned = _group_ids(events, "recovery")
        need = consumed - committed
        checks.update(exact_reissue=set(planned) == need,
                      no_duplicate_reissue=len(planned) == len(set(planned)))
    elif cut == "Q1":
        fetched = set(_group_ids(events, "queue", "get_data"))
        planned = _group_ids(events, "recovery")
        checks.update(exact_reissue=set(planned) == fetched,
                      no_duplicate_reissue=len(planned) == len(set(planned)))
    elif cut in ("L0", "L3"):
        decisions = [e.get("recovery_decision") for e in _events_of(events, "recovery")]
        checks.update(no_commit_claim=not _events_of(events, "checkpoint"),
                      rollback_decision=decisions == (["cold_start"] if cut == "L0" else ["rollback"]))
    elif cut == "L2":
        decisions = [e.get("recovery_decision") for e in _events_of(events, "recovery")]
        rank_apply = _events_of(events, "learner", "rank_apply")
        checks.update(partial_apply=0 <= len(rank_apply) < schedule.get("group_size", 0),
                      rollback_decision=decisions == ["rollback"],
                      one_pre_step_checkpoint=len(_events_of(events, "checkpoint")) == 1)
    elif cut == "C1":
        decisions = [e.get("recovery_decision") for e in _events_of(events, "recovery")]
        checks.update(commit_ack_once=decisions == ["commit_ack"],
                      one_checkpoint=len(_events_of(events, "checkpoint")) == 1)
    elif cut == "C2":
        decisions = [e.get("recovery_decision") for e in _events_of(events, "recovery")]
        acks = _events_of(events, "ack")
        applies = _events_of(events, "learner", "apply")
        expected_loss = bool(schedule.get("ack_loss"))
        checks.update(commit_ack_once=decisions == ["commit_ack"],
                      one_authoritative_apply=len(applies) == 1,
                      ack_loss_matches=len(acks) == 1 and bool(acks[0].get("payload", {}).get("dropped")) == expected_loss)
    else:
        checks["known_cut"] = False

    ok = not viol and all(checks.values())
    return ok, {"violations": viol, "checks": checks}


def summarize_violations(viols: list[dict]) -> dict:
    counts = Counter(v["type"] for v in viols)
    return {"violations": len(viols), "by_type": dict(counts)}


def wilson_upper(k: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return min(1.0, center + half)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    if n <= 0:
        return 0.0
    if k == 0:
        return 1 - alpha ** (1.0 / n)
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = min((lo + hi) / 2, 1 - 1e-12)
        total = 0.0
        for j in range(k + 1):
            logp = (math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                    + j * math.log(mid) + (n - j) * math.log1p(-mid))
            total += math.exp(logp)
        if total >= alpha / 2:
            lo = mid
        else:
            hi = mid
    return lo


def rule_of_three(n: int) -> float:
    return 3.0 / n if n > 0 else 0.0


def aggregate(cut_results: dict[str, dict]) -> dict:
    report = {"cutpoints": {}, "total": {"n": 0, "failures": 0}}
    for cut, result in sorted(cut_results.items()):
        n, failures = int(result["n"]), int(result["failures"])
        report["cutpoints"][cut] = {
            "n": n,
            "failures": failures,
            "wilson_upper": round(wilson_upper(failures, n), 8),
            "cp_upper": round(clopper_pearson_upper(failures, n), 8),
        }
        report["total"]["n"] += n
        report["total"]["failures"] += failures
    total = report["total"]
    total["rule_of_three_upper"] = (round(rule_of_three(total["n"]), 8)
                                      if total["n"] and total["failures"] == 0 else None)
    report["status"] = "PASS" if total["failures"] == 0 else "FAIL"
    return report


def parse_authoritative_steps(spec: str):
    """Parse ``0..99``, comma/newline separated integers, or step:token:hash."""
    result = []
    for part in re.split(r"[\s,]+", spec.strip()):
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)\.\.(-?\d+)", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            direction = 1 if end >= start else -1
            result.extend(range(start, end + direction, direction))
        elif ":" in part:
            fields = part.split(":")
            if len(fields) != 3:
                raise ValueError("权威 step 完整形式必须是 step:token:hash")
            result.append({"step": int(fields[0]), "step_token": fields[1], "checkpoint_hash": fields[2]})
        else:
            result.append(int(part))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--authoritative-steps", default="")
    parser.add_argument("--aggregate", type=Path)
    args = parser.parse_args()
    if args.aggregate:
        report = aggregate(json.loads(args.aggregate.read_text()))
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "PASS" else 1
    if not args.events:
        parser.error("需要 --events 或 --aggregate")
    events = [json.loads(line) for line in args.events.open(encoding="utf-8") if line.strip()]
    try:
        authority = parse_authoritative_steps(args.authoritative_steps)
    except ValueError as exc:
        parser.error(str(exc))
    violations = classify_event_log(events, authority)
    print(json.dumps(summarize_violations(violations), indent=2, ensure_ascii=False))
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
