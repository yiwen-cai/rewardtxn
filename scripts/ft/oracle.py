"""Independent offline auditor for normalized observer evidence, schema 1.

This validates a declared evidence model, not authenticity of the observer or
correctness of a future backend adapter. No method recovery code is imported.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


COMPONENTS = {"model", "optimizer_master", "optimizer_moments", "optimizer_step", "scheduler",
              "rng_python", "rng_numpy", "rng_torch_cpu", "rng_device", "rng_tracker", "data", "policy"}


class EvidenceError(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise EvidenceError("duplicate JSON key")
        result[key] = value
    return result


def _json(data):
    return json.loads(data, object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(EvidenceError("nonfinite JSON")))


def _file(root, relative):
    if not isinstance(relative, str) or relative.startswith("/") or any(
            part in ("", ".", "..") for part in relative.split("/")):
        raise EvidenceError("unsafe artifact path")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise EvidenceError("symlink artifact")
    return path.read_bytes()


def _index(rows):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise EvidenceError("duplicate evidence ID")
        result[row["id"]] = row
    return result


def _reward(definition, raw):
    name = definition["name"]
    if name == "fixture-exact":
        return float(raw["completion"].strip() == raw["label"].strip())
    if name == "areal-gsm8k":
        # Explicit allowlist; never import a dotted path supplied by a log.
        import importlib.metadata
        import areal.reward
        import areal.reward.gsm8k
        from areal.reward.gsm8k import gsm8k_reward_fn
        sources = {"gsm8k.py": Path(areal.reward.gsm8k.__file__),
                   "__init__.py": Path(areal.reward.__file__)}
        if ({name: _sha(path.read_bytes()) for name, path in sources.items()} != definition["source_sha256"]
                or importlib.metadata.version("math-verify") != definition["math_verify_version"]):
            raise EvidenceError("actual verifier source/dependency mismatch")
        worker = areal.reward.get_math_verify_worker()
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        extraction = (ExprExtractionConfig(try_extract_without_anchor=True), LatexExtractionConfig())
        parameters = {"try_extract_without_anchor": True, "precision": 6, "timeout": 5.0}
        if (definition["parameters"] != parameters or worker.precision != 6 or worker.timeout != 5.0
                or worker.gold_extraction_target != extraction or worker.pred_extraction_target != extraction):
            raise EvidenceError("verifier parameters not frozen defaults")
        return float(gsm8k_reward_fn(raw["prompt"], raw["completion"], raw["prompt_ids"],
                                    raw["completion_ids"], answer=raw["label"]))
    raise EvidenceError("unsupported verifier")


def audit_run(evidence_dir, frozen_spec):
    """Return a report without writing to evidence or loading method state.

    frozen_spec has ``freeze`` (pre-run contract) and ``observer_seal`` (trusted
    post-run full-file hashes). A seal cannot authenticate its own producer.
    """
    report = {"schema": 1, "status": "unverifiable", "safety": "unverifiable",
              "training_continuation": "unverifiable", "affected_work_recovery": "unverifiable",
              "rto_seconds": None, "censored": True, "penalized_score": None,
              "missing_evidence": [], "violations": [], "retained_states": [],
              "retained_updates": [], "rolled_back_updates": [], "evidence_level": "unknown",
              "first_role_ready": None, "first_valid_update": None, "target_commit": None}
    missing, violations = report["missing_evidence"], report["violations"]
    try:
        root = Path(evidence_dir).absolute()
        if any(path.is_symlink() for path in (root, *root.parents)):
            raise EvidenceError("symlink evidence root")
        spec = frozen_spec if isinstance(frozen_spec, dict) else _json(Path(frozen_spec).read_bytes())
        freeze, seal = spec["freeze"], spec["observer_seal"]
        blobs = {}
        for name, digest in seal.items():
            data = _file(root, name)
            if _sha(data) != digest:
                raise EvidenceError("observer artifact hash mismatch: " + name)
            blobs[name] = data
        evidence = _json(blobs["evidence.json"])
        if evidence["freeze_sha256"] != _sha(_canonical(freeze)):
            raise EvidenceError("freeze binding mismatch")
        if evidence["run_nonce"] != freeze["run_nonce"] or freeze["schema"] != 1:
            raise EvidenceError("run/schema mismatch")
        report["evidence_level"] = freeze["evidence_level"]
        if freeze["evidence_level"] not in ("fixture", "observer-normalized"):
            raise EvidenceError("unknown evidence level")
        for name, digest in freeze["inputs"].items():
            if name not in blobs or _sha(blobs[name]) != digest:
                raise EvidenceError("frozen input mismatch: " + name)
        events = evidence["events"]
        by_event = _index(events)
        event_order = {event["id"]: index for index, event in enumerate(events)}
        previous = -float("inf")
        for event in events:
            if (event["run_nonce"] != freeze["run_nonce"] or event["timeline"] != freeze["timeline"]
                    or not math.isfinite(event["time"]) or event["time"] < previous):
                raise EvidenceError("unverified controller timeline")
            previous = event["time"]
            for key in ("role", "rank", "pid", "start_time", "boot_id", "cgroup", "event_nonce", "source_monotonic"):
                if key not in event:
                    raise EvidenceError("missing event identity: " + key)
        start, end = freeze["run_start"], evidence["end_time"]
        if not math.isfinite(end) or end < start or any(e["time"] < start or e["time"] > end for e in events):
            raise EvidenceError("invalid observation interval")
        applicability = freeze["applicability"]
        if applicability["status"] == "not_applicable":
            if not applicability.get("source") or not applicability.get("reason") or applicability["decided_at"] > start:
                raise EvidenceError("N/A was not justified before run")
            report["status"] = "not_applicable"
            report["applicability"] = applicability
            return report
        if applicability["status"] != "applicable":
            raise EvidenceError("unknown applicability")
        faults = [event for event in events if event["type"] == "fault_observed"
                  and event.get("fault_id") == freeze["fault_id"]]
        technical = [event for event in events if event["type"] == "technical_invalid"
                     and event.get("origin") in ("observer", "controller", "external")]
        if technical or len(faults) != 1:
            report["status"] = "technical_invalid"
            missing.append("technical control failure or fault not uniquely observed")
            return report
        fault = faults[0]
        fault_time = fault["time"]
        report["penalized_score"] = freeze["recovery_window_seconds"]
        if freeze["reward_transform"] != "identity" or freeze["evidence_level"] != "fixture":
            raise EvidenceError("unsupported training reward transform")
        if freeze["verifier"]["name"] not in ("fixture-exact", "areal-gsm8k"):
            raise EvidenceError("unsupported verifier")
        if freeze["verifier"]["name"] == "fixture-exact" and freeze["evidence_level"] != "fixture":
            raise EvidenceError("fixture verifier cannot certify real observer evidence")
        states, updates, groups = (_index(evidence[key]) for key in ("states", "updates", "groups"))
        final = evidence["final_state"]
        chain, cursor = [], final
        while cursor is not None:
            if cursor in chain:
                violations.append("state parent cycle")
                break
            if cursor not in states:
                raise EvidenceError("missing state ancestor: " + str(cursor))
            chain.append(cursor)
            cursor = states[cursor]["parent"]
        chain.reverse()
        report["retained_states"] = chain
        if not chain or chain[0] != freeze["initial_state"]:
            raise EvidenceError("chain not rooted at frozen initial state")

        def event_at(identifier, kind):
            if identifier not in by_event or by_event[identifier]["type"] != kind:
                raise EvidenceError("missing " + kind + " event: " + str(identifier))
            return by_event[identifier]

        state_times, state_loads, retained, target_times, valid_updates = {}, {}, [], [], []
        consumed_updates, consumed_groups = set(), set()
        prior_consumed = set()
        ranks = set(freeze["expected_ranks"])
        if not ranks or len(ranks) != len(freeze["expected_ranks"]):
            raise EvidenceError("invalid rank contract")
        for state_id in chain:
            state = states[state_id]
            components = state["components"]
            if set(components) != COMPONENTS or any(set(value) != ranks for value in components.values()):
                raise EvidenceError("incomplete state components/ranks: " + state_id)
            for component in components.values():
                for files in component.values():
                    if not files or any(name not in blobs for name in files):
                        raise EvidenceError("missing sealed state file: " + state_id)
            data = _json(blobs[state["data_file"]])
            drawn, consumed, pending = data["drawn"], data["consumed"], data["pending"]
            if (len(drawn) != len(set(drawn)) or len(consumed) != len(set(consumed))
                    or len(pending) != len(set(pending)) or set(consumed) & set(pending)
                    or set(drawn) != set(consumed) | set(pending) or data["cursor"] != len(drawn)):
                violations.append("data cursor/pending coverage mismatch: " + state_id)
            if state_id == chain[0]:
                initial = event_at(state["persist_event"], "initial_state")
                if initial["state"] != state_id or state["updates"]:
                    raise EvidenceError("invalid initial state binding")
                state_times[state_id] = initial["time"]
                prior_consumed = set(consumed)
                consumed_groups.update(consumed)
                continue
            before_errors = len(violations)
            scheduled = event_at(state["save_event"], "checkpoint_schedule")
            persisted = event_at(state["persist_event"], "checkpoint_persisted")
            finalized = [event_at(item, "checkpoint_finalize") for item in state["finalize_events"]]
            if (len(finalized) != len(ranks) or {item["actor_rank"] for item in finalized} != ranks
                    or any(item["state"] != state_id or item["snapshot_id"] != state["snapshot_id"] for item in [scheduled, persisted, *finalized])
                    or any(not scheduled["time"] <= item["time"] <= persisted["time"] for item in finalized)):
                raise EvidenceError("incomplete checkpoint completion mapping: " + state_id)
            state_times[state_id] = persisted["time"]
            loads_before_save = [e for e in events if e["type"] == "checkpoint_loaded"
                                 and event_order[e["id"]] < event_order[scheduled["id"]]]
            if not loads_before_save:
                raise EvidenceError("missing checkpoint load for save execution")
            save_load = loads_before_save[-1]
            state_loads[state_id] = save_load["id"]
            parent = state["parent"]
            parent_persist = event_at(states[parent]["persist_event"],
                                      "initial_state" if parent == chain[0] else "checkpoint_persisted")
            # A reload resets the actual execution state even when epoch is
            # reused. An older ancestor load cannot preserve a later parent.
            if (save_load["epoch"] != state["epoch"]
                    or (save_load["state"] != parent and state_loads.get(parent) != save_load["id"])
                    or event_order[parent_persist["id"]] >= event_order[scheduled["id"]]
                    or (save_load["state"] == parent
                        and event_order[save_load["id"]] <= event_order[parent_persist["id"]])):
                violations.append("saved parent is not retained by actual load execution: " + state_id)
            lower = max(state_times[parent], save_load["time"])
            actual_updates = [e["update"] for e in events if e["type"] == "optimizer_end"
                              and e.get("successful") is True and e["epoch"] == state["epoch"]
                              and lower < e["time"] <= scheduled["time"]]
            if actual_updates != state["updates"]:
                violations.append("checkpoint omits/reorders actual optimizer updates: " + state_id)
            state_groups = set()
            for physical in state["updates"]:
                local_errors = len(violations)
                if physical not in updates:
                    raise EvidenceError("missing optimizer mapping: " + physical)
                update = updates[physical]
                if update["epoch"] != state["epoch"]:
                    violations.append("checkpoint/optimizer epoch mismatch: " + physical)
                begin = event_at(update["start_event"], "optimizer_start")
                finish = event_at(update["end_event"], "optimizer_end")
                if (begin["update"] != physical or finish["update"] != physical
                        or begin["epoch"] != update["epoch"] or finish["epoch"] != update["epoch"]
                        or not begin["time"] <= finish["time"] <= scheduled["time"]):
                    raise EvidenceError("optimizer event mapping mismatch: " + physical)
                if finish.get("successful") is not True or finish.get("scheduler_applied") is not True:
                    violations.append("unsuccessful optimizer retained: " + physical)
                loads = [item for item in events if item["type"] == "checkpoint_loaded"
                         and event_order[item["id"]] < event_order[begin["id"]]]
                if not loads:
                    raise EvidenceError("missing actual loaded state for update: " + physical)
                loaded = loads[-1]
                if (loaded["id"] != save_load["id"] or loaded["epoch"] != update["epoch"]
                        or event_order[begin["id"]] <= event_order[parent_persist["id"]]):
                    violations.append("optimizer/save execution interrupted by reload or parent mismatch: " + physical)
                ancestors = chain[:chain.index(state_id)]
                if loaded["state"] not in ancestors or loaded["state"] != update["loaded_state"]:
                    violations.append("update not descended from actual load: " + physical)
                elif loaded["time"] < state_times[loaded["state"]]:
                    raise EvidenceError("load predates durable state")
                if update["logical_id"] in consumed_updates or physical in retained:
                    violations.append("duplicate retained logical update: " + update["logical_id"])
                consumed_updates.add(update["logical_id"])
                retained.append(physical)
                tensor = _json(blobs[update["tensor_file"]])
                if tensor["transform"] != "identity" or begin["tensor_sha256"] != seal[update["tensor_file"]]:
                    raise EvidenceError("training tensor mapping/transform unsupported")
                rows = []
                hits_target = False
                for group_id in update["groups"]:
                    if group_id not in groups:
                        raise EvidenceError("missing group: " + group_id)
                    group = groups[group_id]
                    logical = group["logical_id"]
                    if group["source_file"] not in freeze["inputs"]:
                        raise EvidenceError("group source not frozen")
                    source = _json(blobs[group["source_file"]])[group["source_index"]]
                    if logical in consumed_groups or logical in state_groups:
                        violations.append("duplicate retained consumption: " + logical)
                    state_groups.add(logical)
                    hits_target = hits_target or logical == freeze["target_group"]
                    samples = group["samples"]
                    if group["k"] != freeze["k"] or sorted(s["index"] for s in samples) != list(range(group["k"])):
                        violations.append("incomplete logical group: " + logical)
                    versions = set()
                    for sample in samples:
                        auths = [e for e in events if e["type"] == "authorize" and e["group"] == logical
                                 and e["sample"] == sample["index"] and e["time"] <= begin["time"]]
                        if not auths:
                            raise EvidenceError("missing sample authorization")
                        auth = auths[-1]
                        if (auth["attempt"] != sample["attempt"] or auth["epoch"] != update["epoch"]
                                or auth["policy_version"] != sample["policy_version"]
                                or sample["verifier_version"] != freeze["verifier"]["version"]
                                or auth["verifier_version"] != sample["verifier_version"]):
                            violations.append("stale/unauthorized sample: " + logical)
                        if not 0 <= update["policy_version"] - sample["policy_version"] <= freeze["max_policy_staleness"]:
                            violations.append("policy staleness violation: " + logical)
                        versions.add(sample["verifier_version"])
                        raw = _json(blobs[sample["payload_file"]])
                        if raw["prompt"] != source["prompt"] or raw["label"] != source["label"]:
                            violations.append("payload prompt/label differs from frozen source: " + logical)
                        if (raw["prompt_sha256"] != group["prompt_sha256"]
                                or raw["prompt_sha256"] != _sha(_canonical(raw["prompt"]))
                                or raw["policy_version"] != sample["policy_version"]):
                            violations.append("response/prompt/policy mismatch: " + logical)
                        if not (len(raw["tokens"]) == len(raw["loss_mask"]) == len(raw["logprobs"])):
                            violations.append("incomplete training payload: " + logical)
                        actual = _reward(freeze["verifier"], raw)
                        if not math.isfinite(sample["reward"]) or actual != sample["reward"]:
                            violations.append("independent reward mismatch: " + logical)
                        rows.append({"group": logical, "sample": sample["index"], "attempt": sample["attempt"],
                                     "payload_sha256": seal[sample["payload_file"]], "reward": sample["reward"],
                                     "tokens": raw["tokens"], "loss_mask": raw["loss_mask"], "logprobs": raw["logprobs"]})
                    if len(versions) != 1:
                        violations.append("mixed group verifier versions: " + logical)
                if tensor["rows"] != rows:
                    violations.append("actual training tensor differs from admitted samples: " + physical)
                if len(violations) == local_errors and finish["successful"] is True:
                    valid_updates.append(finish["time"])
                    if hits_target:
                        target_times.append(persisted["time"])
            if set(consumed) != prior_consumed | state_groups:
                violations.append("checkpoint consumption mapping differs from optimizer inputs: " + state_id)
            consumed_groups.update(state_groups)
            prior_consumed = set(consumed)
            if len(violations) > before_errors:
                target_times = []
        report["retained_updates"] = retained
        report["rolled_back_updates"] = sorted(set(updates) - set(retained))
        all_loads = [e for e in events if e["type"] == "checkpoint_loaded"]
        if not all_loads or all_loads[-1]["state"] not in chain or all_loads[-1]["time"] > state_times[final]:
            raise EvidenceError("final state not linked to latest actual load")
        report["safety"] = "invalid_commit" if violations else "pass"
        valid_after = [time for time in valid_updates if time >= fault_time]
        report["first_valid_update"] = min(valid_after) if valid_after else None
        report["training_continuation"] = "continued" if valid_after else (
            "safe_stop" if any(e["type"] == "safe_stop" and e["time"] >= fault_time for e in events)
            else "timeout" if end - fault_time >= freeze["recovery_window_seconds"] else "unverifiable")
        role_events = [event for event in events if event["type"] in ("role_ready", "role_down")
                       and event["role"] == freeze["target_role"]]
        role_events.append(dict(fault, type="role_down", role=freeze["target_role"]))
        role_events.sort(key=lambda event: event["time"])
        report["first_role_ready"] = next((e["time"] for e in role_events if e["type"] == "role_ready" and e["time"] >= fault_time), None)
        commits = [time for time in target_times if time >= fault_time]
        report["target_commit"] = min(commits) if commits else None
        candidate_times = sorted(set(commits + [e["time"] for e in role_events if e["time"] >= fault_time]))
        recovered_at = None
        for time in candidate_times:
            current = [e for e in role_events if e["time"] <= time]
            if commits and min(commits) <= time and current and current[-1]["type"] == "role_ready":
                recovered_at = time
                break
        dropped = any(e["type"] == "work_dropped" and e["group"] == freeze["target_group"] and e["time"] >= fault_time for e in events)
        report["affected_work_recovery"] = "recovered" if recovered_at is not None and not violations else (
            "safely_dropped" if dropped and not violations else "unresolved")
        if violations:
            report["status"] = "invalid_commit"
        elif recovered_at is not None and recovered_at - fault_time <= freeze["recovery_window_seconds"]:
            report.update(status="correct_recovered", rto_seconds=recovered_at - fault_time,
                          censored=False, penalized_score=recovered_at - fault_time)
        else:
            report["status"] = "timeout" if end - fault_time >= freeze["recovery_window_seconds"] else "safe_stop" if report["training_continuation"] == "safe_stop" else "unverifiable"
    except (EvidenceError, KeyError, TypeError, ValueError, OSError, ImportError, IndexError, AttributeError) as exc:
        missing.append(str(exc))
        report["status"] = "invalid_commit" if violations else "unverifiable"
        report["safety"] = "invalid_commit" if violations else "unverifiable"
        report["affected_work_recovery"] = "unverifiable"
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evidence = Path(args.evidence).resolve()
    output = Path(args.output).resolve()
    if output == evidence or evidence in output.parents or output == Path(args.spec).resolve():
        parser.error("output must be outside read-only evidence and must not overwrite spec")
    report = audit_run(evidence, args.spec)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
