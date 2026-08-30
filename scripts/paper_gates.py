#!/usr/bin/env python3
"""Recompute the E2 paper gate from the self-contained raw run artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_oracle

BASE = Path(__file__).resolve().parent.parent
CUT_POINTS = ["R1", "R2", "R3", "R4", "R5", "Q0", "Q1", "L0", "L2", "L3", "C1", "C2"]
MIN_PER_CUT = 1500
MIN_TOTAL = 30000
FORMAL_PER_CUT = 2500
SCHEDULE_LEVELS = {
    "kill_time": [0.1, 0.35, 0.65, 0.9],
    "ack_loss": [False, True],
    "attempt_order": ["forward", "reverse", "interleaved"],
    "revision": ["v1", "v2", "v3"],
    "group_size": [4, 8, 16],
    "checkpoint_delay": [0, 1, 4],
}
RELEVANT_DIMENSIONS = {
    "R1": ["kill_time", "group_size"], "R2": ["revision", "group_size"],
    "R3": ["revision", "group_size"], "R4": ["attempt_order", "revision"],
    "R5": ["attempt_order", "revision"],
    "Q0": ["kill_time", "attempt_order", "revision", "group_size"],
    "Q1": ["kill_time", "attempt_order", "group_size"],
    "L0": ["kill_time", "checkpoint_delay"],
    "L2": ["kill_time", "attempt_order", "group_size", "checkpoint_delay"],
    "L3": ["kill_time", "checkpoint_delay"],
    "C1": ["kill_time", "revision", "checkpoint_delay"],
    "C2": ["kill_time", "ack_loss", "attempt_order", "revision", "checkpoint_delay"],
}
REQUIRED_FILES = {
    "meta.json", "config.json", "schedule.json", "events.jsonl", "verdicts.jsonl",
    "metrics.json", "resource.jsonl", "verdict.json", "TRACE_REPORT_PAPER.json",
    "stdout.log", "stderr.log", "exit_status.json", "manifests/fixture_manifest.json",
}


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


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("%s:%d is not an object" % (path.name, number))
            rows.append(value)
    return rows


def _git_commit():
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(BASE), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _dimension_value(seed, cut, trial, dimension):
    levels = SCHEDULE_LEVELS[dimension]
    material = "%s:%s:%s" % (seed, cut, dimension)
    local = random.Random(int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16))
    permutation = list(levels)
    local.shuffle(permutation)
    return permutation[trial % len(permutation)]


def _expected_schedule_row(seed, cut, trial):
    dimensions = {name: _dimension_value(seed, cut, trial, name) for name in SCHEDULE_LEVELS}
    core = {"cut": cut, "trial": trial, "dimensions": dimensions,
            "relevant_dimensions": RELEVANT_DIMENSIONS[cut]}
    return {**core, "schedule_id": _sha_bytes(_canonical(core))[:20]}


def _schedule_coverage(rows):
    coverage = {}
    for row in rows:
        cut = row["cut"]
        coverage.setdefault(cut, {})
        for dimension in row["relevant_dimensions"]:
            key = json.dumps(row["dimensions"][dimension], sort_keys=True)
            counts = coverage[cut].setdefault(dimension, {})
            counts[key] = counts.get(key, 0) + 1
    return coverage


def _validate_manifest(run_dir):
    errors = []
    path = run_dir / "artifact_manifest.json"
    if not path.exists():
        return False, ["artifact_manifest.json missing"]
    try:
        manifest = _load_json(path)
        entries = manifest.get("files", [])
        indexed = {entry["path"]: entry for entry in entries}
    except (ValueError, KeyError, TypeError) as exc:
        return False, ["invalid artifact manifest: %s" % exc]
    if len(indexed) != len(entries):
        errors.append("manifest contains duplicate paths")
    missing = sorted(REQUIRED_FILES - set(indexed))
    if missing:
        errors.append("manifest missing required paths: %s" % missing)
    for relative, entry in indexed.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append("manifest path escapes run dir: %s" % relative)
            continue
        target = run_dir / relative
        if not target.is_file():
            errors.append("manifest target missing: %s" % relative)
            continue
        if target.stat().st_size != entry.get("bytes"):
            errors.append("size mismatch: %s" % relative)
        if _sha_file(target) != entry.get("sha256"):
            errors.append("sha256 mismatch: %s" % relative)
    return not errors, errors


def _validate_schedule(config, schedule):
    errors = []
    if config.get("schedule_levels") != SCHEDULE_LEVELS:
        errors.append("config schedule levels differ from frozen gate")
    if config.get("relevant_dimensions") != RELEVANT_DIMENSIONS:
        errors.append("config relevant dimensions differ from frozen gate")
    if config.get("cuts") != CUT_POINTS:
        errors.append("config cut list differs from frozen gate")
    seed = config.get("seed")
    per_cut = config.get("per_cut")
    rows = schedule.get("trials")
    if not isinstance(seed, int) or not isinstance(per_cut, int) or not isinstance(rows, list):
        return False, ["schedule/config shape invalid"], {}, []
    expected_count = len(CUT_POINTS) * per_cut
    if len(rows) != expected_count:
        errors.append("schedule row count %d != %d" % (len(rows), expected_count))
    if schedule.get("seed") != seed:
        errors.append("schedule seed differs from config")
    by_key = {}
    valid_rows = []
    for row in rows:
        try:
            key = (row["cut"], row["trial"])
        except (KeyError, TypeError):
            errors.append("malformed schedule row")
            continue
        if key in by_key:
            errors.append("duplicate schedule trial %r" % (key,))
        by_key[key] = row
        if key[0] not in CUT_POINTS or not isinstance(key[1], int) or not 0 <= key[1] < per_cut:
            errors.append("out-of-range schedule trial %r" % (key,))
            continue
        if row != _expected_schedule_row(seed, key[0], key[1]):
            errors.append("non-deterministic schedule row %r" % (key,))
        else:
            valid_rows.append(row)
    expected_keys = {(cut, trial) for cut in CUT_POINTS for trial in range(per_cut)}
    missing = expected_keys - set(by_key)
    if missing:
        errors.append("missing schedule trials: %d" % len(missing))
    coverage = _schedule_coverage(valid_rows)
    if schedule.get("coverage") != coverage:
        errors.append("reported schedule coverage mismatch")
    for cut in CUT_POINTS:
        for dimension in RELEVANT_DIMENSIONS[cut]:
            counts = coverage.get(cut, {}).get(dimension, {})
            expected_levels = {json.dumps(value, sort_keys=True) for value in SCHEDULE_LEVELS[dimension]}
            if set(counts) != expected_levels or max(counts.values(), default=0) - min(counts.values(), default=0) > 1:
                errors.append("unbalanced schedule coverage %s/%s" % (cut, dimension))
    return not errors, errors, by_key, rows


def _validate_raw_trials(events, verdicts, schedule_by_key, per_cut):
    errors = []
    events_by_key = {}
    for event in events:
        key = (event.get("cut"), event.get("trial"))
        events_by_key.setdefault(key, []).append(event)
    verdict_by_key = {}
    cut_results = {cut: {"n": 0, "failures": 0} for cut in CUT_POINTS}
    first_invalid_index = None
    for index, verdict in enumerate(verdicts):
        key = (verdict.get("cut"), verdict.get("trial"))
        if key in verdict_by_key:
            errors.append("duplicate raw verdict %r" % (key,))
            continue
        verdict_by_key[key] = verdict
        schedule = schedule_by_key.get(key)
        trial_events = events_by_key.get(key, [])
        if schedule is None:
            errors.append("verdict has no schedule %r" % (key,))
            continue
        if verdict.get("schedule_id") != schedule["schedule_id"] or verdict.get("schedule") != schedule["dimensions"]:
            errors.append("verdict schedule mismatch %r" % (key,))
        for sequence, event in enumerate(trial_events):
            if (event.get("schedule_id") != schedule["schedule_id"] or event.get("sequence") != sequence
                    or event.get("exp_id") != "paper-e2-trace"):
                errors.append("event identity/order mismatch %r" % (key,))
                break
        if verdict.get("event_count") != len(trial_events):
            errors.append("event count mismatch %r" % (key,))
        if verdict.get("events_sha256") != _sha_bytes(_canonical(trial_events)):
            errors.append("event digest mismatch %r" % (key,))
        try:
            ok, detail = trace_oracle.verify_trial(key[0], trial_events, verdict.get("authoritative"),
                                                   schedule["dimensions"])
        except Exception as exc:
            ok, detail = False, {"oracle_error": "%s: %s" % (type(exc).__name__, exc)}
            errors.append("oracle could not evaluate %r" % (key,))
        if verdict.get("status") != ("PASS" if ok else "FAIL") or verdict.get("detail") != detail:
            errors.append("recorded oracle verdict mismatch %r" % (key,))
        if key[0] in cut_results:
            cut_results[key[0]]["n"] += 1
            if not ok:
                cut_results[key[0]]["failures"] += 1
                if first_invalid_index is None:
                    first_invalid_index = index

    expected_keys = set(schedule_by_key)
    missing_verdicts = expected_keys - set(verdict_by_key)
    missing_events = expected_keys - set(events_by_key)
    if missing_verdicts:
        errors.append("missing raw verdicts: %d" % len(missing_verdicts))
    if missing_events:
        errors.append("missing trial events: %d" % len(missing_events))
    unknown_events = set(events_by_key) - expected_keys
    if unknown_events:
        errors.append("events outside schedule: %d" % len(unknown_events))
    if first_invalid_index is not None and first_invalid_index != len(verdicts) - 1:
        errors.append("runner did not fail fast on first invalid commit")
    expected_order = [(cut, trial) for cut in CUT_POINTS for trial in range(per_cut)]
    actual_order = [(item.get("cut"), item.get("trial")) for item in verdicts]
    if actual_order != expected_order:
        errors.append("raw verdict order differs from frozen schedule order")
    if len(verdicts) != len(CUT_POINTS) * per_cut:
        errors.append("raw verdict count is incomplete")
    return not errors, errors, cut_results


def evaluate_trace(report: dict, run_dir=None) -> dict:
    gates = {}
    errors = {}
    if run_dir is None:
        gates["G0_self_contained_artifact"] = False
        errors["artifact"] = ["run directory is required; aggregate fields are not evidence"]
        return _finish(gates, errors, None, report)
    run_dir = Path(run_dir)

    artifact_ok, artifact_errors = _validate_manifest(run_dir)
    gates["G0_artifact_integrity"] = artifact_ok
    if artifact_errors:
        errors["artifact"] = artifact_errors
    try:
        meta = _load_json(run_dir / "meta.json")
        config = _load_json(run_dir / "config.json")
        schedule = _load_json(run_dir / "schedule.json")
        exit_status = _load_json(run_dir / "exit_status.json")
        raw_verdict = _load_json(run_dir / "verdict.json")
        fixture_manifest = _load_json(run_dir / "manifests" / "fixture_manifest.json")
        events = _load_jsonl(run_dir / "events.jsonl")
        verdicts = _load_jsonl(run_dir / "verdicts.jsonl")
    except (OSError, ValueError, TypeError) as exc:
        gates["G0_parseable_raw_artifacts"] = False
        errors["parse"] = [str(exc)]
        return _finish(gates, errors, None, report)
    gates["G0_parseable_raw_artifacts"] = True

    identity_errors = []
    config_sha = _sha_file(run_dir / "config.json")
    schedule_sha = _sha_file(run_dir / "schedule.json")
    if meta.get("config_sha256") != config_sha or report.get("config_sha256") != config_sha:
        identity_errors.append("config_sha256 mismatch")
    if meta.get("schedule_sha256") != schedule_sha or report.get("schedule_sha256") != schedule_sha:
        identity_errors.append("schedule_sha256 mismatch")
    current_commit = _git_commit()
    if not current_commit or meta.get("commit_sha") != current_commit or report.get("commit_sha") != current_commit:
        identity_errors.append("commit_sha does not match current checkout")
    if not meta.get("working_tree_clean"):
        identity_errors.append("formal run was produced from a dirty worktree")
    for relative, expected_hash in meta.get("source_sha256", {}).items():
        source = BASE / relative
        if not source.is_file() or _sha_file(source) != expected_hash:
            identity_errors.append("source hash mismatch: %s" % relative)
    gates["G0_commit_and_config_identity"] = not identity_errors
    if identity_errors:
        errors["identity"] = identity_errors

    formal_errors = []
    if config.get("mode") != "formal" or report.get("mode") != "formal":
        formal_errors.append("artifact is not a formal run")
    if config.get("per_cut") != FORMAL_PER_CUT:
        formal_errors.append("formal per_cut must be %d" % FORMAL_PER_CUT)
    if not exit_status.get("completed") or exit_status.get("return_code") != 0:
        formal_errors.append("runner did not complete successfully")
    if raw_verdict.get("status") != "PASS":
        formal_errors.append("raw verdict is not PASS")
    if fixture_manifest.get("manifest_type") != "typed-fixture-evidence":
        formal_errors.append("typed fixture manifest missing")
    gates["G0_formal_completed_run"] = not formal_errors
    if formal_errors:
        errors["formal"] = formal_errors

    schedule_ok, schedule_errors, schedule_by_key, _ = _validate_schedule(config, schedule)
    gates["G0_schedule_integrity"] = schedule_ok
    if schedule_errors:
        errors["schedule"] = schedule_errors[:25]

    per_cut = config.get("per_cut") if isinstance(config.get("per_cut"), int) else 0
    raw_ok, raw_errors, cut_results = _validate_raw_trials(
        events, verdicts, schedule_by_key, per_cut)
    gates["G0_raw_trial_integrity"] = raw_ok
    if raw_errors:
        errors["raw_trials"] = raw_errors[:25]

    recomputed = trace_oracle.aggregate(cut_results)
    aggregate_matches = (report.get("cutpoints") == recomputed["cutpoints"]
                         and report.get("total") == recomputed["total"]
                         and report.get("status") == recomputed["status"])
    gates["G0_report_matches_raw"] = aggregate_matches
    if not aggregate_matches:
        errors["aggregate"] = ["TRACE_REPORT_PAPER.json differs from raw recomputation"]

    gates["G1_zero_failures"] = all(cut_results[cut]["failures"] == 0 for cut in CUT_POINTS)
    gates["G2_min_per_cut"] = all(cut_results[cut]["n"] >= MIN_PER_CUT for cut in CUT_POINTS)
    gates["G3_total_30000"] = recomputed["total"]["n"] >= MIN_TOTAL
    gates["G3_rule_of_three_recomputed"] = recomputed["total"].get("rule_of_three_upper") is not None
    gates["G4_upper_bounds_recomputed"] = all(
        recomputed["cutpoints"][cut].get("cp_upper") is not None
        and recomputed["cutpoints"][cut].get("wilson_upper") is not None for cut in CUT_POINTS)
    return _finish(gates, errors, recomputed, report)


def _finish(gates, errors, recomputed, report):
    total = recomputed.get("total", {}) if recomputed else {}
    missing = [cut for cut in CUT_POINTS if cut not in (recomputed or {}).get("cutpoints", {})]
    return {"status": "PASS" if gates and all(gates.values()) else "FAIL", "gates": gates,
            "errors": errors, "summary": {"total_n": total.get("n", 0),
            "total_failures": total.get("failures", 0),
            "rule_of_three_upper": total.get("rule_of_three_upper"), "missing_cuts": missing,
            "reported_status": report.get("status")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-report", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.trace_report
    if report_path.is_dir():
        run_dir = report_path
        report_path = run_dir / "TRACE_REPORT_PAPER.json"
    else:
        run_dir = report_path.parent
    try:
        report = _load_json(report_path)
        verdict = evaluate_trace(report, run_dir)
    except (OSError, ValueError) as exc:
        verdict = {"status": "FAIL", "gates": {"G0_report_parseable": False},
                   "errors": {"report": [str(exc)]}, "summary": {}}
    print(json.dumps(verdict, indent=2, ensure_ascii=False))
    return 0 if verdict["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
