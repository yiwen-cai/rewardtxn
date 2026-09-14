#!/usr/bin/env python3
"""Read-only audit of E7 pilot completed-group loss and reward bindings.

The CPU reproduction executes two AST-extracted functions from the actual
fully-async implementation with a six-group in-memory queue.  Historical
consumption is read from diagnosis_summary.json, not re-derived from .pt files;
the companion raw-debug audit must validate that summary independently.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import re
import sys
import time
from types import MethodType, SimpleNamespace


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "third_party/slime/slime/rollout/fully_async_rollout.py"
DEFAULT_MANIFEST = BASE / "runs/e7_restart_0.5B_20260911_pilot.json"
WARM_RE = re.compile(r"fully-async rollout (\d+): target=(\d+) queue_warm=(\d+)")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def reproduce_queue_loss(source=SOURCE):
    tree = ast.parse(source.read_text())
    worker_class = next(node for node in tree.body
                        if isinstance(node, ast.ClassDef) and node.name == "AsyncRolloutWorker")
    drain = next(node for node in worker_class.body
                 if isinstance(node, ast.FunctionDef) and node.name == "get_completed_groups")
    collect = next(node for node in tree.body
                   if isinstance(node, ast.AsyncFunctionDef) and node.name == "_generate_rollout_async")
    future = next(node for node in tree.body
                  if isinstance(node, ast.ImportFrom) and node.module == "__future__")
    namespace = dict(Sample=SimpleNamespace, asyncio=asyncio, queue=queue,
                     logger=logging.getLogger("e7-pilot-queue-audit"), time=time, os=os)
    selected = ast.Module(body=[future, drain, collect], type_ignores=[])
    exec(compile(selected, str(source), "exec"), namespace)
    worker = SimpleNamespace(output_queue=queue.Queue())
    worker.get_completed_groups = MethodType(namespace["get_completed_groups"], worker)
    worker.queue_size = worker.output_queue.qsize
    for group_id in range(6):
        worker.output_queue.put((group_id, [SimpleNamespace(index=8 * group_id)]))
    namespace["_get_global_worker"] = lambda args, buffer: worker
    requeued = []
    buffer = SimpleNamespace(add_samples=requeued.extend)
    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=4)
    returned = asyncio.run(namespace["_generate_rollout_async"](args, 0, buffer))
    returned_ids = [group[0].index // 8 for group in returned]
    queue_after = worker.queue_size()
    next_drain = worker.get_completed_groups()
    reproduced = returned_ids == [0, 1, 2, 3] and queue_after == 0 and not requeued and not next_drain
    return {
        "source": str(source.relative_to(BASE)),
        "source_sha256": sha256(source),
        "input_group_ids": list(range(6)),
        "target_groups": 4,
        "returned_group_ids": returned_ids,
        "queue_size_after": queue_after,
        "next_drain_group_count": len(next_drain),
        "requeued_group_count": len(requeued),
        "completed_group_loss_reproduced": reproduced,
        "lost_group_ids": [4, 5] if reproduced else None,
    }


def reward_payload_parity(run_dirs):
    """Execute both real clean RM control flows with only audit I/O stubbed.

    Exactly 32 complete groups, evenly spaced through each B6 reward log,
    are selected.  Capped payloads are excluded, so independent regrading is
    not confused with the lossy 4,000-character audit representation.
    """
    math_path = BASE / "third_party/slime/slime/rollout/rm_hub/math_utils.py"
    math_namespace = {}
    # The host Python can predate PEP 604; postponing annotations changes no
    # verifier execution and avoids importing the GPU-serving package.
    exec(compile("from __future__ import annotations\n" + math_path.read_text(),
                 str(math_path), "exec"), math_namespace)
    paths = {"group_rm": BASE / "scripts/day2_custom_rm.py",
             "b6": BASE / "scripts/phase2_seal_rm.py"}
    selected_names = {
        "group_rm": {"_v1_reward", "_v2_reward", "_rm_one", "rm_function"},
        "b6": {"_v1_reward", "_v2_reward", "_fault_for", "_rm_batch_group",
               "_make_log_record", "rm_function"},
    }
    namespaces = {}
    verdict_asts = {}
    for group, path in paths.items():
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name in selected_names[group]]
        verdict_asts[group] = {}
        for node in selected:
            if node.name in ("_v1_reward", "_v2_reward"):
                normalized = ast.parse(ast.get_source_segment(path.read_text(), node)).body[0]
                if (isinstance(normalized.body[0], ast.Expr)
                        and isinstance(normalized.body[0].value, ast.Constant)
                        and isinstance(normalized.body[0].value.value, str)):
                    normalized.body = normalized.body[1:]
                verdict_asts[group][node.name] = ast.dump(normalized, include_attributes=False)
        namespace = {name: math_namespace[name] for name in
                     ("extract_answer", "grade_answer_mathd", "grade_answer_sympy")}
        namespace.update(asyncio=asyncio, time=time, FAULT="none", START=-1, END=-1,
                         V2_MODE="strict", SEAL=True, AUTO_FIX=True, GROUP_RM=True,
                         K=8, _WINDOWS={},
                         _log=lambda *args, **kwargs: None,
                         _cas_write_many=lambda records: [True] * len(records),
                         _seal_write=lambda *args, **kwargs: None)
        future = ast.parse("from __future__ import annotations").body[0]
        exec(compile(ast.Module(body=[future] + selected, type_ignores=[]), str(path), "exec"), namespace)
        namespaces[group] = namespace
    checks = []
    for run_dir in run_dirs:
        metadata = json.loads((run_dir / "meta.json").read_text())
        if metadata["baseline_mode"] != "b6":
            continue
        by_group = {}
        for row in read_jsonl(run_dir / "rewards.jsonl"):
            by_group.setdefault(row["group_index"], []).append(row)
        eligible = [rows for rows in by_group.values()
                    if len(rows) == 8 and all(len(row.get("response", "")) < 4000
                                             and len(row.get("label", "")) < 200 for row in rows)]
        count = min(32, len(eligible))
        positions = [i * (len(eligible) - 1) // max(1, count - 1) for i in range(count)]
        rows = [row for position in positions for row in eligible[position]]
        # Reverse the supplied order to test restoration to input positions,
        # not merely equal outputs for pre-sorted groups.
        rows.reverse()
        samples = [SimpleNamespace(group_index=row["group_index"], index=row["index"],
                                   rollout_id=row["rollout_id"], response=row["response"],
                                   label=row["label"]) for row in rows]
        async def run_rm(namespace):
            namespace["_lock"] = asyncio.Lock()
            return await namespace["rm_function"](SimpleNamespace(), samples)

        outputs = {group: asyncio.run(run_rm(namespace))
                   for group, namespace in namespaces.items()}
        checks.append({
            "run": str(run_dir.relative_to(BASE)), "eligible_complete_groups": len(eligible),
            "selected_group_ids": sorted({row["group_index"] for row in rows}),
            "samples": len(rows), "input_order": "reversed",
            "rm_output_lengths_match": all(len(output) == len(rows) for output in outputs.values()),
            "group_rm_vs_b6_mismatches": sum(a != b for a, b in zip(outputs["group_rm"], outputs["b6"])),
            "group_rm_vs_logged_reward_mismatches": sum(a != row["reward"] for a, row in zip(outputs["group_rm"], rows)),
            "b6_vs_logged_reward_mismatches": sum(a != row["reward"] for a, row in zip(outputs["b6"], rows)),
        })
    return {
        "source_sha256": {group: sha256(path) for group, path in paths.items()},
        "math_utils_sha256": sha256(math_path),
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "sympy": math_namespace["sympy"].__version__,
                    "pylatexenc": __import__("pylatexenc").__version__},
        "v1_and_v2_reward_function_asts_identical": verdict_asts["group_rm"] == verdict_asts["b6"],
        "audit_io_stubbed_only": True,
        "runs": checks,
        "limitation": "Sampled clean payload parity does not reproduce I/O timing, asynchronous scheduling, or fault/retry semantics.",
    }


def audit_run(run_dir, reproduction):
    metadata = json.loads((run_dir / "meta.json").read_text())
    summary_path = run_dir / "diagnosis_summary.json"
    summary = json.loads(summary_path.read_text())
    rewards = read_jsonl(run_dir / "rewards.jsonl")
    seals_path = run_dir / "seals.jsonl"
    seals = read_jsonl(seals_path) if seals_path.exists() else []
    rejects_path = run_dir / "cas_rejects.jsonl"
    rejects = read_jsonl(rejects_path) if rejects_path.exists() else []
    samples = [sample for batch in summary["consumed_rollouts"] for sample in batch["samples"]]
    reward_keys = Counter((row["group_index"], row["index"]) for row in rewards)
    reward_lookup = {(row["group_index"], row["index"]): row for row in rewards}
    consumed_keys = [(row["group_index"], row["sample_index"]) for row in samples]
    missing = [key for key in consumed_keys if key not in reward_lookup]
    mismatches = [key for key, sample in zip(consumed_keys, samples)
                  if key in reward_lookup and reward_lookup[key]["reward"] != sample["reward"]]
    reward_groups = {row["group_index"] for row in rewards}
    consumed_groups = {row["group_index"] for row in samples}
    log_path = run_dir / "logs/train.log"
    log = log_path.read_text(errors="replace")
    warm = []
    for line_number, line in enumerate(log.splitlines(), 1):
        match = WARM_RE.search(line)
        if match:
            rollout_id, target, queue_warm = map(int, match.groups())
            warm.append(dict(rollout_id=rollout_id, target=target,
                             queue_warm=queue_warm, log_line=line_number))
    rollout_counts = Counter(row["rollout_id"] for row in warm)
    unique_rollouts = all(count == 1 for count in rollout_counts.values())
    consistent_target = all(row["target"] == metadata["batch_groups_U"] for row in warm)
    lower_bound_valid = bool(warm) and unique_rollouts and consistent_target and reproduction["completed_group_loss_reproduced"]
    warm_excess = sum(max(0, row["queue_warm"] - row["target"]) for row in warm)
    return {
        "run": str(run_dir.relative_to(BASE)),
        "group": metadata["baseline_mode"],
        "seed": metadata["seed"],
        "summary_sha256": sha256(summary_path),
        "log_sha256": sha256(log_path),
        "consumed_batches_from_summary": len(summary["consumed_rollouts"]),
        "consumed_samples_from_summary": len(samples),
        "consumed_groups_from_summary": len(consumed_groups),
        "duplicate_consumed_group_sample_keys": len(consumed_keys) - len(set(consumed_keys)),
        "reward_log_rows": len(rewards),
        "reward_log_groups": len(reward_groups),
        "duplicate_reward_group_sample_keys": sum(count - 1 for count in reward_keys.values()),
        "consumed_reward_keys_missing": len(missing),
        "consumed_reward_value_mismatches": len(mismatches),
        "reward_groups_not_consumed": len(reward_groups - consumed_groups),
        "reward_verifiers": dict(Counter(row["verifier"] for row in rewards)),
        "injected_reward_rows": sum(bool(row.get("injected")) for row in rewards),
        "seal_statuses": dict(Counter(row["status"] for row in seals)),
        "autofix_seals": sum(bool(row.get("autofix")) for row in seals),
        "cas_rejections": len(rejects),
        "aborted_requeue_events": log.count("requeued aborted group"),
        "task_exception_events": log.count("process task raised"),
        "timeout_events": log.count("TIMEOUT"),
        "warm_log_count": len(warm),
        "warm_log_rollout_ids_unique": unique_rollouts,
        "warm_log_rollout_ids_complete": sorted(rollout_counts) == list(range(metadata["params"]["num_rollout"])),
        "warm_log_target_matches_config": consistent_target,
        "warm_exceeds_target_steps": sum(row["queue_warm"] > row["target"] for row in warm),
        "maximum_queue_warm": max((row["queue_warm"] for row in warm), default=0),
        "proven_minimum_completed_groups_discarded": warm_excess if lower_bound_valid else None,
        "warm_loss_evidence_examples": sorted(warm, key=lambda row: row["queue_warm"], reverse=True)[:5],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reward-parity", action="store_true",
                        help="also compare real clean RM outputs on 768 uncapped saved payloads (requires Python 3.10+, sympy/pylatexenc)")
    args = parser.parse_args()
    if args.reward_parity and sys.version_info < (3, 10):
        parser.error("--reward-parity requires Python 3.10+ because the original verifier uses zip(strict=False)")
    manifest = json.loads(args.manifest.read_text())
    reproduction = reproduce_queue_loss()
    result = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "queue_reproduction": reproduction,
        "runs": [audit_run(BASE / path, reproduction) for path in manifest["runs"]],
        "limitations": [
            "Consumption/binding checks use saved diagnosis_summary.json; raw .pt data require independent verification.",
            "The queue loss lower bound assumes the inspected source matches the runtime implementation, one collector, and no other queue consumer.",
            "Only excess already present at the logged rollout start contributes to the lower bound; concurrent arrivals can increase loss.",
            "Reward groups minus consumed groups also includes final prefetch/unconsumed tail; that difference is not the exact loss count.",
            "The reproduction establishes a shared scheduler defect, not that it caused or quantifies RewardTxn's validation accuracy deficit.",
            "Reward value binding is not independent regrading of response text; truncated reward log responses are not used as full replay inputs.",
        ],
    }
    if args.reward_parity:
        result["reward_payload_parity"] = reward_payload_parity([BASE / path for path in manifest["runs"]])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
