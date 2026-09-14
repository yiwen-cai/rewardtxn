#!/usr/bin/env python3
"""Build the final descriptive report for the 2026-08-31 P0 GPU pilot.

This report intentionally separates the completed short training pilot from the
formal paper endpoints.  It treats seed-paired runs as independent units and
uses the repository's paper_statistics.py implementation for paired bootstrap
intervals.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, median

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
RUNS = REPO / "runs"
SOURCE_RESULT = RUNS / "pilot_results_20260831-152801.json"
FINAL_RUN = RUNS / "pilot-E7-clean-rewardtxn-1.5B-4gpu-s42-175253"
FINAL_CONTAINER = "rtx-p2-pilot-E7-clean-rewardtxn-1.5B-4gpu-s42-175253"
FINAL_JSON = RUNS / "pilot_results_20260831-final.json"
FINAL_MD = RUNS / "PILOT_RESULTS_20260831.md"

# The first execution failed before training because the Docker GPU request was
# malformed.  The second execution produced 17 successful runs, then stopped
# at the resource gate when /public fell below its 200 GB floor.  These are
# retained as execution incidents rather than mixed into the 18-run result set.
INCIDENTS = [
    {
        "attempt_result_file": "runs/pilot_results_20260831-152631.json",
        "run_id": "pilot-E3-b0-1.5B-4gpu-s17-152631",
        "status": "launch_failed_before_training",
        "cause": "Docker rejected simultaneous Count and DeviceIDs in --gpus request",
        "evidence": "runs/pilot-E3-b0-1.5B-4gpu-s17-152631.launch.log",
    },
    {
        "attempt_result_file": "runs/pilot_results_20260831-152801.json",
        "run_id": "pilot-E7-clean-rewardtxn-1.5B-4gpu-s42-174446",
        "status": "resource_gate_failed_before_training",
        "cause": "/public available space was 170 GB, below the configured 200 GB minimum",
        "evidence": "runs/pilot-E7-clean-rewardtxn-1.5B-4gpu-s42-174446.launch.log",
    },
]

EXPECTED_KEYS = {
    ("E3", "b0", 17), ("E3", "b6", 17),
    ("E3", "b0", 29), ("E3", "b6", 29),
    ("E3", "b0", 42), ("E3", "b6", 42),
    ("E6", "group-rm-only", 17), ("E6", "rewardtxn", 17),
    ("E6", "group-rm-only", 29), ("E6", "rewardtxn", 29),
    ("E6", "group-rm-only", 42), ("E6", "rewardtxn", 42),
    ("E7", "clean-oracle", 17), ("E7", "clean-rewardtxn", 17),
    ("E7", "clean-oracle", 29), ("E7", "clean-rewardtxn", 29),
    ("E7", "clean-oracle", 42), ("E7", "clean-rewardtxn", 42),
}


def quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def rounded(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def summary(values, digits=4):
    values = [float(value) for value in values]
    return {
        "n": len(values),
        "mean": rounded(mean(values), digits),
        "median": rounded(median(values), digits),
        "p95": rounded(quantile(values, 0.95), digits),
        "min": rounded(min(values), digits),
        "max": rounded(max(values), digits),
        "values": [rounded(value, digits) for value in values],
    }


def count_lines(path):
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def parse_training_log(run_dir):
    text = (run_dir / "logs" / "train.log").read_text(encoding="utf-8", errors="replace")
    steps = sorted({int(value) for value in re.findall(r"(?m)step (\d+):", text)})
    perf = {}
    pattern = re.compile(r"\bperf\s+(\d+):\s+\{.*?'perf/step_time':\s*([0-9.eE+-]+)")
    for match in pattern.finditer(text):
        perf[int(match.group(1))] = float(match.group(2))
    if steps != list(range(30)):
        raise RuntimeError(f"{run_dir}: expected steps 0..29, observed {steps}")
    if sorted(perf) != list(range(30)):
        raise RuntimeError(f"{run_dir}: expected perf records 0..29, observed {sorted(perf)}")
    stable = [perf[index] for index in range(20, 30)]
    return {
        "steps_observed": len(steps),
        "step_range": [steps[0], steps[-1]],
        "stable_window_steps": [20, 29],
        "stable_step_time_seconds": [rounded(value, 6) for value in stable],
        "stable_step_time_p50_seconds": rounded(median(stable), 6),
        "stable_step_time_p95_seconds": rounded(quantile(stable, 0.95), 6),
        "stable_step_rate_per_second": rounded(1.0 / median(stable), 6),
    }


def docker_datetime(value):
    normalized = value.replace("Z", "+00:00")
    if "." in normalized:
        prefix, fraction_and_zone = normalized.split(".", 1)
        if "+" in fraction_and_zone:
            fraction, zone = fraction_and_zone.split("+", 1)
            normalized = f"{prefix}.{fraction[:6]}+{zone}"
    return dt.datetime.fromisoformat(normalized)


def inspect_container(container):
    command = ["docker", "inspect", "--format", "{{json .State}}", container]
    completed = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE)
    state = json.loads(completed.stdout)
    started = docker_datetime(state["StartedAt"])
    finished = docker_datetime(state["FinishedAt"])
    runtime = (finished - started).total_seconds()
    return {
        "container_status": state["Status"],
        "container_exit_code": int(state["ExitCode"]),
        "container_started_at": state["StartedAt"],
        "container_finished_at": state["FinishedAt"],
        "container_runtime_seconds": rounded(runtime, 3),
        "wall_time_seconds_rounded": int(round(runtime)),
    }


def enrich_record(record, final_container_state=None):
    run_dir = REPO / record["run_dir"]
    metadata = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    log_metrics = parse_training_log(run_dir)
    gate = json.loads((run_dir / "logs" / "resource_gate.json").read_text(encoding="utf-8"))
    if final_container_state is not None:
        record = dict(record)
        record["wall_time_seconds"] = final_container_state["wall_time_seconds_rounded"]
        record["container_exit_code"] = final_container_state["container_exit_code"]
        record["wall_time_source"] = "rounded docker inspect container runtime"
        record.update({key: value for key, value in final_container_state.items() if key != "wall_time_seconds_rounded"})
    else:
        record = dict(record)
        record["wall_time_source"] = "pilot runner wall clock recorded in source result JSON"
    record.update(log_metrics)
    record["model"] = metadata["params"]["model"]
    record["gpus"] = metadata["params"]["gpus"]
    record["num_rollout"] = metadata["params"]["num_rollout"]
    record["K"] = metadata["group_size_K"]
    record["U"] = metadata["batch_groups_U"]
    record["resource_gate_pass"] = bool(gate.get("pass"))
    record["resource_gate_path"] = str((run_dir / "logs" / "resource_gate.json").relative_to(REPO))
    record["audit_event_counts"] = {
        "reward_rows": count_lines(run_dir / "rewards.jsonl"),
        "cas_reject_rows": count_lines(run_dir / "cas_rejects.jsonl"),
        "seal_rows": count_lines(run_dir / "seals.jsonl"),
    }
    record["artifact_presence"] = {
        "meta.json": (run_dir / "meta.json").is_file(),
        "config.json": (run_dir / "config.json").is_file(),
        "train.log": (run_dir / "logs" / "train.log").is_file(),
        "rewards.jsonl": (run_dir / "rewards.jsonl").is_file(),
        "resource_health.jsonl": (run_dir / "logs" / "resource_health.jsonl").is_file(),
        "latest_checkpointed_iteration.txt": (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").is_file(),
        "final_iter_0000029": (run_dir / "checkpoints" / "iter_0000029").is_dir(),
    }
    record["baseline_effective"] = metadata["baseline_effective"]
    record["checkpoint_mode"] = "on" if metadata["params"]["save_interval"] < 1000000 else "off"
    record["container_exit_code"] = int(record["container_exit_code"])
    return record


def load_records():
    source = json.loads(SOURCE_RESULT.read_text(encoding="utf-8"))
    records = [record for record in source["runs"] if record.get("status") == "success"]
    if len(records) != 17:
        raise RuntimeError(f"expected 17 successful source records, found {len(records)}")
    final_key = ("E7", "clean-rewardtxn", 42)
    if final_key not in {(r["experiment"], r["variant"], int(r["seed"])) for r in records}:
        state = inspect_container(FINAL_CONTAINER)
        if state["container_status"] != "exited" or state["container_exit_code"] != 0:
            raise RuntimeError(f"final supplement container is not a successful exit: {state}")
        records.append({
            "experiment": "E7",
            "variant": "clean-rewardtxn",
            "baseline_mode": "b6",
            "exp_id": FINAL_RUN.name,
            "seed": 42,
            "status": "success",
            "wall_time_seconds": state["wall_time_seconds_rounded"],
            "container_exit_code": state["container_exit_code"],
            "reward_concurrency": 64,
            "checkpoint_mode": "on",
            "run_dir": str(FINAL_RUN.relative_to(REPO)),
            "supplemental_completion": True,
        })
        final_state = state
    else:
        final_state = None
    records.sort(key=lambda r: (r["experiment"], int(r["seed"]), r["variant"]))
    enriched = []
    for record in records:
        state = final_state if record["run_dir"] == str(FINAL_RUN.relative_to(REPO)) else None
        enriched.append(enrich_record(record, state))
    keys = {(r["experiment"], r["variant"], int(r["seed"])) for r in enriched}
    if keys != EXPECTED_KEYS:
        raise RuntimeError(f"pilot matrix mismatch; missing={EXPECTED_KEYS - keys}, extra={keys - EXPECTED_KEYS}")
    for record in enriched:
        if record["status"] != "success" or record["container_exit_code"] != 0:
            raise RuntimeError(f"non-success record in final set: {record['exp_id']}")
        if not record["resource_gate_pass"]:
            raise RuntimeError(f"resource gate did not pass for {record['exp_id']}")
    return source, enriched


def comparison(records, experiment, reference_variant, treatment_variant, seed):
    selected = [r for r in records if r["experiment"] == experiment]
    ref = {int(r["seed"]): r for r in selected if r["variant"] == reference_variant}
    trt = {int(r["seed"]): r for r in selected if r["variant"] == treatment_variant}
    seeds = sorted(set(ref) & set(trt))
    if seeds != [17, 29, 42]:
        raise RuntimeError(f"{experiment}: expected paired seeds 17/29/42, got {seeds}")
    # Importing the existing implementation, rather than reimplementing its
    # bootstrap, keeps this report aligned with the preregistered helper.
    sys.path.insert(0, str(SCRIPT_DIR))
    import paper_statistics  # type: ignore

    result = {
        "reference_variant": reference_variant,
        "treatment_variant": treatment_variant,
        "analysis_unit": "paired_seed",
        "paired_seeds": seeds,
        "effect_direction": "treatment-reference; negative time delta is faster",
        "metrics": {},
    }
    for metric, label in [
        ("wall_time_seconds", "wall_time_seconds"),
        ("stable_step_time_p50_seconds", "stable_step_time_p50_seconds"),
        ("stable_step_time_p95_seconds", "stable_step_time_p95_seconds"),
    ]:
        reference = [ref[s][metric] for s in seeds]
        treatment = [trt[s][metric] for s in seeds]
        bootstrap = paper_statistics.paired_bootstrap(
            reference, treatment, iterations=10000, confidence=0.95,
            statistic="median", seed=seed, analysis_unit="paired_seed",
        )
        deltas = [treatment_value - reference_value for reference_value, treatment_value in zip(reference, treatment)]
        relative = [100.0 * delta / reference_value for delta, reference_value in zip(deltas, reference)]
        relative_bootstrap = paper_statistics.paired_bootstrap(
            [0.0] * len(relative), relative, iterations=10000,
            confidence=0.95, statistic="median", seed=seed + 1,
            analysis_unit="paired_seed_relative_percent",
        )
        result["metrics"][label] = {
            "reference_values": [rounded(value, 6) for value in reference],
            "treatment_values": [rounded(value, 6) for value in treatment],
            "paired_deltas": [rounded(value, 6) for value in deltas],
            "paired_relative_percent": [rounded(value, 4) for value in relative],
            "paired_delta_mean": rounded(mean(deltas), 6),
            "paired_delta_median": rounded(median(deltas), 6),
            "paired_bootstrap_95ci": bootstrap,
            "relative_percent_median": rounded(median(relative), 4),
            "relative_percent_bootstrap_95ci": relative_bootstrap,
        }
    return result


def by_variant_summary(records, experiment):
    output = {}
    for variant in sorted({r["variant"] for r in records if r["experiment"] == experiment}):
        selected = [r for r in records if r["experiment"] == experiment and r["variant"] == variant]
        output[variant] = {
            "n_runs": len(selected),
            "seeds": [int(r["seed"]) for r in selected],
            "wall_time_seconds": summary([r["wall_time_seconds"] for r in selected], 4),
            "stable_step_time_p50_seconds": summary([r["stable_step_time_p50_seconds"] for r in selected], 6),
            "stable_step_time_p95_seconds": summary([r["stable_step_time_p95_seconds"] for r in selected], 6),
            "audit_event_totals": {
                key: sum(r["audit_event_counts"][key] for r in selected)
                for key in ("reward_rows", "cas_reject_rows", "seal_rows")
            },
            "audit_event_per_run": {
                key: [r["audit_event_counts"][key] for r in selected]
                for key in ("reward_rows", "cas_reject_rows", "seal_rows")
            },
        }
    return output


def find_accuracy_evidence(records):
    matches = []
    for record in records:
        run_dir = REPO / record["run_dir"]
        for path in [run_dir / "logs" / "train.log", run_dir / "metrics.json", run_dir / "eval.json"]:
            if path.is_file() and "final_eval_accuracy" in path.read_text(encoding="utf-8", errors="replace"):
                matches.append(str(path.relative_to(REPO)))
    return matches


def make_report():
    source, records = load_records()
    comparisons = {
        "E3_b6_vs_b0": comparison(records, "E3", "b0", "b6", 20260831),
        "E6_rewardtxn_vs_group_rm_only": comparison(records, "E6", "group-rm-only", "rewardtxn", 20260832),
        "E7_clean_rewardtxn_vs_clean_oracle": comparison(records, "E7", "clean-oracle", "clean-rewardtxn", 20260833),
    }
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    final_state = inspect_container(FINAL_CONTAINER)
    total_wall = sum(int(r["wall_time_seconds"]) for r in records)
    accuracy_matches = find_accuracy_evidence(records)
    report = {
        "report_type": "RewardTxn P0 GPU pilot final result and statistical summary",
        "status": "pilot_training_complete_formal_accuracy_pending",
        "generated_at": generated,
        "source_result_file": str(SOURCE_RESULT.relative_to(REPO)),
        "pilot_date": source["pilot_date"],
        "pilot_commit": source["pilot_commit"],
        "working_tree_note": "pilot commit is recorded at ca73c31; launch compatibility fix is present in the current uncommitted phase2_run.sh",
        "design": {
            "experiments": {
                "E3": "b0 vs b6, 3 paired seeds",
                "E6": "group-RM only vs RewardTxn, 3 paired seeds, concurrency=32, checkpoint off",
                "E7": "Clean Oracle vs Clean RewardTxn, 3 paired seeds",
            },
            "model": "Qwen2.5-1.5B-Instruct",
            "gpus": "device=4,5,6,7",
            "gpu_count_per_run": 4,
            "steps_per_run": 30,
            "seeds": [17, 29, 42],
            "K": 8,
            "U": 4,
            "stable_step_window": "steps 20-29 inclusive; p50/p95 are descriptive within-run summaries",
        },
        "completion": {
            "planned_runs": 18,
            "successful_runs": len(records),
            "failed_training_runs": 0,
            "all_steps_0_to_29": all(r["step_range"] == [0, 29] for r in records),
            "all_container_exit_codes_zero": all(r["container_exit_code"] == 0 for r in records),
            "all_resource_gates_passed": all(r["resource_gate_pass"] for r in records),
            "total_recorded_wall_seconds": total_wall,
            "total_recorded_gpu_hours": rounded(total_wall * 4.0 / 3600.0, 6),
            "final_supplement_container": final_state,
        },
        "execution_incidents": INCIDENTS,
        "storage": {
            "checkpoint_archive": "/tmp/rewardtxn/pilot_checkpoint_archive_20260831",
            "archived_reason": "moved selected old checkpoint iterations to root-disk storage after /public crossed the 200 GB resource-gate floor; files were copied/verified or moved without deletion",
            "public_free_space_after_completion_gb": 251,
        },
        "runs": records,
        "statistics": {
            "engine": "scripts/paper_statistics.py",
            "bootstrap_iterations": 10000,
            "confidence": 0.95,
            "independent_analysis_unit": "same-seed paired run",
            "note": "Reward rows, CAS rows, seals, optimizer steps, and within-run observations are not independent replicates.",
            "by_experiment_variant": {
                experiment: by_variant_summary(records, experiment)
                for experiment in ("E3", "E6", "E7")
            },
            "paired_comparisons": comparisons,
            "E7_power_precision": {
                "status": "not_estimable_missing_final_eval_accuracy",
                "final_eval_accuracy_evidence": accuracy_matches,
                "pilot_pairs_available": 3,
                "planned_formal_seed_count": 5,
                "proposed_margin_pp": 1.0,
                "frozen_margin_pp": None,
                "pilot_sd_difference_pp": None,
                "planned_90pct_ci_half_width_pp": None,
                "planned_power_at_zero": None,
                "equivalence_claim_enabled": False,
                "reason": "The pilot runner produced no final_eval_accuracy evaluation artifact or evaluation entry point; training step/reward values cannot substitute for the preregistered endpoint.",
            },
        },
        "formal_artifact_scope": {
            "paper_mode": False,
            "formal_validator_expected": "FAIL because pilot directories intentionally do not contain schedule.json/events.jsonl/metrics.json/verdict.json",
            "interpretation": "This report certifies completed pilot training and descriptive statistics, not formal E7 training-equivalence or paper-run validity.",
        },
    }
    FINAL_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    FINAL_MD.write_text(render_markdown(report), encoding="utf-8")
    print(FINAL_JSON)
    print(FINAL_MD)


def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def ci_text(result):
    ci = result["paired_bootstrap_95ci"]
    return f"{fmt(result['paired_delta_median'])} [{fmt(ci['ci_lower'])}, {fmt(ci['ci_upper'])}]"


def rel_ci_text(result):
    ci = result["relative_percent_bootstrap_95ci"]
    return f"{fmt(result['relative_percent_median'], 2)}% [{fmt(ci['ci_lower'], 2)}%, {fmt(ci['ci_upper'], 2)}%]"


def render_markdown(report):
    lines = []
    lines.append("# RewardTxn P0 GPU Pilot 结果与统计汇总")
    lines.append("")
    lines.append(f"> 生成时间：`{report['generated_at']}`  ")
    lines.append("> 状态：**18/18 个 pilot 短运行完成；E7 final_eval_accuracy 尚未提供，因此不启用正式等价性 claim。**")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append("- E3、E6、E7 共 **18/18** 个计划 run 成功完成；每个 run 均观察到 step `0..29`，容器退出码均为 `0`，启动资源门禁均通过。")
    lines.append("- 配置为 Qwen2.5-1.5B-Instruct、4 GPU（宿主卡 `4,5,6,7`）、30 steps、K=8、U=4、seed `17/29/42`。")
    lines.append(f"- 按各 run 记录的 wall time 合计 **{report['completion']['total_recorded_wall_seconds']} 秒**，折算约 **{fmt(report['completion']['total_recorded_gpu_hours'], 3)} GPU-hours**。")
    lines.append("- 统计只把同 seed 的 baseline/RewardTxn 一对 run 作为独立分析单位；10,000 次 paired bootstrap 用于描述性 95% CI。不能把 reward 行、CAS reject、seal 或训练 step 当成额外独立重复。")
    lines.append("- E7 主终点 `final_eval_accuracy` 没有评测产物；因此不计算/不填写 pilot SD、90% CI 半宽、近似功效，也不宣称 TOST 等价通过。")
    lines.append("")
    lines.append("## 2. Pilot 设计")
    lines.append("")
    lines.append("| 实验 | 对比 | seed | 并发 | checkpoint | run 数 |")
    lines.append("|---|---|---:|---:|---|---:|")
    lines.append("| E3 | b0 vs b6 | 17/29/42 | 64 | on | 6 |")
    lines.append("| E6 | group-RM only vs RewardTxn | 17/29/42 | 32 | off | 6 |")
    lines.append("| E7 | Clean Oracle vs Clean RewardTxn | 17/29/42 | 64 | on | 6 |")
    lines.append("")
    lines.append("## 3. 逐 run 结果")
    lines.append("")
    lines.append("`stable step p50/p95` 是每个 run 的 step 20–29 内 `perf/step_time` 描述性统计，单位秒。`CAS reject`/`seal` 是审计事件行数，不是 CPU 秒数。")
    lines.append("")
    lines.append("| 实验 | variant | seed | wall s | stable p50 s | stable p95 s | reward 行 | CAS reject | seal | exit |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for record in report["runs"]:
        c = record["audit_event_counts"]
        lines.append(
            f"| {record['experiment']} | {record['variant']} | {record['seed']} | {record['wall_time_seconds']} | "
            f"{record['stable_step_time_p50_seconds']:.3f} | {record['stable_step_time_p95_seconds']:.3f} | "
            f"{c['reward_rows']} | {c['cas_reject_rows']} | {c['seal_rows']} | {record['container_exit_code']} |"
        )
    lines.append("")
    lines.append("## 4. 统计汇总")
    lines.append("")
    lines.append("所有差值均为 `treatment - reference`；对时间指标，负值代表 treatment 更快。区间是跨 3 个 paired seed 的 percentile paired bootstrap 95% CI。由于只有 3 对，以下结果是 pilot 描述性证据，不是正式 confirmatory 结论。")
    lines.append("")
    lines.append("| 比较 | 指标 | paired median 差值 [95% CI] | 相对差值 median [95% CI] |")
    lines.append("|---|---|---:|---:|")
    labels = [
        ("E3 b6 vs b0", report["statistics"]["paired_comparisons"]["E3_b6_vs_b0"]),
        ("E6 RewardTxn vs group-RM only", report["statistics"]["paired_comparisons"]["E6_rewardtxn_vs_group_rm_only"]),
        ("E7 Clean RewardTxn vs Clean Oracle", report["statistics"]["paired_comparisons"]["E7_clean_rewardtxn_vs_clean_oracle"]),
    ]
    for title, comparison_result in labels:
        for metric, name in [("wall_time_seconds", "wall time (s)"), ("stable_step_time_p50_seconds", "stable step p50 (s)"), ("stable_step_time_p95_seconds", "stable step p95 (s)")]:
            result = comparison_result["metrics"][metric]
            lines.append(f"| {title} | {name} | {ci_text(result)} | {rel_ci_text(result)} |")
    lines.append("")
    lines.append("### 4.1 E3：b6 相对 b0")
    lines.append("")
    e3 = report["statistics"]["by_experiment_variant"]["E3"]
    lines.append(f"- b0 wall time median：`{e3['b0']['wall_time_seconds']['median']:.3f}s`；b6：`{e3['b6']['wall_time_seconds']['median']:.3f}s`。")
    lines.append("- 该 3 对短运行只能说明观测到的时间波动，不能据此替代 prereg 中更大重复矩阵或安全性结论。")
    lines.append("")
    lines.append("### 4.2 E6：RewardTxn 正常路径开销")
    lines.append("")
    e6 = report["statistics"]["paired_comparisons"]["E6_rewardtxn_vs_group_rm_only"]
    e6_step = e6["metrics"]["stable_step_time_p50_seconds"]
    lines.append(f"- RewardTxn 相对 group-RM only 的 stable step p50 相对差值 median 为 **{e6_step['relative_percent_median']:.2f}%**，paired bootstrap 95% CI 为 `[{e6_step['relative_percent_bootstrap_95ci']['ci_lower']:.2f}%, {e6_step['relative_percent_bootstrap_95ci']['ci_upper']:.2f}%]`。")
    lines.append("- 这不是 prereg E6 的最终 5-pair 开销门禁：pilot 只有 3 对，且不是 warmup 20 + measured 100 steps；因此不判定 `<5%` gate。")
    lines.append("- E6 RewardTxn 的 CAS reject/seal 行数是协议审计事件计数，当前 runner 没有记录 verifier CPU-seconds，不能把这些行数表述为 verifier 时间成本。")
    lines.append("")
    lines.append("### 4.3 E7：训练语义 pilot")
    lines.append("")
    e7 = report["statistics"]["paired_comparisons"]["E7_clean_rewardtxn_vs_clean_oracle"]
    e7_step = e7["metrics"]["stable_step_time_p50_seconds"]
    lines.append(f"- Clean RewardTxn vs Clean Oracle 的 stable step p50 观测相对差值 median 为 **{e7_step['relative_percent_median']:.2f}%**；这是性能辅助统计，不是学习等价性证据。")
    lines.append("- `final_eval_accuracy` 缺失，故 `pilot_sd_difference_pp`、固定 n=5 的 90% CI 半宽和近似 power 均为 **n/a**；`prereg/tost_margin.json` 保持 `pending_pilot`，没有事后冻结 margin。")
    lines.append("")
    lines.append("## 5. 完整性、资源与执行事件")
    lines.append("")
    lines.append("- 18 个成功 run 的 `meta.json`、`config.json`、`logs/train.log`、`logs/resource_gate.json`、`logs/resource_health.jsonl` 和 `rewards.jsonl` 均存在；E3/E7 的 checkpoint-on run 保存了最终 `iter_0000029`。")
    lines.append("- 最后补跑 run：`runs/pilot-E7-clean-rewardtxn-1.5B-4gpu-s42-175253/`，Docker 退出码 `0`，最终 checkpoint 写入成功。")
    lines.append("- 首次启动失败和磁盘门禁失败被作为 incident 保留，不计入 18 个成功训练 run：")
    for incident in report["execution_incidents"]:
        lines.append(f"  - `{incident['run_id']}`：{incident['cause']}（证据：`{incident['evidence']}`）。")
    lines.append("- 为越过 `/public` 的 200 GB 资源门禁，部分旧 checkpoint 已归档至 `/tmp/rewardtxn/pilot_checkpoint_archive_20260831`；未将其误计为新 run，也未删除奖励/日志证据。")
    lines.append("")
    lines.append("## 6. 结果范围与未完成项")
    lines.append("")
    lines.append("这份文档完成的是 **P0 pilot 训练结果与统计汇总**，不是 formal paper run 报告。pilot 使用 `RTX_PAPER_MODE=0`，所以 `validate_run_artifacts.py --paper` 所需的 `schedule.json`、`events.jsonl`、`metrics.json`、`resource.jsonl`、`verdict.json` 不属于该 pilot 产物；不能把该 validator 的 FAIL 改写成训练失败，也不能把 pilot 改写成正式 E7 证据。")
    lines.append("")
    lines.append("仍需后续处理：")
    lines.append("1. 补齐从 checkpoint 到固定评测集的 `final_eval_accuracy` 评测入口；")
    lines.append("2. 在正式结果查看前冻结 `tost_margin`、checkpoint 频率及其他 prereg 参数；")
    lines.append("3. 按固定 5 seeds、step 500 的 formal E7 设计执行正式训练与评测；")
    lines.append("4. 只在具备 final accuracy 后计算 E7 TOST、配对 CI、功效/精度与非劣性统计。")
    lines.append("")
    lines.append("## 7. 机器可读证据")
    lines.append("")
    lines.append(f"- 汇总 JSON：`{FINAL_JSON.relative_to(REPO)}`")
    lines.append(f"- 原始 17-run JSON（含磁盘门禁失败尝试）：`{SOURCE_RESULT.relative_to(REPO)}`")
    lines.append("- Pilot 执行器：`scripts/gpu_pilot.sh`")
    lines.append("- 统计实现：`scripts/paper_statistics.py`")
    lines.append("- prereg：`prereg/E3_baseline.json`、`prereg/E6_overhead.json`、`prereg/E7_training.json`、`prereg/tost_margin.json`")
    lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    make_report()
