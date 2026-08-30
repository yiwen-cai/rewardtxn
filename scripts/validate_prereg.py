#!/usr/bin/env python3
"""Validate the paper preregistration before any formal experiment starts.

Default mode is a hard launch gate: every root ``prereg/*.json`` document must
be frozen at the same commit, every numeric threshold must be concrete, and no
pilot-dependent value may remain.  ``--allow-pending`` is only for development;
it accepts the explicit ``pending_pilot`` state and lists every value that still
has to be frozen.  Structural errors and result-dependent decision rules fail in
both modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BASE = Path(__file__).resolve().parent.parent
DEFAULT_PREREG_DIR = BASE / "prereg"
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

ROOT_DOCUMENTS = {
    "prereg_index.json",
    "E1_E2_correctness.json",
    "E3_baseline.json",
    "E4_ablation.json",
    "E5_replay.json",
    "E6_overhead.json",
    "E7_training.json",
    "E8_soak.json",
    "stop_rules.json",
    "tost_margin.json",
}


def _read_json(path: Path, errors: List[str]) -> Optional[Dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append("%s: JSON 读取失败: %s" % (path.name, exc))
        return None
    if not isinstance(value, dict):
        errors.append("%s: 根节点必须是 JSON object" % path.name)
        return None
    return value


def _get(document: Dict[str, Any], dotted_path: str) -> Any:
    value = document  # type: Any
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(dotted_path)
        value = value[part]
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_number(
    documents: Dict[str, Dict[str, Any]],
    filename: str,
    dotted_path: str,
    errors: List[str],
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> None:
    label = "%s:%s" % (filename, dotted_path)
    try:
        value = _get(documents[filename], dotted_path)
    except KeyError:
        errors.append("%s 缺失数值阈值" % label)
        return
    if not _is_number(value):
        errors.append("%s 必须是数值，不能是范围字符串或 null" % label)
        return
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        errors.append("%s 必须 >= %s" % (label, minimum))
    if maximum is not None and numeric > maximum:
        errors.append("%s 必须 <= %s" % (label, maximum))


def _pending_or_value(
    document: Dict[str, Any],
    filename: str,
    dotted_path: str,
    pending: List[str],
    errors: List[str],
    kind: str = "number",
) -> None:
    label = "%s:%s" % (filename, dotted_path)
    try:
        value = _get(document, dotted_path)
    except KeyError:
        errors.append("%s 缺失 pilot 冻结字段" % label)
        return
    if value is None:
        pending.append(label)
        return
    if kind == "number" and not _is_number(value):
        errors.append("%s 冻结后必须为数值" % label)
    elif kind == "positive_number" and (not _is_number(value) or float(value) <= 0.0):
        errors.append("%s 冻结后必须为正数" % label)
    elif kind == "positive_int" and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        errors.append("%s 冻结后必须为正整数" % label)
    elif kind == "object" and not isinstance(value, dict):
        errors.append("%s 冻结后必须为 object" % label)
    elif kind == "boolean" and not isinstance(value, bool):
        errors.append("%s 冻结后必须为 boolean" % label)


def _collect_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            for text in _collect_strings(child):
                yield text
    elif isinstance(value, list):
        for child in value:
            for text in _collect_strings(child):
                yield text


def _validate_index(
    prereg_dir: Path,
    documents: Dict[str, Dict[str, Any]],
    errors: List[str],
) -> None:
    index = documents["prereg_index.json"]
    experiments = index.get("experiments")
    if not isinstance(experiments, dict):
        errors.append("prereg_index.json:experiments 必须是 object")
        return
    expected = {
        "E1_E2": "prereg/E1_E2_correctness.json",
        "E3": "prereg/E3_baseline.json",
        "E4": "prereg/E4_ablation.json",
        "E5": "prereg/E5_replay.json",
        "E6": "prereg/E6_overhead.json",
        "E7": "prereg/E7_training.json",
        "E8": "prereg/E8_soak.json",
    }
    if experiments != expected:
        errors.append("prereg_index.json:experiments 必须完整且使用冻结的 E1_E2/E3…E8 路径")
    if index.get("tost_margin") != "prereg/tost_margin.json":
        errors.append("prereg_index.json:tost_margin 路径不正确")
    if index.get("stop_rules") != "prereg/stop_rules.json":
        errors.append("prereg_index.json:stop_rules 路径不正确")
    if index.get("freeze_tag") != "paper-e0":
        errors.append("prereg_index.json:freeze_tag 必须预先固定为 paper-e0")
    actual = {path.name for path in prereg_dir.glob("*.json")}
    missing = sorted(ROOT_DOCUMENTS - actual)
    extra = sorted(actual - ROOT_DOCUMENTS)
    if missing:
        errors.append("prereg/ 缺少根预注册 JSON: %s" % ", ".join(missing))
    if extra:
        errors.append("prereg/ 存在未纳入冻结校验的根 JSON: %s" % ", ".join(extra))


def _validate_numeric_contract(documents: Dict[str, Dict[str, Any]], errors: List[str]) -> None:
    required_numbers = [
        ("E1_E2_correctness.json", "sample_sizes.E1_real_process_per_cutpoint", 1, None),
        ("E1_E2_correctness.json", "sample_sizes.E1_interface_level_per_case", 1, None),
        ("E1_E2_correctness.json", "sample_sizes.E2_total_trace_injections", 30000, None),
        ("E1_E2_correctness.json", "sample_sizes.E2_min_per_cutpoint", 1500, None),
        ("E1_E2_correctness.json", "statistics.confidence_level", 0.0, 1.0),
        ("E3_baseline.json", "config.K", 1, None),
        ("E3_baseline.json", "config.U", 1, None),
        ("E3_baseline.json", "config.gpus", 1, None),
        ("E3_baseline.json", "config.short_run_steps", 1, None),
        ("E3_baseline.json", "sample_sizes.reps_per_cell", 1, None),
        ("E3_baseline.json", "sample_sizes.paired_reps", 2, None),
        ("E3_baseline.json", "sample_sizes.paired_steps", 1, None),
        ("E3_baseline.json", "statistics.bootstrap_iterations", 1, None),
        ("E3_baseline.json", "statistics.confidence_level", 0.0, 1.0),
        ("E4_ablation.json", "sample_sizes.deterministic_per_variant_fault", 1, None),
        ("E4_ablation.json", "sample_sizes.real_system_reps_per_cell", 1, None),
        ("E4_ablation.json", "statistics.bootstrap_iterations", 1, None),
        ("E4_ablation.json", "statistics.familywise_alpha", 0.0, 1.0),
        ("E5_replay.json", "core_workload.K", 1, None),
        ("E5_replay.json", "core_workload.U", 1, None),
        ("E5_replay.json", "core_workload.failed_group_fraction", 0.0, 1.0),
        ("E5_replay.json", "core_workload.reps", 2, None),
        ("E5_replay.json", "statistics.bootstrap_iterations", 1, None),
        ("E5_replay.json", "statistics.core_savings_gate_fraction", 0.0, 1.0),
        ("E5_replay.json", "statistics.core_ci_lower_gate_fraction", 0.0, 1.0),
        ("E6_overhead.json", "two_tier_design.core_cell.reward_concurrency", 1, None),
        ("E6_overhead.json", "two_tier_design.core_cell.U", 1, None),
        ("E6_overhead.json", "two_tier_design.core_reps", 2, None),
        ("E6_overhead.json", "two_tier_design.screening_reps", 1, 1),
        ("E6_overhead.json", "two_tier_design.screening_repetition_rule.confirmatory_reps", 2, None),
        ("E6_overhead.json", "two_tier_design.steps_per_run.warmup", 1, None),
        ("E6_overhead.json", "two_tier_design.steps_per_run.measured", 1, None),
        ("E6_overhead.json", "reward_concurrency_definition.core_value", 1, None),
        ("E6_overhead.json", "statistics.bootstrap_iterations", 1, None),
        ("E6_overhead.json", "statistics.core_overhead_ci_upper_gate_fraction", 0.0, 1.0),
        ("E7_training.json", "config.formal_steps", 1, None),
        ("E7_training.json", "config.eval_every_steps", 1, None),
        ("E7_training.json", "equivalence.alpha_each_one_sided", 0.0, 0.5),
        ("E7_training.json", "faulted_comparison.alpha", 0.0, 0.5),
        ("E7_training.json", "faulted_comparison.multiplicity.familywise_alpha", 0.0, 1.0),
        ("E7_training.json", "pilot_power_precision.pilot_pairs_min", 3, None),
        ("E7_training.json", "pilot_power_precision.formal_seed_count", 5, 5),
        ("E7_training.json", "pilot_power_precision.target_power", 0.0, 1.0),
        ("E7_training.json", "statistics.bootstrap_iterations", 1, None),
        ("E8_soak.json", "soak.runs", 1, None),
        ("E8_soak.json", "soak.hours_per_run", 1, None),
        ("E8_soak.json", "rto_freeze.pilot_runs_min", 1, None),
        ("E8_soak.json", "rto_freeze.pilot_quantile", 0.0, 1.0),
        ("E8_soak.json", "rto_freeze.hard_cap_seconds", 1, None),
        ("tost_margin.json", "proposed_margin_pp", 0.0, None),
        ("tost_margin.json", "alpha_each_one_sided", 0.0, 0.5),
        ("tost_margin.json", "faulted_noninferiority.alpha", 0.0, 0.5),
    ]
    for filename, path, minimum, maximum in required_numbers:
        _require_number(documents, filename, path, errors, minimum, maximum)
    strictly_positive = [
        ("E5_replay.json", "core_workload.failed_group_fraction"),
        ("E5_replay.json", "statistics.core_savings_gate_fraction"),
        ("E5_replay.json", "statistics.core_ci_lower_gate_fraction"),
        ("E6_overhead.json", "statistics.core_overhead_ci_upper_gate_fraction"),
        ("tost_margin.json", "proposed_margin_pp"),
    ]
    probabilities = [
        ("E1_E2_correctness.json", "statistics.confidence_level"),
        ("E3_baseline.json", "statistics.confidence_level"),
        ("E4_ablation.json", "statistics.familywise_alpha"),
        ("E5_replay.json", "statistics.confidence_level"),
        ("E6_overhead.json", "statistics.confidence_level"),
        ("E7_training.json", "equivalence.alpha_each_one_sided"),
        ("E7_training.json", "faulted_comparison.alpha"),
        ("E7_training.json", "faulted_comparison.multiplicity.familywise_alpha"),
        ("E7_training.json", "pilot_power_precision.target_power"),
        ("E8_soak.json", "rto_freeze.pilot_quantile"),
        ("tost_margin.json", "alpha_each_one_sided"),
        ("tost_margin.json", "faulted_noninferiority.alpha"),
    ]
    for filename, path in strictly_positive:
        try:
            value = _get(documents[filename], path)
        except KeyError:
            continue
        if _is_number(value) and value <= 0:
            errors.append("%s:%s 必须 > 0" % (filename, path))
    for filename, path in probabilities:
        try:
            value = _get(documents[filename], path)
        except KeyError:
            continue
        if _is_number(value) and not 0.0 < value < 1.0:
            errors.append("%s:%s 必须严格位于 (0, 1)" % (filename, path))
    for filename, path in [
        ("E7_training.json", "equivalence.alpha_each_one_sided"),
        ("E7_training.json", "faulted_comparison.alpha"),
        ("tost_margin.json", "alpha_each_one_sided"),
        ("tost_margin.json", "faulted_noninferiority.alpha"),
    ]:
        try:
            value = _get(documents[filename], path)
        except KeyError:
            continue
        if _is_number(value) and value >= 0.5:
            errors.append("%s:%s 必须 < 0.5" % (filename, path))


def _validate_analysis_units(documents: Dict[str, Dict[str, Any]], errors: List[str]) -> None:
    for filename in [
        "E1_E2_correctness.json", "E3_baseline.json", "E4_ablation.json",
        "E5_replay.json", "E6_overhead.json", "E7_training.json", "E8_soak.json",
    ]:
        try:
            value = _get(documents[filename], "statistics.independent_analysis_unit")
        except KeyError:
            errors.append("%s:statistics.independent_analysis_unit 缺失" % filename)
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append("%s:statistics.independent_analysis_unit 必须明确" % filename)


def _validate_no_result_dependent_freedom(
    documents: Dict[str, Dict[str, Any]], errors: List[str]
) -> None:
    e7 = documents["E7_training.json"]
    if not isinstance(e7.get("config", {}).get("formal_steps"), int):
        errors.append("E7_training.json:config.formal_steps 必须固定为整数")
    if e7.get("config", {}).get("formal_extension", {}).get("allowed") is not False:
        errors.append("E7_training.json:正式 endpoint 不得按资源或结果延长")
    e6_rule = None
    try:
        e6_rule = _get(documents["E6_overhead.json"], "two_tier_design.screening_repetition_rule.selection")
    except KeyError:
        pass
    if e6_rule != "fixed_predeclared":
        errors.append("E6_overhead.json:筛选补重复必须使用 fixed_predeclared 规则")
    if documents["E6_overhead.json"].get("two_tier_design", {}).get("screening_inference") != "descriptive_only":
        errors.append("E6_overhead.json:单次 screening 只能 descriptive_only")
    core = documents["E5_replay.json"].get("core_workload")
    required_core = {
        "model", "gpus", "K", "U", "failed_group_fraction", "reward_cost",
        "fault", "reference", "treatment", "primary_metric", "paired_seeds", "reps",
    }
    if not isinstance(core, dict) or not required_core.issubset(core):
        errors.append("E5_replay.json:core_workload 未完全固定")
    else:
        core_seeds = core.get("paired_seeds")
        if (
            not isinstance(core_seeds, list)
            or core.get("reps") != len(core_seeds)
            or any(not isinstance(seed, int) for seed in core_seeds)
            or len(set(core_seeds)) != core.get("reps")
        ):
            errors.append("E5_replay.json:core_workload reps 必须等于唯一整数 paired_seeds 数")
    e6 = documents["E6_overhead.json"]
    matrix_concurrency = e6.get("matrix", {}).get("reward_concurrency")
    defined_concurrency = e6.get("reward_concurrency_definition", {}).get("values")
    core_concurrency = e6.get("reward_concurrency_definition", {}).get("core_value")
    if (
        not isinstance(defined_concurrency, list)
        or matrix_concurrency != defined_concurrency
        or core_concurrency not in defined_concurrency
    ):
        errors.append("E6_overhead.json:reward concurrency 定义、矩阵与 core_value 必须一致")
    core_cell = e6.get("two_tier_design", {}).get("core_cell", {})
    if core_cell.get("primary_checkpoint_mode") != "off":
        errors.append("E6_overhead.json:core overhead gate 必须固定 primary_checkpoint_mode=off")
    confirmatory_cells = e6.get("two_tier_design", {}).get("screening_repetition_rule", {}).get("scale_confirmatory_cells")
    if not isinstance(confirmatory_cells, list) or len(confirmatory_cells) != 2:
        errors.append("E6_overhead.json:必须预先固定 2 卡和 8 卡 scale confirmatory cells")
    elif {cell.get("gpus") for cell in confirmatory_cells if isinstance(cell, dict)} != {2, 8}:
        errors.append("E6_overhead.json:scale confirmatory cells 必须恰为 2 卡和 8 卡")
    e7_seeds = e7.get("seeds")
    e7_n = e7.get("pilot_power_precision", {}).get("formal_seed_count")
    if (
        not isinstance(e7_seeds, list)
        or any(not isinstance(seed, int) for seed in e7_seeds)
        or len(set(e7_seeds)) != e7_n
    ):
        errors.append("E7_training.json:固定 seeds 数必须等于 formal_seed_count 且唯一")
    pairs = documents["E7_training.json"].get("faulted_comparison", {}).get("confirmatory_pairs")
    if not isinstance(pairs, list) or len(pairs) != 2:
        errors.append("E7_training.json:faulted confirmatory comparison family 必须固定为两个 pair")
    method = documents["E7_training.json"].get("faulted_comparison", {}).get("multiplicity", {}).get("method")
    if method != "Holm-Bonferroni":
        errors.append("E7_training.json:faulted comparison 必须固定 Holm-Bonferroni")
    semantics = documents["E3_baseline.json"].get("baseline_semantics")
    if not isinstance(semantics, dict) or set(semantics) != {"B%d" % index for index in range(7)}:
        errors.append("E3_baseline.json:baseline_semantics 必须完整固定 B0–B6")
    forbidden_phrases = ("仅对接近门禁", "资源允许延长至", "based on observed formal", "if formal results")
    for filename, document in documents.items():
        for text in _collect_strings(document):
            if any(phrase in text for phrase in forbidden_phrases):
                errors.append("%s:存在结果依赖或欠约束规则: %s" % (filename, text))


def _validate_pilot_fields(
    documents: Dict[str, Dict[str, Any]], pending: List[str], errors: List[str]
) -> None:
    pending_specs = [
        ("E3_baseline.json", "config.checkpoint_frequency_steps", "positive_int"),
        ("E3_baseline.json", "parameter_freeze.frozen_params", "object"),
        ("E7_training.json", "pilot_power_precision.pilot_sd_difference_pp", "number"),
        ("E7_training.json", "pilot_power_precision.equivalence_claim_enabled", "boolean"),
        ("E7_training.json", "pilot_power_precision.planned_power_at_zero", "number"),
        ("E7_training.json", "pilot_power_precision.planned_90pct_ci_half_width_pp", "number"),
        ("E8_soak.json", "rto_freeze.pilot_p99_recovery_seconds", "positive_number"),
        ("E8_soak.json", "rto_freeze.frozen_max_recovery_seconds", "positive_number"),
        ("tost_margin.json", "frozen_margin_pp", "positive_number"),
        ("tost_margin.json", "faulted_noninferiority.frozen_margin_pp", "positive_number"),
    ]
    for filename, path, kind in pending_specs:
        _pending_or_value(documents[filename], filename, path, pending, errors, kind)
    frozen_params = documents["E3_baseline.json"].get("parameter_freeze", {}).get("frozen_params")
    required_params = documents["E3_baseline.json"].get("parameter_freeze", {}).get("required_params")
    if isinstance(frozen_params, dict) and isinstance(required_params, dict):
        for baseline, names in required_params.items():
            values = frozen_params.get(baseline)
            if not isinstance(values, dict):
                errors.append("E3_baseline.json:parameter_freeze.frozen_params.%s 缺失" % baseline)
                continue
            for name in names:
                if values.get(name) in (None, ""):
                    errors.append("E3_baseline.json:%s.%s 尚未冻结" % (baseline, name))
        b2_retries = frozen_params.get("B2", {}).get("max_retry_attempts") if isinstance(frozen_params.get("B2"), dict) else None
        b4_timeout = frozen_params.get("B4", {}).get("reservation_timeout_seconds") if isinstance(frozen_params.get("B4"), dict) else None
        b5_frequency = frozen_params.get("B5", {}).get("checkpoint_frequency_steps") if isinstance(frozen_params.get("B5"), dict) else None
        global_frequency = documents["E3_baseline.json"].get("config", {}).get("checkpoint_frequency_steps")
        if not isinstance(b2_retries, int) or isinstance(b2_retries, bool) or b2_retries <= 0:
            errors.append("E3_baseline.json:B2.max_retry_attempts 必须冻结为正整数")
        if not _is_number(b4_timeout) or b4_timeout <= 0:
            errors.append("E3_baseline.json:B4.reservation_timeout_seconds 必须冻结为正数")
        if b5_frequency != global_frequency:
            errors.append("E3_baseline.json:B5 checkpoint_frequency_steps 必须与公平性全局值一致")
    clean_margin = documents["tost_margin.json"].get("frozen_margin_pp")
    faulted_margin = documents["tost_margin.json"].get("faulted_noninferiority", {}).get("frozen_margin_pp")
    if _is_number(clean_margin) and _is_number(faulted_margin) and float(clean_margin) != float(faulted_margin):
        errors.append("tost_margin.json:clean 与 faulted margin 必须相同；当前未提供预注册的独立领域理由字段")
    planned_power = documents["E7_training.json"].get("pilot_power_precision", {}).get("planned_power_at_zero")
    target_power = documents["E7_training.json"].get("pilot_power_precision", {}).get("target_power")
    half_width = documents["E7_training.json"].get("pilot_power_precision", {}).get("planned_90pct_ci_half_width_pp")
    claim_enabled = documents["E7_training.json"].get("pilot_power_precision", {}).get("equivalence_claim_enabled")
    precision_pass = _is_number(half_width) and _is_number(clean_margin) and half_width <= clean_margin
    power_pass = _is_number(planned_power) and _is_number(target_power) and planned_power >= target_power
    if claim_enabled is True and not power_pass:
        errors.append("E7_training.json:启用等价性 claim 时 pilot 计划功效必须达到 target_power")
    if claim_enabled is True and not precision_pass:
        errors.append("E7_training.json:启用等价性 claim 时 pilot 90% CI 半宽必须不大于冻结 margin")
    pilot_sd = documents["E7_training.json"].get("pilot_power_precision", {}).get("pilot_sd_difference_pp")
    if _is_number(pilot_sd) and pilot_sd < 0.0:
        errors.append("E7_training.json:pilot_sd_difference_pp 不得为负")
    if _is_number(planned_power) and not 0.0 <= planned_power <= 1.0:
        errors.append("E7_training.json:planned_power_at_zero 必须位于 [0, 1]")
    if _is_number(half_width) and half_width < 0.0:
        errors.append("E7_training.json:planned_90pct_ci_half_width_pp 不得为负")
    rto = documents["E8_soak.json"].get("rto_freeze", {})
    if _is_number(rto.get("frozen_max_recovery_seconds")) and _is_number(rto.get("hard_cap_seconds")):
        if rto["frozen_max_recovery_seconds"] > rto["hard_cap_seconds"]:
            errors.append("E8_soak.json:冻结 RTO 超过 hard_cap_seconds，正式 soak 必须阻断")
    if _is_number(rto.get("pilot_p99_recovery_seconds")) and _is_number(rto.get("frozen_max_recovery_seconds")):
        expected_rto = math.ceil(max(60.0, 2.0 * rto["pilot_p99_recovery_seconds"]) / 10.0) * 10.0
        if rto["frozen_max_recovery_seconds"] != expected_rto:
            errors.append("E8_soak.json:冻结 RTO 不符合 ceil(max(60, 2*p99)/10)*10 规则")


def _validate_freeze_state(
    documents: Dict[str, Dict[str, Any]], pending: List[str], errors: List[str]
) -> Optional[str]:
    frozen_commits = []
    for filename in sorted(ROOT_DOCUMENTS):
        document = documents[filename]
        status = document.get("status")
        frozen_at = document.get("frozen_at")
        freeze_commit = document.get("freeze_commit")
        if status == "pending_pilot":
            pending.append("%s:status" % filename)
            if frozen_at is not None or freeze_commit is not None:
                errors.append("%s:pending_pilot 状态不得填写部分冻结元数据" % filename)
        elif status == "frozen":
            if not isinstance(frozen_at, str) or not UTC_RE.match(frozen_at):
                errors.append("%s:frozen_at 必须是 UTC YYYY-MM-DDTHH:MM:SSZ" % filename)
            if not isinstance(freeze_commit, str) or not COMMIT_RE.match(freeze_commit):
                errors.append("%s:freeze_commit 必须是 7–40 位小写十六进制 commit" % filename)
            else:
                frozen_commits.append(freeze_commit)
        else:
            errors.append("%s:status 必须是 pending_pilot 或 frozen" % filename)
    unique_commits = sorted(set(frozen_commits))
    if len(unique_commits) > 1:
        errors.append("所有 prereg JSON 的 freeze_commit 必须一致: %s" % ", ".join(unique_commits))
    for filename, nested_path in [
        ("E3_baseline.json", "parameter_freeze.status"),
        ("E7_training.json", "pilot_power_precision.status"),
        ("E8_soak.json", "rto_freeze.status"),
    ]:
        try:
            status = _get(documents[filename], nested_path)
        except KeyError:
            errors.append("%s:%s 缺失" % (filename, nested_path))
            continue
        if status == "pending_pilot":
            pending.append("%s:%s" % (filename, nested_path))
        elif status != "frozen":
            errors.append("%s:%s 必须是 pending_pilot 或 frozen" % (filename, nested_path))
    return unique_commits[0] if len(unique_commits) == 1 else None


def _without_freeze_metadata(value: Any) -> Any:
    """Return semantic prereg content, excluding self-referential stamp fields."""
    ignored = {"status", "frozen_at", "freeze_commit", "freeze_tag", "freeze_contract"}
    if isinstance(value, dict):
        return {
            key: _without_freeze_metadata(child)
            for key, child in value.items()
            if key not in ignored
        }
    if isinstance(value, list):
        return [_without_freeze_metadata(child) for child in value]
    return value


def _semantic_digest(documents: Dict[str, Dict[str, Any]]) -> str:
    payload = {
        filename: _without_freeze_metadata(document)
        for filename, document in sorted(documents.items())
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _git_output(arguments: Sequence[str], cwd: Path) -> Tuple[int, str]:
    process = subprocess.run(
        ["git"] + list(arguments),
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return process.returncode, process.stdout.strip()


def _validate_git_freeze_contract(
    prereg_dir: Path,
    documents: Dict[str, Dict[str, Any]],
    freeze_commit: Optional[str],
    errors: List[str],
) -> Optional[str]:
    if freeze_commit is None:
        return None
    freeze_tag = documents["prereg_index.json"].get("freeze_tag")
    if not isinstance(freeze_tag, str) or not freeze_tag.strip():
        errors.append("prereg_index.json:freeze_tag 必须在正式冻结前固定")
        return None
    code, root_text = _git_output(["rev-parse", "--show-toplevel"], prereg_dir)
    if code != 0:
        errors.append("正式冻结校验必须在 git worktree 内运行")
        return None
    git_root = Path(root_text)
    code, resolved_commit = _git_output(["rev-parse", "--verify", "%s^{commit}" % freeze_commit], git_root)
    if code != 0:
        errors.append("freeze_commit 在当前仓库中不存在: %s" % freeze_commit)
        return None
    code, tag_commit = _git_output(["rev-parse", "--verify", "refs/tags/%s^{commit}" % freeze_tag], git_root)
    if code != 0:
        errors.append("冻结 tag 不存在: %s" % freeze_tag)
        return None
    code, _ = _git_output(["merge-base", "--is-ancestor", resolved_commit, tag_commit], git_root)
    if code != 0:
        errors.append("freeze_commit 必须是 %s 的祖先（或同一 commit）" % freeze_tag)
    anchored = {}  # type: Dict[str, Dict[str, Any]]
    for filename in sorted(ROOT_DOCUMENTS):
        relative = "prereg/%s" % filename
        code, text = _git_output(["show", "%s:%s" % (resolved_commit, relative)], git_root)
        if code != 0:
            errors.append("freeze_commit 缺少 %s" % relative)
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append("freeze_commit 中 %s 不是合法 JSON: %s" % (relative, exc))
            continue
        if not isinstance(value, dict):
            errors.append("freeze_commit 中 %s 根节点不是 object" % relative)
            continue
        anchored[filename] = value
    if len(anchored) == len(ROOT_DOCUMENTS):
        current_digest = _semantic_digest(documents)
        anchored_digest = _semantic_digest(anchored)
        if current_digest != anchored_digest:
            errors.append(
                "freeze_commit 与当前/tag prereg 语义内容不一致: anchored=%s current=%s"
                % (anchored_digest, current_digest)
            )
        return anchored_digest
    return None


def validate_prereg(
    prereg_dir: Path,
    allow_pending: bool = False,
    verify_git: bool = True,
) -> Dict[str, Any]:
    prereg_dir = prereg_dir.resolve()
    errors = []  # type: List[str]
    pending = []  # type: List[str]
    documents = {}  # type: Dict[str, Dict[str, Any]]
    for filename in sorted(ROOT_DOCUMENTS):
        path = prereg_dir / filename
        if not path.is_file():
            errors.append("缺少 %s" % path)
            continue
        document = _read_json(path, errors)
        if document is not None:
            documents[filename] = document
    if len(documents) != len(ROOT_DOCUMENTS):
        return {
            "status": "FAIL",
            "mode": "development_allow_pending" if allow_pending else "formal_launch_gate",
            "errors": errors,
            "pending": pending,
        }
    _validate_index(prereg_dir, documents, errors)
    _validate_numeric_contract(documents, errors)
    _validate_analysis_units(documents, errors)
    _validate_no_result_dependent_freedom(documents, errors)
    _validate_pilot_fields(documents, pending, errors)
    freeze_commit = _validate_freeze_state(documents, pending, errors)
    semantic_digest = None
    if not allow_pending and verify_git:
        semantic_digest = _validate_git_freeze_contract(
            prereg_dir, documents, freeze_commit, errors
        )
    pending = sorted(set(pending))
    if pending and not allow_pending:
        errors.append("仍有 %d 项待 pilot/冻结；正式实验禁止启动" % len(pending))
    passed = not errors
    return {
        "status": "PASS" if passed else "FAIL",
        "mode": "development_allow_pending" if allow_pending else "formal_launch_gate",
        "freeze_commit": freeze_commit,
        "semantic_prereg_sha256": semantic_digest,
        "validated_files": sorted(documents),
        "pending_count": len(pending),
        "pending": pending,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prereg-dir", type=Path, default=DEFAULT_PREREG_DIR)
    parser.add_argument("--allow-pending", action="store_true", help="开发模式；列出而不拒绝 pending_pilot")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    args = parser.parse_args()
    report = validate_prereg(args.prereg_dir, args.allow_pending)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print("status: %s (%s)" % (report["status"], report["mode"]))
        if report.get("freeze_commit"):
            print("freeze_commit: %s" % report["freeze_commit"])
        if report.get("pending"):
            print("待冻结项 (%d):" % len(report["pending"]))
            for item in report["pending"]:
                print("  - %s" % item)
        if report.get("errors"):
            print("错误 (%d):" % len(report["errors"]))
            for item in report["errors"]:
                print("  - %s" % item)
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
