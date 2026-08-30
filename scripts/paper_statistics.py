#!/usr/bin/env python3
"""Dependency-free statistical helpers for the paper experiments.

The independent analysis unit is always an independently repeated paired run,
seed, or a preregistered block.  Events, optimizer steps, and samples within a
run are *not* independent observations and must be aggregated before calling
these functions (or kept together by ``paired_block_bootstrap``).

All paired effects use ``treatment - reference``.  The implementation only
uses the Python standard library and is compatible with Python 3.8.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


Number = Union[int, float]


def _numbers(values: Sequence[Number], name: str) -> List[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("%s must not be empty" % name)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("%s must contain only finite numbers" % name)
    return result


def _paired(reference: Sequence[Number], treatment: Sequence[Number]) -> Tuple[List[float], List[float]]:
    ref = _numbers(reference, "reference")
    trt = _numbers(treatment, "treatment")
    if len(ref) != len(trt):
        raise ValueError("reference and treatment must have the same length")
    return ref, trt


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("at least two independent units are required")
    center = _mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))


def _statistic(name: str) -> Callable[[Sequence[float]], float]:
    if name == "mean":
        return _mean
    if name == "median":
        return _median
    raise ValueError("statistic must be 'mean' or 'median'")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_summary(
    observed: float,
    samples: Sequence[float],
    confidence: float,
    iterations: int,
    seed: int,
    analysis_unit: str,
    method: str,
) -> Dict[str, object]:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    alpha = 1.0 - confidence
    return {
        "method": method,
        "analysis_unit": analysis_unit,
        "estimate": observed,
        "confidence": confidence,
        "ci_lower": _quantile(samples, alpha / 2.0),
        "ci_upper": _quantile(samples, 1.0 - alpha / 2.0),
        "iterations": iterations,
        "seed": seed,
    }


def paired_bootstrap(
    reference: Sequence[Number],
    treatment: Sequence[Number],
    iterations: int = 10000,
    confidence: float = 0.95,
    statistic: str = "median",
    seed: int = 20260830,
    analysis_unit: str = "paired_run",
) -> Dict[str, object]:
    """Percentile bootstrap over independent matched pairs.

    Each index must identify one independent run/seed.  The estimate is the
    requested treatment statistic minus the corresponding reference statistic.
    """
    ref, trt = _paired(reference, treatment)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not analysis_unit.strip():
        raise ValueError("analysis_unit must be explicit")
    stat = _statistic(statistic)
    rng = random.Random(seed)
    n = len(ref)
    estimates = []
    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        estimates.append(stat([trt[i] for i in indices]) - stat([ref[i] for i in indices]))
    result = _bootstrap_summary(
        stat(trt) - stat(ref), estimates, confidence, iterations, seed,
        analysis_unit, "paired_percentile_bootstrap",
    )
    result.update({"n_units": n, "statistic": statistic, "effect": "treatment-reference"})
    return result


def paired_block_bootstrap(
    reference: Sequence[Number],
    treatment: Sequence[Number],
    block_ids: Sequence[object],
    iterations: int = 10000,
    confidence: float = 0.95,
    statistic: str = "median",
    seed: int = 20260830,
    analysis_unit: str = "preregistered_block",
) -> Dict[str, object]:
    """Paired bootstrap that resamples whole preregistered blocks.

    All observations sharing a block id remain together.  Blocks, rather than
    the observations inside them, are the independent analysis units.
    """
    ref, trt = _paired(reference, treatment)
    if len(block_ids) != len(ref):
        raise ValueError("block_ids must have the same length as the paired data")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not analysis_unit.strip():
        raise ValueError("analysis_unit must be explicit")
    blocks = []  # type: List[object]
    positions = {}  # type: Dict[object, List[int]]
    for index, block_id in enumerate(block_ids):
        try:
            positions.setdefault(block_id, []).append(index)
        except TypeError:
            raise ValueError("block ids must be hashable")
        if block_id not in blocks:
            blocks.append(block_id)
    if len(blocks) < 2:
        raise ValueError("at least two independent blocks are required")
    stat = _statistic(statistic)
    rng = random.Random(seed)
    estimates = []
    for _ in range(iterations):
        sampled_blocks = [blocks[rng.randrange(len(blocks))] for _ in blocks]
        sampled_indices = [index for block in sampled_blocks for index in positions[block]]
        estimates.append(
            stat([trt[i] for i in sampled_indices]) - stat([ref[i] for i in sampled_indices])
        )
    result = _bootstrap_summary(
        stat(trt) - stat(ref), estimates, confidence, iterations, seed,
        analysis_unit, "paired_block_percentile_bootstrap",
    )
    result.update({
        "n_blocks": len(blocks),
        "n_observations": len(ref),
        "statistic": statistic,
        "effect": "treatment-reference",
    })
    return result


def block_bootstrap(
    values: Sequence[Number],
    block_ids: Sequence[object],
    iterations: int = 10000,
    confidence: float = 0.95,
    statistic: str = "median",
    seed: int = 20260830,
    analysis_unit: str = "preregistered_block",
) -> Dict[str, object]:
    """Unpaired percentile bootstrap that resamples complete blocks."""
    numeric = _numbers(values, "values")
    zeros = [0.0] * len(numeric)
    paired = paired_block_bootstrap(
        zeros, numeric, block_ids, iterations, confidence, statistic, seed, analysis_unit
    )
    paired["method"] = "block_percentile_bootstrap"
    paired.pop("effect", None)
    return paired


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iterations = 200
    epsilon = 3.0e-14
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("incomplete beta did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    """CDF of Student's t distribution (exposed for reproducibility tests)."""
    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    if value == 0.0:
        return 0.5
    x = degrees_of_freedom / (degrees_of_freedom + value * value)
    tail = 0.5 * _regularized_beta(x, degrees_of_freedom / 2.0, 0.5)
    return 1.0 - tail if value > 0.0 else tail


def student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    """Numerical inverse of ``student_t_cdf``."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")
    if probability == 0.5:
        return 0.0
    if probability < 0.5:
        return -student_t_quantile(1.0 - probability, degrees_of_freedom)
    low, high = 0.0, 1.0
    while student_t_cdf(high, degrees_of_freedom) < probability:
        high *= 2.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if student_t_cdf(middle, degrees_of_freedom) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def paired_tost(
    reference: Sequence[Number],
    treatment: Sequence[Number],
    margin: Number,
    alpha: float = 0.05,
    analysis_unit: str = "paired_seed",
) -> Dict[str, object]:
    """Paired two-one-sided t tests for symmetric equivalence bounds."""
    ref, trt = _paired(reference, treatment)
    if len(ref) < 2:
        raise ValueError("TOST requires at least two independent pairs")
    bound = float(margin)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("margin must be a positive finite number")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if not analysis_unit.strip():
        raise ValueError("analysis_unit must be explicit")
    differences = [treatment_value - reference_value for reference_value, treatment_value in zip(ref, trt)]
    estimate = _mean(differences)
    sd = _sample_sd(differences)
    df = len(differences) - 1
    se = sd / math.sqrt(len(differences))
    if se == 0.0:
        p_lower = 0.0 if estimate > -bound else 1.0
        p_upper = 0.0 if estimate < bound else 1.0
        critical = student_t_quantile(1.0 - alpha, df)
        ci_lower = ci_upper = estimate
        t_lower = math.inf if estimate > -bound else -math.inf
        t_upper = -math.inf if estimate < bound else math.inf
    else:
        t_lower = (estimate + bound) / se
        t_upper = (estimate - bound) / se
        p_lower = 1.0 - student_t_cdf(t_lower, df)
        p_upper = student_t_cdf(t_upper, df)
        critical = student_t_quantile(1.0 - alpha, df)
        ci_lower = estimate - critical * se
        ci_upper = estimate + critical * se
    return {
        "method": "paired_tost",
        "analysis_unit": analysis_unit,
        "n_pairs": len(differences),
        "effect": "treatment-reference",
        "estimate": estimate,
        "sd_difference": sd,
        "margin_lower": -bound,
        "margin_upper": bound,
        "alpha_each_one_sided": alpha,
        "ci_level": 1.0 - 2.0 * alpha,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_lower": t_lower,
        "t_upper": t_upper,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_value": max(p_lower, p_upper),
        "equivalent": p_lower < alpha and p_upper < alpha,
    }


def paired_noninferiority(
    reference: Sequence[Number],
    treatment: Sequence[Number],
    margin: Number,
    alpha: float = 0.05,
    higher_is_better: bool = True,
    analysis_unit: str = "paired_seed",
) -> Dict[str, object]:
    """One-sided paired t non-inferiority test.

    ``margin`` is the largest tolerated degradation in the original metric.
    The reported effect is always treatment-reference.
    """
    ref, trt = _paired(reference, treatment)
    if len(ref) < 2:
        raise ValueError("non-inferiority requires at least two independent pairs")
    bound = float(margin)
    if not math.isfinite(bound) or bound <= 0.0:
        raise ValueError("margin must be a positive finite number")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be between 0 and 0.5")
    if not analysis_unit.strip():
        raise ValueError("analysis_unit must be explicit")
    raw_differences = [treatment_value - reference_value for reference_value, treatment_value in zip(ref, trt)]
    tested_differences = raw_differences if higher_is_better else [-value for value in raw_differences]
    estimate = _mean(raw_differences)
    tested_estimate = _mean(tested_differences)
    sd = _sample_sd(tested_differences)
    df = len(tested_differences) - 1
    se = sd / math.sqrt(len(tested_differences))
    if se == 0.0:
        p_value = 0.0 if tested_estimate > -bound else 1.0
        t_statistic = math.inf if tested_estimate > -bound else -math.inf
        tested_lower = tested_estimate
    else:
        t_statistic = (tested_estimate + bound) / se
        p_value = 1.0 - student_t_cdf(t_statistic, df)
        tested_lower = tested_estimate - student_t_quantile(1.0 - alpha, df) * se
    if higher_is_better:
        confidence_bound_name = "lower"
        confidence_bound = tested_lower
        null_boundary = -bound
    else:
        confidence_bound_name = "upper"
        confidence_bound = -tested_lower
        null_boundary = bound
    return {
        "method": "paired_t_noninferiority",
        "analysis_unit": analysis_unit,
        "n_pairs": len(raw_differences),
        "effect": "treatment-reference",
        "estimate": estimate,
        "margin": bound,
        "higher_is_better": higher_is_better,
        "alpha": alpha,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "confidence_level_one_sided": 1.0 - alpha,
        "confidence_bound_name": confidence_bound_name,
        "confidence_bound": confidence_bound,
        "null_boundary": null_boundary,
        "noninferior": p_value < alpha,
    }


def cliffs_delta(first: Sequence[Number], second: Sequence[Number]) -> Dict[str, object]:
    """Unpaired Cliff's delta; positive means values in ``first`` are larger."""
    left = _numbers(first, "first")
    right = _numbers(second, "second")
    greater = 0
    less = 0
    for left_value in left:
        for right_value in right:
            if left_value > right_value:
                greater += 1
            elif left_value < right_value:
                less += 1
    delta = (greater - less) / float(len(left) * len(right))
    magnitude = abs(delta)
    if magnitude < 0.147:
        label = "negligible"
    elif magnitude < 0.33:
        label = "small"
    elif magnitude < 0.474:
        label = "medium"
    else:
        label = "large"
    return {
        "method": "cliffs_delta",
        "effect": "first-second",
        "n_first": len(left),
        "n_second": len(right),
        "delta": delta,
        "magnitude": label,
    }


def wilcoxon_signed_rank(
    reference: Sequence[Number],
    treatment: Sequence[Number],
    analysis_unit: str = "paired_run",
) -> Dict[str, object]:
    """Two-sided Wilcoxon signed-rank test with zero differences discarded."""
    if not analysis_unit.strip():
        raise ValueError("analysis_unit must be explicit")
    ref, trt = _paired(reference, treatment)
    differences = [trt_value - ref_value for ref_value, trt_value in zip(ref, trt)]
    nonzero = [difference for difference in differences if difference != 0.0]
    if not nonzero:
        return {
            "method": "wilcoxon_signed_rank_exact",
            "analysis_unit": analysis_unit,
            "n_pairs": len(differences),
            "n_nonzero": 0,
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "p_value": 1.0,
        }
    ordered = sorted(range(len(nonzero)), key=lambda index: abs(nonzero[index]))
    ranks = [0.0] * len(nonzero)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and abs(nonzero[ordered[end]]) == abs(nonzero[ordered[start]]):
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    w_plus = sum(rank for rank, difference in zip(ranks, nonzero) if difference > 0.0)
    w_minus = sum(rank for rank, difference in zip(ranks, nonzero) if difference < 0.0)
    n = len(nonzero)
    if n <= 100:
        doubled_ranks = [int(round(rank * 2.0)) for rank in ranks]
        counts = [0] * (sum(doubled_ranks) + 1)
        counts[0] = 1
        reachable = 0
        for rank in doubled_ranks:
            for subtotal in range(reachable, -1, -1):
                if counts[subtotal]:
                    counts[subtotal + rank] += counts[subtotal]
            reachable += rank
        observed = int(round(w_plus * 2.0))
        total_assignments = 2 ** n
        lower_probability = sum(counts[:observed + 1]) / float(total_assignments)
        upper_probability = sum(counts[observed:]) / float(total_assignments)
        p_value = min(1.0, 2.0 * min(lower_probability, upper_probability))
        method = "wilcoxon_signed_rank_exact"
    else:
        tie_counts = {}  # type: Dict[float, int]
        for difference in nonzero:
            tie_counts[abs(difference)] = tie_counts.get(abs(difference), 0) + 1
        mean_w = n * (n + 1) / 4.0
        variance_w = (
            n * (n + 1) * (2 * n + 1)
            - sum(count ** 3 - count for count in tie_counts.values())
        ) / 24.0
        if variance_w == 0.0:
            p_value = 1.0
        else:
            z_value = max(0.0, (abs(w_plus - mean_w) - 0.5) / math.sqrt(variance_w))
            p_value = min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z_value / math.sqrt(2.0)))))
        method = "wilcoxon_signed_rank_normal_approximation"
    return {
        "method": method,
        "analysis_unit": analysis_unit,
        "n_pairs": len(differences),
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": min(w_plus, w_minus),
        "p_value": p_value,
    }


def holm_bonferroni(
    p_values: Union[Sequence[Number], Mapping[str, Number]],
    alpha: float = 0.05,
) -> Dict[str, object]:
    """Holm-Bonferroni family-wise correction in original input order."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if isinstance(p_values, Mapping):
        items = [(str(label), float(value)) for label, value in p_values.items()]
    else:
        items = [(str(index), float(value)) for index, value in enumerate(p_values)]
    if not items:
        raise ValueError("p_values must not be empty")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for _, value in items):
        raise ValueError("p-values must be finite and between 0 and 1")
    ordered = sorted(enumerate(items), key=lambda entry: entry[1][1])
    adjusted = [0.0] * len(items)
    running_max = 0.0
    continue_rejecting = True
    rejected = [False] * len(items)
    for rank, (original_index, (_, p_value)) in enumerate(ordered):
        multiplier = len(items) - rank
        running_max = max(running_max, min(1.0, multiplier * p_value))
        adjusted[original_index] = running_max
        if continue_rejecting and p_value <= alpha / multiplier:
            rejected[original_index] = True
        else:
            continue_rejecting = False
    results = []
    for index, (label, p_value) in enumerate(items):
        results.append({
            "label": label,
            "p_value": p_value,
            "adjusted_p_value": adjusted[index],
            "reject": rejected[index],
        })
    return {"method": "holm_bonferroni", "alpha": alpha, "family_size": len(items), "results": results}


def plan_paired_tost_sample_size(
    pilot_differences: Sequence[Number],
    margin: Number,
    candidates: Sequence[int] = (5, 7, 9, 12),
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> Dict[str, object]:
    """Apply the preregistered normal-approximation pilot planning rule.

    This is a planning calculation only.  Pilot observations must never enter
    the formal TOST.  It assumes the true paired difference is zero and uses
    the pilot paired SD to choose the smallest candidate whose approximate TOST
    power reaches ``target_power`` and whose 90% CI half-width is <= margin.
    """
    differences = _numbers(pilot_differences, "pilot_differences")
    if len(differences) < 3:
        raise ValueError("at least three independent pilot pairs are required")
    bound = float(margin)
    if bound <= 0.0 or not math.isfinite(bound):
        raise ValueError("margin must be a positive finite number")
    candidate_values = sorted(set(int(candidate) for candidate in candidates))
    if not candidate_values or candidate_values[0] < 2:
        raise ValueError("candidate sample sizes must be integers >= 2")
    sd = _sample_sd(differences)
    rows = []
    selected = None  # type: Optional[int]
    for n_pairs in candidate_values:
        critical = student_t_quantile(1.0 - alpha, n_pairs - 1)
        if sd == 0.0:
            half_width = 0.0
            approximate_power = 1.0
        else:
            standard_error = sd / math.sqrt(n_pairs)
            half_width = critical * standard_error
            z_acceptance = bound / standard_error - critical
            approximate_power = max(
                0.0,
                min(1.0, 2.0 * (0.5 * (1.0 + math.erf(z_acceptance / math.sqrt(2.0)))) - 1.0),
            )
        acceptable = approximate_power >= target_power and half_width <= bound
        rows.append({
            "n_pairs": n_pairs,
            "approximate_power_at_zero": approximate_power,
            "tost_90pct_ci_half_width": half_width,
            "acceptable": acceptable,
        })
        if acceptable and selected is None:
            selected = n_pairs
    return {
        "method": "paired_tost_pilot_normal_approximation",
        "pilot_n_pairs": len(differences),
        "pilot_sd_difference": sd,
        "margin": bound,
        "alpha_each_one_sided": alpha,
        "target_power": target_power,
        "candidates": rows,
        "selected_n_pairs": selected,
        "decision": "freeze_selected_n" if selected is not None else "do_not_make_equivalence_claim",
    }


def _read_payload(path: Path) -> Dict[str, object]:
    if str(path) == "-":
        return json.load(sys.stdin)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "method",
        choices=["paired-bootstrap", "block-bootstrap", "tost", "noninferiority", "cliffs", "wilcoxon", "holm"],
    )
    parser.add_argument("input", type=Path, help="JSON input file, or - for stdin")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--statistic", choices=["mean", "median"], default="median")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    payload = _read_payload(args.input)
    if args.method == "paired-bootstrap":
        if payload.get("block_ids") is not None:
            result = paired_block_bootstrap(
                payload["reference"], payload["treatment"], payload["block_ids"],
                args.iterations, args.confidence, args.statistic, args.seed,
                str(payload.get("analysis_unit", "preregistered_block")),
            )
        else:
            result = paired_bootstrap(
                payload["reference"], payload["treatment"], args.iterations,
                args.confidence, args.statistic, args.seed,
                str(payload.get("analysis_unit", "paired_run")),
            )
    elif args.method == "block-bootstrap":
        result = block_bootstrap(
            payload["values"], payload["block_ids"], args.iterations,
            args.confidence, args.statistic, args.seed,
            str(payload.get("analysis_unit", "preregistered_block")),
        )
    elif args.method == "tost":
        result = paired_tost(
            payload["reference"], payload["treatment"], payload["margin"], args.alpha,
            str(payload.get("analysis_unit", "paired_seed")),
        )
    elif args.method == "noninferiority":
        result = paired_noninferiority(
            payload["reference"], payload["treatment"], payload["margin"], args.alpha,
            bool(payload.get("higher_is_better", True)),
            str(payload.get("analysis_unit", "paired_seed")),
        )
    elif args.method == "cliffs":
        result = cliffs_delta(payload["first"], payload["second"])
    elif args.method == "wilcoxon":
        result = wilcoxon_signed_rank(
            payload["reference"], payload["treatment"],
            str(payload.get("analysis_unit", "paired_run")),
        )
    else:
        result = holm_bonferroni(payload["p_values"], args.alpha)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
