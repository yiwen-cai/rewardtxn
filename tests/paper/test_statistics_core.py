from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import paper_statistics as stats  # noqa: E402


class DistributionTests(unittest.TestCase):
    def test_student_t_reference_values(self):
        self.assertAlmostEqual(stats.student_t_cdf(1.0, 1), 0.75, places=10)
        self.assertAlmostEqual(stats.student_t_quantile(0.975, 10), 2.2281388519, places=7)

    def test_paired_tost_passes_only_when_both_sides_pass(self):
        passing = stats.paired_tost(
            [80.0, 80.0, 80.0, 80.0, 80.0],
            [80.00, 80.02, 79.98, 80.01, 79.99],
            margin=1.0,
        )
        failing = stats.paired_tost(
            [80.0, 80.0, 80.0, 80.0, 80.0],
            [81.2, 81.1, 81.3, 81.2, 81.1],
            margin=1.0,
        )
        self.assertTrue(passing["equivalent"])
        self.assertLess(passing["ci_lower"], passing["ci_upper"])
        self.assertFalse(failing["equivalent"])

    def test_noninferiority_higher_and_lower_is_better(self):
        accuracy = stats.paired_noninferiority(
            [80.0, 81.0, 79.0, 80.5, 79.5],
            [79.8, 80.9, 79.1, 80.4, 79.6],
            margin=1.0,
            higher_is_better=True,
        )
        latency = stats.paired_noninferiority(
            [10.0, 11.0, 9.0, 10.5, 9.5],
            [10.1, 10.9, 9.1, 10.4, 9.6],
            margin=1.0,
            higher_is_better=False,
        )
        self.assertTrue(accuracy["noninferior"])
        self.assertTrue(latency["noninferior"])
        self.assertEqual(latency["confidence_bound_name"], "upper")

    def test_cliffs_delta_extremes_and_ties(self):
        self.assertEqual(stats.cliffs_delta([3, 4], [1, 2])["delta"], 1.0)
        self.assertEqual(stats.cliffs_delta([1, 1], [1, 1])["delta"], 0.0)
        self.assertEqual(stats.cliffs_delta([1, 2], [3, 4])["delta"], -1.0)

    def test_wilcoxon_exact_and_zero_handling(self):
        result = stats.wilcoxon_signed_rank([0, 0, 0], [1, 2, 3])
        self.assertEqual(result["w_plus"], 6.0)
        self.assertEqual(result["w_minus"], 0.0)
        self.assertAlmostEqual(result["p_value"], 0.25)
        all_zero = stats.wilcoxon_signed_rank([1, 2], [1, 2])
        self.assertEqual(all_zero["p_value"], 1.0)
        self.assertEqual(all_zero["n_nonzero"], 0)

    def test_holm_bonferroni_adjustment_and_step_down_rejection(self):
        result = stats.holm_bonferroni({"a": 0.01, "b": 0.03, "c": 0.04}, alpha=0.05)
        by_label = {row["label"]: row for row in result["results"]}
        self.assertAlmostEqual(by_label["a"]["adjusted_p_value"], 0.03)
        self.assertAlmostEqual(by_label["b"]["adjusted_p_value"], 0.06)
        self.assertAlmostEqual(by_label["c"]["adjusted_p_value"], 0.06)
        self.assertTrue(by_label["a"]["reject"])
        self.assertFalse(by_label["b"]["reject"])
        self.assertFalse(by_label["c"]["reject"])


if __name__ == "__main__":
    unittest.main()
