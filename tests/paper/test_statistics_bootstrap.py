from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import paper_statistics as stats  # noqa: E402


class BootstrapTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_pairs(self):
        result = stats.paired_bootstrap(
            [1, 2, 3, 4], [2, 3, 4, 5], iterations=500, statistic="mean", seed=7
        )
        self.assertEqual(result["estimate"], 1.0)
        self.assertEqual(result["ci_lower"], 1.0)
        self.assertEqual(result["ci_upper"], 1.0)
        self.assertEqual(result["analysis_unit"], "paired_run")
        self.assertEqual(result["effect"], "treatment-reference")

    def test_block_bootstrap_resamples_whole_blocks(self):
        result = stats.paired_block_bootstrap(
            [0, 0, 10, 10],
            [1, 1, 13, 13],
            ["seed-a", "seed-a", "seed-b", "seed-b"],
            iterations=500,
            statistic="mean",
            seed=11,
        )
        self.assertEqual(result["n_blocks"], 2)
        self.assertEqual(result["n_observations"], 4)
        self.assertEqual(result["estimate"], 2.0)
        self.assertGreaterEqual(result["ci_lower"], 1.0)
        self.assertLessEqual(result["ci_upper"], 3.0)

    def test_bootstrap_rejects_pseudoreplication_shape_errors(self):
        with self.assertRaises(ValueError):
            stats.paired_bootstrap([1, 2], [1])
        with self.assertRaises(ValueError):
            stats.paired_block_bootstrap([1, 2], [2, 3], ["one", "one"])
        with self.assertRaises(ValueError):
            stats.paired_bootstrap([1], [2], analysis_unit="")

    def test_pilot_planner_uses_only_paired_differences(self):
        result = stats.plan_paired_tost_sample_size(
            [-0.1, 0.0, 0.1], margin=1.0, candidates=[5], target_power=0.8
        )
        self.assertEqual(result["pilot_n_pairs"], 3)
        self.assertEqual(result["selected_n_pairs"], 5)
        self.assertEqual(result["decision"], "freeze_selected_n")
        with self.assertRaises(ValueError):
            stats.plan_paired_tost_sample_size([0.0, 0.1], margin=1.0)


if __name__ == "__main__":
    unittest.main()
