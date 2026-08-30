from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from validate_prereg import validate_prereg  # noqa: E402


class PendingPreregTests(unittest.TestCase):
    def test_current_prereg_passes_development_mode_and_lists_pending(self):
        report = validate_prereg(REPO / "prereg", allow_pending=True)
        self.assertEqual(report["status"], "PASS")
        self.assertGreater(report["pending_count"], 0)
        self.assertIn("tost_margin.json:frozen_margin_pp", report["pending"])
        self.assertIn(
            "E8_soak.json:rto_freeze.frozen_max_recovery_seconds", report["pending"]
        )

    def test_current_prereg_is_rejected_by_formal_launch_gate(self):
        report = validate_prereg(REPO / "prereg", allow_pending=False)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("正式实验禁止启动" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
