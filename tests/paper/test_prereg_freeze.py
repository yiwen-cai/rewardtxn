from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from validate_prereg import ROOT_DOCUMENTS, validate_prereg  # noqa: E402


class FrozenPreregTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.prereg = Path(self.temporary.name) / "prereg"
        self.prereg.mkdir()
        for filename in ROOT_DOCUMENTS:
            shutil.copy2(REPO / "prereg" / filename, self.prereg / filename)

    def tearDown(self):
        self.temporary.cleanup()

    def _load(self, filename):
        return json.loads((self.prereg / filename).read_text(encoding="utf-8"))

    def _save(self, filename, value):
        (self.prereg / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def _freeze_valid_fixture(self):
        commit = "a" * 40
        for filename in ROOT_DOCUMENTS:
            document = self._load(filename)
            document["status"] = "frozen"
            document["frozen_at"] = "2026-08-30T12:00:00Z"
            document["freeze_commit"] = commit
            self._save(filename, document)
        e3 = self._load("E3_baseline.json")
        e3["config"]["checkpoint_frequency_steps"] = 10
        e3["parameter_freeze"]["status"] = "frozen"
        e3["parameter_freeze"]["frozen_params"] = {
            "B2": {"incomplete_group_policy": "drop", "retry_scope": "whole-group", "max_retry_attempts": 3},
            "B4": {"reservation_timeout_seconds": 60, "reclaim_policy": "timeout", "consume_ack_policy": "after-consume"},
            "B5": {"checkpoint_frequency_steps": 10, "restart_scope": "whole-step", "checkpoint_durability": "fsync-before-resume"},
        }
        self._save("E3_baseline.json", e3)
        e7 = self._load("E7_training.json")
        pilot = e7["pilot_power_precision"]
        pilot.update({
            "status": "frozen",
            "pilot_sd_difference_pp": 0.2,
            "equivalence_claim_enabled": True,
            "planned_power_at_zero": 0.9,
            "planned_90pct_ci_half_width_pp": 0.5,
        })
        self._save("E7_training.json", e7)
        e8 = self._load("E8_soak.json")
        e8["rto_freeze"].update({
            "status": "frozen",
            "pilot_p99_recovery_seconds": 100.0,
            "frozen_max_recovery_seconds": 200.0,
        })
        self._save("E8_soak.json", e8)
        margin = self._load("tost_margin.json")
        margin["frozen_margin_pp"] = 1.0
        margin["faulted_noninferiority"]["frozen_margin_pp"] = 1.0
        self._save("tost_margin.json", margin)

    def _git(self, *arguments):
        return subprocess.run(
            ["git"] + list(arguments),
            cwd=str(Path(self.temporary.name)),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        ).stdout.strip()

    def _make_real_freeze_history(self):
        self._freeze_valid_fixture()
        for filename in ROOT_DOCUMENTS:
            document = self._load(filename)
            document["status"] = "pending_pilot"
            document["frozen_at"] = None
            document["freeze_commit"] = None
            self._save(filename, document)
        for filename, path in [
            ("E3_baseline.json", ("parameter_freeze", "status")),
            ("E7_training.json", ("pilot_power_precision", "status")),
            ("E8_soak.json", ("rto_freeze", "status")),
        ]:
            document = self._load(filename)
            document[path[0]][path[1]] = "pending_pilot"
            self._save(filename, document)
        self._git("init", "-q")
        self._git("config", "user.name", "RewardTxn Test")
        self._git("config", "user.email", "rewardtxn-test@example.invalid")
        self._git("add", "prereg")
        self._git("commit", "-q", "-m", "anchor prereg content")
        anchor = self._git("rev-parse", "HEAD")
        for filename in ROOT_DOCUMENTS:
            document = self._load(filename)
            document["status"] = "frozen"
            document["frozen_at"] = "2026-08-30T12:00:00Z"
            document["freeze_commit"] = anchor
            self._save(filename, document)
        for filename, path in [
            ("E3_baseline.json", ("parameter_freeze", "status")),
            ("E7_training.json", ("pilot_power_precision", "status")),
            ("E8_soak.json", ("rto_freeze", "status")),
        ]:
            document = self._load(filename)
            document[path[0]][path[1]] = "frozen"
            self._save(filename, document)
        self._git("add", "prereg")
        self._git("commit", "-q", "-m", "stamp prereg metadata")
        self._git("tag", "paper-e0")
        return anchor

    def test_fully_frozen_structural_fixture_passes_without_git_probe(self):
        self._freeze_valid_fixture()
        report = validate_prereg(self.prereg, allow_pending=False, verify_git=False)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["pending_count"], 0)
        self.assertEqual(report["freeze_commit"], "a" * 40)

    def test_mismatched_freeze_commit_is_rejected(self):
        self._freeze_valid_fixture()
        e5 = self._load("E5_replay.json")
        e5["freeze_commit"] = "b" * 40
        self._save("E5_replay.json", e5)
        report = validate_prereg(self.prereg, allow_pending=False, verify_git=False)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("freeze_commit 必须一致" in error for error in report["errors"]))

    def test_missing_numeric_threshold_is_rejected_even_in_development(self):
        e7 = self._load("E7_training.json")
        del e7["config"]["formal_steps"]
        self._save("E7_training.json", e7)
        report = validate_prereg(self.prereg, allow_pending=True, verify_git=False)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("formal_steps" in error for error in report["errors"]))

    def test_result_dependent_screening_rule_is_rejected(self):
        e6 = self._load("E6_overhead.json")
        e6["two_tier_design"]["screening_repetition_rule"]["selection"] = "observed_results"
        self._save("E6_overhead.json", e6)
        report = validate_prereg(self.prereg, allow_pending=True, verify_git=False)
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(any("fixed_predeclared" in error for error in report["errors"]))

    def test_git_contract_avoids_self_reference_and_detects_semantic_mutation(self):
        anchor = self._make_real_freeze_history()
        report = validate_prereg(self.prereg, allow_pending=False, verify_git=True)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["freeze_commit"], anchor)
        self.assertIsNotNone(report["semantic_prereg_sha256"])

        e5 = self._load("E5_replay.json")
        e5["core_workload"]["K"] = 16
        self._save("E5_replay.json", e5)
        mutated = validate_prereg(self.prereg, allow_pending=False, verify_git=True)
        self.assertEqual(mutated["status"], "FAIL")
        self.assertTrue(any("语义内容不一致" in error for error in mutated["errors"]))


if __name__ == "__main__":
    unittest.main()
