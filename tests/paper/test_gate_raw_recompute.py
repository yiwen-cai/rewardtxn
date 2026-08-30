import copy
import json
import sys
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
import paper_gates


class GateRawRecomputeTests(unittest.TestCase):
    def test_schedule_mutation_is_rejected(self):
        seed, per_cut = 19, 12
        rows = [paper_gates._expected_schedule_row(seed, cut, trial)
                for cut in paper_gates.CUT_POINTS for trial in range(per_cut)]
        config = {"seed": seed, "per_cut": per_cut, "cuts": paper_gates.CUT_POINTS,
                  "schedule_levels": paper_gates.SCHEDULE_LEVELS,
                  "relevant_dimensions": paper_gates.RELEVANT_DIMENSIONS}
        schedule = {"seed": seed, "trials": rows,
                    "coverage": paper_gates._schedule_coverage(rows)}
        self.assertTrue(paper_gates._validate_schedule(config, schedule)[0])
        mutated = copy.deepcopy(schedule)
        mutated["trials"][0]["dimensions"]["group_size"] = 999
        self.assertFalse(paper_gates._validate_schedule(config, mutated)[0])

    def test_aggregate_only_report_can_never_pass(self):
        forged = {
            "status": "PASS",
            "cutpoints": {cut: {"n": 2500, "failures": 0, "cp_upper": 0.1,
                                  "wilson_upper": 0.1} for cut in paper_gates.CUT_POINTS},
            "total": {"n": 30000, "failures": 999, "rule_of_three_upper": 0.0001},
        }
        verdict = paper_gates.evaluate_trace(forged)
        self.assertEqual(verdict["status"], "FAIL")
        self.assertFalse(verdict["gates"]["G0_self_contained_artifact"])

    def test_reported_totals_are_not_used_for_summary(self):
        forged = {"status": "PASS", "total": {"n": 30000, "failures": 0}}
        verdict = paper_gates.evaluate_trace(copy.deepcopy(forged))
        self.assertEqual(verdict["summary"]["total_n"], 0)
        self.assertEqual(verdict["status"], "FAIL")

    def test_manifest_detects_file_mutation(self):
        # The integration smoke run used by developers is optional; this unit
        # case exercises the verifier with a minimal real file/hash pair.
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            run_dir = Path(root)
            for relative in paper_gates.REQUIRED_FILES:
                path = run_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            entries = []
            for relative in sorted(paper_gates.REQUIRED_FILES):
                path = run_dir / relative
                entries.append({"path": relative, "bytes": path.stat().st_size,
                                "sha256": paper_gates._sha_file(path)})
            (run_dir / "artifact_manifest.json").write_text(json.dumps(
                {"schema_version": "1.0", "files": entries}))
            self.assertTrue(paper_gates._validate_manifest(run_dir)[0])
            (run_dir / "events.jsonl").write_text('{"mutated":true}\n')
            ok, errors = paper_gates._validate_manifest(run_dir)
            self.assertFalse(ok)
            self.assertTrue(any("events.jsonl" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
