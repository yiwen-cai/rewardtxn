import json
import sys
import tempfile
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
import paper_trace_runner as runner
import trace_oracle


class TraceRunnerTests(unittest.TestCase):
    def test_schedule_is_deterministic_stratified_and_auditable(self):
        first = runner.build_schedule(123, 12)
        second = runner.build_schedule(123, 12)
        self.assertEqual(first, second)
        self.assertNotEqual(first, runner.build_schedule(124, 12))
        coverage = runner.schedule_coverage(first)
        for cut, dimensions in runner.RELEVANT_DIMENSIONS.items():
            for dimension in dimensions:
                counts = coverage[cut][dimension]
                self.assertEqual(set(counts), {json.dumps(value, sort_keys=True)
                                               for value in runner.SCHEDULE_LEVELS[dimension]})
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_r2_and_r3_do_not_share_trial_state(self):
        rows = {(row["cut"], row["trial"]): row for row in runner.build_schedule(5, 1)}
        with tempfile.TemporaryDirectory() as root:
            r2_dir = Path(root) / "R2"
            r3_dir = Path(root) / "R3"
            r2_dir.mkdir()
            r3_dir.mkdir()
            r2_events, r2_auth = runner.fixture_r2(rows[("R2", 0)], r2_dir, 17, 1.0)
            r3_events, r3_auth = runner.fixture_r3(rows[("R3", 0)], r3_dir, 17, 1.0)
            self.assertEqual(len([event for event in r2_events if event["type"] == "group"]), 1)
            self.assertEqual(len([event for event in r3_events if event["type"] == "group"]), 1)
            self.assertTrue(trace_oracle.verify_trial("R2", r2_events, r2_auth,
                                                       rows[("R2", 0)]["dimensions"])[0])
            self.assertTrue(trace_oracle.verify_trial("R3", r3_events, r3_auth,
                                                       rows[("R3", 0)]["dimensions"])[0])

    def test_output_replacement_moves_old_pass_out_of_canonical_path(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "formal"
            output.mkdir()
            (output / "verdict.json").write_text('{"status":"PASS"}')
            runner._prepare_output(output)
            self.assertTrue(output.is_dir())
            self.assertFalse((output / "verdict.json").exists())
            superseded = list(Path(root).glob("formal.superseded.*"))
            self.assertEqual(len(superseded), 1)
            self.assertTrue((superseded[0] / "verdict.json").exists())

    def test_formal_rejects_smoke_sample_count(self):
        with self.assertRaises(SystemExit):
            runner._parse_args(["--mode", "formal", "--per-cut", "3"])


if __name__ == "__main__":
    unittest.main()
