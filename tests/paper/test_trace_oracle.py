import sys
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / "scripts"))
import trace_oracle


class TraceOracleMutationTests(unittest.TestCase):
    def setUp(self):
        self.authority = {"steps": [{"step": 7, "step_token": "t7", "checkpoint_hash": "h7"}],
                          "rewards": [{"logical_id": "g:0:0", "authoritative_reward": 1.0}]}
        self.events = [
            {"type": "checkpoint", "step": 7, "step_token": "t7", "checkpoint_hash": "h7",
             "payload": {"role": "commit"}},
            {"type": "learner", "step": 7, "payload": {"op": "apply", "step_token": "t7"}},
            {"type": "reward", "logical_id": "g:0:0",
             "payload": {"committed": True, "reward": 1.0}},
        ]

    def kinds(self, events=None):
        return {item["type"] for item in trace_oracle.classify_event_log(events or self.events,
                                                                          self.authority)}

    def test_valid_full_authority(self):
        self.assertEqual(self.kinds(), set())

    def test_unknown_token_and_step_are_rejected(self):
        changed = [dict(event) for event in self.events]
        changed[0] = {**changed[0], "step": 8, "step_token": "unknown"}
        kinds = self.kinds(changed)
        self.assertIn("b_unknown_step", kinds)
        self.assertIn("b_unknown_step_token", kinds)

    def test_missing_and_duplicate_token_step_hash_are_rejected(self):
        missing = [{"type": "checkpoint", "step": 7, "checkpoint_hash": None,
                    "payload": {"role": "commit"}}]
        kinds = self.kinds(missing)
        self.assertIn("b_missing_step_token", kinds)
        self.assertIn("c_missing_checkpoint_hash", kinds)

        duplicate = [self.events[0], dict(self.events[0])]
        kinds = self.kinds(duplicate)
        self.assertIn("b_duplicate_step_token", kinds)
        self.assertIn("b_duplicate_step", kinds)

    def test_hash_and_apply_mutations_are_rejected(self):
        changed = [dict(event) for event in self.events]
        changed[0] = {**changed[0], "checkpoint_hash": "forged"}
        self.assertIn("c_token_hash_mismatch", self.kinds(changed))

        duplicate_apply = self.events + [dict(self.events[1])]
        self.assertIn("b_duplicate_apply", self.kinds(duplicate_apply))

    def test_reward_authority_is_external(self):
        changed = [dict(event) for event in self.events]
        changed[2] = {**changed[2], "payload": {"committed": True, "reward": 0.0,
                                                "authoritative_reward": 0.0}}
        self.assertIn("d_wrong_reward_committed", self.kinds(changed))

    def test_unknown_committed_reward_is_rejected(self):
        changed = [dict(event) for event in self.events]
        changed[2] = {**changed[2], "logical_id": "unknown"}
        self.assertIn("d_unknown_reward_committed", self.kinds(changed))

    def test_missing_and_duplicate_committed_reward_are_rejected(self):
        self.assertIn("d_missing_authoritative_reward", self.kinds(self.events[:2]))
        duplicate = self.events + [dict(self.events[2])]
        self.assertIn("d_duplicate_reward_commit", self.kinds(duplicate))

    def test_cli_range_parser_supports_zero_to_ninety_nine(self):
        self.assertEqual(trace_oracle.parse_authoritative_steps("0..99"), list(range(100)))
        self.assertEqual(trace_oracle.parse_authoritative_steps("3..1"), [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
