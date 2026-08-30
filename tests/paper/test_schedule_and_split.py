import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


BASE = Path(__file__).resolve().parents[2]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, BASE / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScheduleTests(unittest.TestCase):
    def test_e7_seed_is_batch_order_independent(self):
        one = load_module("fault_one", "scripts/fault_schedule_gen.py").gen_e7([29], 500, 8, 4)[0]
        batch = load_module("fault_batch", "scripts/fault_schedule_gen.py").gen_e7([17, 29], 500, 8, 4)[1]
        self.assertEqual(one, batch)

    def test_e8_is_deterministic_real_poisson_schedule(self):
        module = load_module("fault_e8", "scripts/fault_schedule_gen.py")
        left = module.gen_e8(1, 8, 2.0)[0]
        right = module.gen_e8(1, 8, 2.0)[0]
        self.assertEqual(left, right)
        self.assertEqual(left["arrival_model"], "exponential_interarrival")
        self.assertEqual(left["realized_event_count"], len(left["events"]))
        self.assertFalse(module.validate(left))
        offsets = [event["wall_clock_sec"] for event in left["events"]]
        self.assertEqual(offsets, sorted(set(offsets)))


class SplitTests(unittest.TestCase):
    def test_split_hash_binds_source_bytes(self):
        module = load_module("gsm_split", "scripts/gen_gsm8k_split.py")
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "data.jsonl"
            source.write_text("a\nb\nc\nd\n")
            first = module.build_record(source, 42, 1)
            source.write_text("x\nb\nc\nd\n")
            second = module.build_record(source, 42, 1)
        self.assertNotEqual(first["source_sha256"], second["source_sha256"])
        self.assertNotEqual(first["split_hash"], second["split_hash"])


if __name__ == "__main__":
    unittest.main()
