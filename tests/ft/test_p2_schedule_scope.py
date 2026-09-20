"""X2 CPU scope boundary tests; no GPU or schedule batch."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('scope_driver', HERE / 'p2_schedule_driver.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
compiler = driver.local_module('p2_schedule_cases')


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_boundary(self, boundary, interleave=0):
        case = compiler.compile_case('X2', boundary, interleave)
        path = self.root / case['id']
        result = driver.run_case(case, path, 25)
        self.assertEqual(result['status'], 'passed', result)
        rows = [json.loads(line) for line in (path / 'owner0-operations.jsonl').read_text().splitlines()]
        return case, path, result, rows

    def test_scope_inventory_exact_80_preserves_previous_252(self):
        self.assertEqual(len(driver.STAGE_FOUR), 80)
        self.assertEqual(len(driver.SUPPORTED), 408)
        previous = set(driver.REPRESENTATIVES + driver.STAGE_ONE + driver.STAGE_TWO + driver.STAGE_THREE)
        self.assertEqual(len(previous), 252)
        self.assertTrue(previous <= set(driver.SUPPORTED))
        self.assertEqual(set(driver.STAGE_FOUR) & previous, {'X2.b02.i04', 'X2.b09.i05'})

    def test_real_exit_new_process_epoch_and_later_corruption_owner(self):
        _, path, result, _ = self.run_boundary(4, 9)
        takeover = result['must_reach']['exited_owner_new_epoch_acceptance']
        self.assertEqual(takeover['epoch'], 1)
        launches = json.loads((path / 'launches.json').read_text())
        self.assertEqual(len(launches), 3)
        self.assertTrue(all(item['returncode_before_cleanup'] == 0 for item in launches))
        self.assertEqual(len({item['identity']['pid'] for item in launches}), 3)
        self.assertEqual(json.loads((path / 'state-run/control.json').read_text())['epoch'], 2)

    def test_real_foreign_root_and_sample_mismatch_are_distinct(self):
        case, path, _, rows = self.run_boundary(7)
        self.assertTrue((path / 'foreign-run/control.json').exists())
        broken = [r for r in rows if r['operation'] != 'state.accept_result.foreign']
        with self.assertRaises(driver.ContractFailure):
            driver.check_x2_boundary(path, case, broken)
        case, path, _, rows = self.run_boundary(6)
        altered = copy.deepcopy(rows)
        next(r for r in altered if r['operation'] == 'fixture.submit_scope')['submitted']['sample'] = 'target:0'
        with self.assertRaises(driver.ContractFailure):
            driver.check_x2_boundary(path, case, altered)

    def test_conflicting_duplicate_and_late_old_order(self):
        self.run_boundary(1, 8)
        case, path, _, rows = self.run_boundary(3)
        altered = copy.deepcopy(rows)
        next(r for r in altered if r['operation'] == 'state.accept_result.old_target_first')['sequence'] = 0
        with self.assertRaises(driver.ContractFailure):
            driver.check_x2_boundary(path, case, altered)

    def test_mapping_duplicate_and_inactive_owner_are_rejected(self):
        self.run_boundary(8)
        _, path, result, rows = self.run_boundary(9, 9)
        event = next(r for r in rows if r['operation'] == 'state.commit_generation.closed_owner')
        self.assertIn('inactive', event['reason'])
        self.assertEqual(json.loads((path / 'state-run/control.json').read_text())['epoch'], 1)
        self.assertIn('specific_mutation_audited', result['must_reach'])

    def test_idempotent_and_never_authorized(self):
        self.run_boundary(0)
        self.run_boundary(5)


if __name__ == '__main__':
    unittest.main()
