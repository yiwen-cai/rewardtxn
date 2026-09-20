"""F4 CPU event-model contracts, not natural GPU window tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('f4_driver', HERE / 'p2_schedule_driver.py')
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
compiler = driver.local_module('p2_schedule_cases')


class F4Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_boundary(self, b, i=0):
        case = compiler.compile_case('F4', b, i)
        path = self.root / case['id']
        result = driver.run_case(case, path, 25)
        self.assertEqual(result['status'], 'passed', result)
        rows = [json.loads(line) for line in (path / 'owner0-operations.jsonl').read_text().splitlines()]
        return case, path, result, rows

    def test_408_support_exact_and_retains_330(self):
        previous = set(driver.REPRESENTATIVES + driver.STAGE_ONE + driver.STAGE_TWO + driver.STAGE_THREE + driver.STAGE_FOUR)
        self.assertEqual(len(previous), 330)
        self.assertEqual(len(driver.STAGE_FIVE), 80)
        self.assertEqual(len(driver.SUPPORTED), 408)
        self.assertEqual(previous & set(driver.STAGE_FIVE), {'F4.b00.i00', 'F4.b06.i03'})
        executable = {c['id'] for c in compiler.compile_manifest()['cases'] if c['execution_status'] == 'executable'}
        self.assertEqual(set(driver.SUPPORTED), executable)

    def test_unique_shortfall_and_duplicate_do_not_make_eight(self):
        for b in (2, 3):
            case, path, result, rows = self.run_boundary(b, 8)
            self.assertEqual(result['boundary_checks']['unique_generated'], 7)
            self.assertEqual(result['boundary_checks']['generation_events'], 7 if b == 2 else 8)
            self.assertEqual(result['oracle_model_lane']['primary']['status'], 'technical_invalid')
            missing = json.loads((path / 'oracle-missing-report.json').read_text())
            self.assertEqual(missing['status'], 'unverifiable')
            self.assertTrue(any('missing optimizer_start' in m for m in missing['missing_evidence']))
            self.assertEqual(json.loads((path / 'oracle-invalid-retained-report.json').read_text())['status'], 'invalid_commit')
            if b == 3:
                broken = copy.deepcopy(rows)
                next(e for e in broken if e['operation'] == 'fixture.generate_sample_again')['sample'] = 7
                with self.assertRaises(driver.ContractFailure):
                    driver.check_f4_boundary(path, case, broken)

    def test_missed_window_frozen_before_execution_and_primary_survives_i08(self):
        case, path, result, rows = self.run_boundary(9, 8)
        contract = json.loads((path / 'f4-window-contract.json').read_text())
        self.assertEqual(contract['preclassified'], 'technical_invalid_missed_window')
        self.assertEqual(result['oracle_model_lane']['primary']['status'], 'technical_invalid')
        self.assertEqual(json.loads((path / 'oracle-primary-report.json').read_text())['status'], 'technical_invalid')
        self.assertEqual(json.loads((path / 'oracle-healthy-report.json').read_text())['status'], 'correct_recovered')
        self.assertEqual(json.loads((path / 'oracle-missing-report.json').read_text())['status'], 'unverifiable')
        broken = copy.deepcopy(rows)
        next(e for e in broken if e['operation'] == 'fixture.check_fault_window')['executing_sample'] = 0
        with self.assertRaises(driver.ContractFailure):
            driver.check_f4_boundary(path, case, broken)

    def test_lost_and_retry_are_model_only_with_real_preserved_bytes(self):
        self.run_boundary(7)
        case, path, result, rows = self.run_boundary(8)
        retries = [e for e in rows if e['operation'] == 'fixture.native_retry_model']
        self.assertEqual(retries[0]['before'], retries[0]['after'])
        self.assertIn('no_native_retry', retries[0]['evidence_level'])
        broken = [e for e in rows if e['operation'] != 'fixture.retry_score']
        with self.assertRaises(driver.ContractFailure):
            driver.check_f4_boundary(path, case, broken)
        self.assertEqual(len(json.loads((path / 'launches.json').read_text())), 1)

    def test_scoring_and_before_admission_gap(self):
        for b in (1, 4, 5):
            case, path, _, rows = self.run_boundary(b)
            if b == 1:
                broken = copy.deepcopy(rows)
                next(e for e in broken if e['operation'] == 'fixture.reward_end')['reward'] = 0.0
                with self.assertRaises(driver.ContractFailure):
                    driver.check_f4_boundary(path, case, broken)

    def test_wrong_verifier_and_inflight_nonlegacy_branches(self):
        self.run_boundary(6, 1)
        self.run_boundary(0, 1)


if __name__ == '__main__':
    unittest.main()
