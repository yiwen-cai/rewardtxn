"""Actual subprocess exit receipts, independent of the native launcher."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ft import areal_rewardtxn


class ExitReceipt(unittest.TestCase):
    def test_success_and_failure_are_actual_child_wait_results(self):
        for code in (0, 3):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'phase.json').write_text(json.dumps({'phase': 'first'}))
                worker = root / 'worker.py'
                worker.write_text(f'raise SystemExit({code})\n')
                with patch.object(areal_rewardtxn, '__file__', str(worker)):
                    with self.assertRaises(SystemExit) as stopped:
                        areal_rewardtxn.supervise_trainer(['--config', str(root / 'config.yaml')])
                self.assertEqual(stopped.exception.code, code)
                receipt = json.loads((root / 'first-trainer-exit.json').read_text())
                self.assertEqual(receipt['exit_code'], code)
                self.assertNotEqual(receipt['supervisor_pid'], receipt['trainer_pid'])


if __name__ == '__main__':
    unittest.main()
