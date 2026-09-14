import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import e7_restart_finish as finish


class RestartPermissions(unittest.TestCase):
    def test_eval_ownership_repaired_before_quality_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/e7restart-test'
            (run / 'logs').mkdir(parents=True)
            spec = {'validation_split': 'validation.json', 'test_split': 'test.json'}
            for stage in ['pilot', 'formal']:
                calls = []

                def call(args, **kwargs):
                    calls.append(args)
                    if 'scripts/e7_diagnosis_eval.py' in args:
                        (run / 'restart_eval').mkdir(exist_ok=True)

                def check_run(*args):
                    self.assertEqual(calls[-1][-4:], ['chown', '-R',
                                     f'{finish.os.getuid()}:{finish.os.getgid()}', '/eval-output'])
                    self.assertIn(f'{run / "restart_eval"}:/eval-output', calls[-1])
                    evaluations = [c for c in calls if 'scripts/e7_diagnosis_eval.py' in c]
                    self.assertEqual(len(evaluations), 1 if stage == 'pilot' else 2)
                    return {'quality_pass': True}

                with patch.object(finish, 'ROOT', root), \
                     patch.object(finish, 'read', side_effect=[spec, {'image': 'test-image'}]), \
                     patch.object(finish, 'call', side_effect=call), \
                     patch.object(finish.subprocess, 'check_output', return_value=json.dumps(
                         [{'State': {'ExitCode': 0, 'OOMKilled': False}}]).encode()), \
                     patch.object(finish, 'check_run', side_effect=check_run), \
                     patch.object(sys, 'argv', ['finish', 'e7restart-test', '--stage', stage, '--gpu', '0']):
                    finish.main()
                self.assertTrue(json.loads((run / 'restart_eval/quality.json').read_text())['quality_pass'])


if __name__ == '__main__':
    unittest.main()
