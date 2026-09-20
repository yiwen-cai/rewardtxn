"""Opt-in real torchrun, CPU-only container; preserve raw evidence on failure."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.ft.container_run import supervise

ROOT = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get('FT_TORCHRUN_DOCKER_IMAGE')


@unittest.skipUnless(IMAGE, 'set FT_TORCHRUN_DOCKER_IMAGE; no GPU required')
class TorchrunCPU(unittest.TestCase):
    def test_real_torchrun_inherits_env_and_consumes_event_once(self):
        evidence = os.environ.get('FT_TORCHRUN_EVIDENCE_DIR')
        if evidence:
            Path(evidence).mkdir(parents=True, exist_ok=True)
        context = contextlib.nullcontext(tempfile.mkdtemp(prefix='torchrun-', dir=evidence)) if evidence else tempfile.TemporaryDirectory()
        with context as directory:
            root = Path(directory)
            python = os.environ.get('FT_NAMESPACE_DOCKER_PYTHON', '/opt/.venv/bin/python')
            env = {'PATH': '/opt/.venv/bin:/usr/local/bin:/usr/bin:/bin', 'PYTHONPATH': '/workspace:/workspace/third_party/areal',
                   'USER': 'ft_cpu_probe', 'LOGNAME': 'ft_cpu_probe', 'OMP_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': ''}
            config = {'argv': [python, '/workspace/tests/ft/torchrun_cpu_fixture.py', 'launch'], 'env': env,
                      'schedule': [{'event_id': 'torchrun-cpu-once', 'target': 'trainer', 'waiters': ['trainer'],
                                    'evidence': {'boundary': 'cpu_torchrun_entry'}}],
                      'timeouts': {'run': 120, 'handshake': 5, 'lease': 10}}
            input_path = root / 'input.json'; input_path.write_text(json.dumps(config))
            sources = ['tests/ft/torchrun_cpu_fixture.py', 'third_party/areal/areal/infra/utils/proc.py',
                       'third_party/areal/areal/infra/utils/launcher.py', 'third_party/areal/areal/infra/launcher/local.py']
            (root / 'source-sha256.json').write_text(json.dumps({p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}))
            output = root / 'run'
            result = supervise(input_path, ROOT, output, IMAGE, python)
            diagnosis = {'result': result, 'log': (output / 'launcher.log').read_text()[-5000:]}
            self.assertTrue(result['cleanup_confirmed'], diagnosis)
            self.assertFalse(result['cleanup_errors'], diagnosis)
            self.assertIsNone(result['failure'], diagnosis)
            events = [json.loads(line) for line in (output / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['kind'] == 'launcher_created' for e in events), 1)
            observed = [e for e in events if e['kind'] == 'method_observation']
            self.assertEqual([e.get('launcher_exit_code') for e in observed], [0], diagnosis)
            launched = json.loads((output / 'torchrun-launch.json').read_text())
            self.assertEqual(launched['torchrun'], '/opt/.venv/bin/torchrun')
            self.assertEqual(launched['python3'], '/opt/.venv/bin/python3')
            workers = [json.loads(p.read_text()) for p in output.glob('torchrun-worker-*.json')]
            self.assertEqual(len(workers), 2, diagnosis)
            workers.sort(key=lambda w: int(w['restart_count']))
            self.assertEqual([w['restart_count'] for w in workers], ['0', '1'])
            self.assertEqual([w['assignment']['status'] for w in workers], ['pending', 'already_fired'])
            registrations = [e for e in events if e['kind'] == 'descendant_registered']
            self.assertEqual(len(registrations), 2)
            for worker in workers:
                self.assertEqual(worker['env'], launched['env'])
                self.assertEqual(worker['base_env'], launched['base_env'])
                self.assertEqual((worker['rank'], worker['world_size']), ('0', '1'))
                registration = next(e for e in registrations if e['incarnation'] == worker['incarnation'])
                self.assertEqual(registration['identity'], worker['identity'])
                self.assertIn(worker['torchrun_parent']['identity'], registration['ancestry'])
                self.assertTrue(any('torchrun' in arg for arg in worker['torchrun_parent']['argv']))
            sent = [e for e in events if e['kind'] == 'signal_sent']
            exits = [e for e in events if e['kind'] == 'process_exit_observed']
            self.assertEqual(len(sent), 1); self.assertEqual(len(exits), 1)
            self.assertEqual(sent[0]['incarnation'], workers[0]['incarnation'])
            self.assertEqual(sent[0]['identity'], workers[0]['identity'])
            self.assertEqual(exits[0]['identity'], sent[0]['identity'])
            self.assertEqual(sum(e['kind'] == 'ready' for e in events), 1)
            replacement = json.loads((output / 'torchrun-replacement.json').read_text())
            self.assertTrue(replacement['already_fired_rejected_before_send'])
            self.assertEqual(replacement['incarnation'], workers[1]['incarnation'])
            self.assertFalse(json.loads((output / 'inspect_created.json').read_text())['HostConfig']['DeviceRequests'])


if __name__ == '__main__': unittest.main()
