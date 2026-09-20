"""Docker command/ownership contracts with a fake CLI; no GPU execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.ft import container_run
from scripts.ft.native_gpu import PROFILE, validate_config, match_cuda_uuids

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'docs/experiments/rewardtxn-ft-20260916/native-trainer-control.json'
IDS = [f'GPU-{i:08x}-0000-0000-0000-000000000000' for i in range(4)]


class ProfileContracts(unittest.TestCase):
    def test_cuda_uuid_optional_prefix_and_hex_case(self):
        expected = ['GPU-e3a46342-07b6-3c91-ce7f-833f0201a336', *IDS[1:]]
        raw = [value[4:].upper() for value in expected]
        raw[1] = expected[1]
        self.assertEqual(match_cuda_uuids(raw, expected), expected)
        self.assertEqual(match_cuda_uuids(list(reversed(raw)), expected), list(reversed(expected)))
        self.assertEqual(raw[0], expected[0][4:].upper())  # raw caller data is retained

    def test_cuda_uuid_rejects_partial_malformed_duplicate_and_wrong_device(self):
        for value in (IDS[0][:12], IDS[0] + 'extra', ' ' + IDS[0], IDS[0] + '\n',
                      'gpu-' + IDS[0][4:], 'MIG-' + IDS[0][4:], IDS[0].replace('-', ''), None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                match_cuda_uuids([value, *IDS[1:]], IDS)
        with self.assertRaises(ValueError):
            match_cuda_uuids([IDS[0], IDS[0][4:], *IDS[2:]], IDS)
        with self.assertRaises(ValueError):
            match_cuda_uuids(IDS[:3], IDS)
        with self.assertRaises(RuntimeError):
            match_cuda_uuids(['GPU-ffffffff-0000-0000-0000-000000000000', *IDS[1:]], IDS)

    def test_config_cannot_expand_schedule_or_env(self):
        config = json.loads(CONFIG.read_text())
        validate_config(config)
        config['env']['AWS_SECRET_ACCESS_KEY'] = 'not-a-secret-fixture'
        with self.assertRaises(ValueError): validate_config(config)
        config = json.loads(CONFIG.read_text()); config['timeouts']['run'] = 901
        with self.assertRaises(ValueError): validate_config(config)

    def test_full_id_network_cleanup_after_start_failure(self):
        calls = []; state = {}; cid, nid = 'a' * 64, 'b' * 64
        def fake(argv, timeout=10):
            calls.append(argv)
            if argv[:2] == ['network', 'create']:
                state['nonce'] = argv[argv.index('--label') + 1].split('=', 1)[1]
                return nid
            if argv[:2] == ['network', 'inspect']:
                self.assertEqual(argv[2], nid)
                return json.dumps([{'Id': nid, 'Internal': True, 'Labels': {'rewardtxn.ft.nonce': state['nonce']}}])
            if argv[0] == 'create':
                state['create'] = argv
                Path(argv[argv.index('--cidfile') + 1]).write_text(cid)
                return cid
            if argv[0] == 'inspect':
                self.assertEqual(argv[1], cid)
                return json.dumps([{'Id': cid, 'Config': {'User': '1028:1029', 'Labels': {'rewardtxn.ft.nonce': state['nonce']}},
                    'HostConfig': {'ReadonlyRootfs': True, 'RestartPolicy': {'Name': 'no'}, 'CapDrop': ['ALL'],
                        'SecurityOpt': ['no-new-privileges'], 'DeviceRequests': [{'Count': 0, 'DeviceIDs': IDS}]},
                    'State': {'Running': False, 'ExitCode': 0}}])
            if argv[0] == 'start':
                raise RuntimeError('injected Docker start failure')
            if argv[0] == 'logs': return 'fixture has no CUDA or controller'
            if argv[0] == 'rm': self.assertEqual(argv[1], cid); return cid
            if argv[:2] == ['network', 'rm']: self.assertEqual(argv[2], nid); return nid
            raise AssertionError(argv)
        with tempfile.TemporaryDirectory() as directory, patch.object(container_run, 'docker', fake), patch(
                'scripts.ft.native_gpu.idle_snapshot') as idle, patch.object(container_run.os, 'getuid', return_value=1028):
            output = Path(directory) / 'run'
            result = container_run.supervise(CONFIG, ROOT, output, 'fixed-image', '/opt/.venv/bin/python',
                                             profile=PROFILE, gpu_uuids=IDS)
            self.assertIn('start failure', result['failure'])
            self.assertTrue(result['cleanup_confirmed'])
            self.assertEqual(result['cleanup_errors'], [])
            self.assertEqual(idle.call_count, 2)
            self.assertTrue((output / 'areal').is_dir())
            self.assertTrue((output / 'name_resolve').is_dir())
            self.assertTrue(json.loads((output / 'network-cleanup.json').read_text())['removed'])
            argv = state['create']
            self.assertEqual(argv[argv.index('--gpus') + 1], '"device=' + ','.join(IDS) + '"')
            for option in ('--memory=128g', '--cpus=32', '--shm-size=16g', '--pids-limit=4096', '--network=' + nid):
                self.assertIn(option, argv)
            self.assertNotIn('--privileged', argv)
            self.assertIn('--read-only', argv)


if __name__ == '__main__': unittest.main()
