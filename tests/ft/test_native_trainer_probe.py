"""CPU contracts, never substitutes for a CUDA/native-recovery run."""
import ast
import copy
import json
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.ft.areal_native_trainer_probe import Probe, verify_checkpoint, METADATA, install
from scripts.ft.native_gpu import validate_uuids, PROFILE
from scripts.ft.namespace_run import read_config
from scripts.ft.container_run import check_containment

ROOT = Path(__file__).resolve().parents[2]


class NativeContracts(unittest.TestCase):
    def test_pending_requires_complete_predecessor(self):
        client = Mock(injection={'status': 'pending'})
        probe = Probe(client, Mock())
        probe.updated({'update_successful': 0}, 'skip')
        probe.updated({'update_successful': 1}, 'u1')
        with self.assertRaisesRegex(RuntimeError, 'lacks complete'):
            probe.updated({'update_successful': 1}, 'u2')
        client.ready.assert_not_called()

    def test_once_and_replacement_observes(self):
        client = Mock(injection={'status': 'pending'})
        client.wait_release.side_effect = RuntimeError('simulated process boundary')
        probe = Probe(client, Mock()); probe.saved = {'checkpoint': 'CPU contract only'}
        probe.updated({'update_successful': 1}, 'u1')
        with self.assertRaisesRegex(RuntimeError, 'process boundary'):
            probe.updated({'update_successful': 1}, 'u2')
        client.ready.assert_called_once()
        replacement = Mock(injection={'status': 'already_fired'})
        probe = Probe(replacement, Mock())
        for _ in range(3):
            probe.updated({'update_successful': 1}, 'replacement')
        replacement.ready.assert_not_called()

    def test_files_not_none_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            manifest = lambda p: [{'path': f.name, 'size': f.stat().st_size, 'sha256': 'fixture'} for f in p.iterdir() if f.is_file()]
            with self.assertRaisesRegex(RuntimeError, 'incomplete RecoverInfo'):
                verify_checkpoint(path, path, 0, manifest)
            for name in METADATA:
                (path / name).write_text('{}')
            (path / 'step_info.json').write_text('{"global_step":0}')
            with self.assertRaisesRegex(RuntimeError, 'incomplete DCP'):
                verify_checkpoint(path, path, 0, manifest)
            (path / '.metadata').write_text('fixture')
            (path / 'rank0.distcp').write_text('fixture')
            self.assertTrue(verify_checkpoint(path, path, 0, manifest)['checkpoint'])
            with self.assertRaisesRegex(RuntimeError, 'step mismatch'):
                verify_checkpoint(path, path, 1, manifest)

    def test_real_source_signatures(self):
        for file, name, expected in [
                ('trainer/rl_trainer.py', '_save_recover_checkpoint', ['self','epoch','epoch_step','global_step']),
                ('engine/megatron_engine.py', 'optimizer_step', ['self'])]:
            tree = ast.parse((ROOT / 'third_party/areal/areal' / file).read_text())
            methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
            self.assertEqual(len(methods), 1)
            self.assertEqual([a.arg for a in methods[0].args.args], expected)

    @unittest.skipUnless(os.environ.get('FT_NATIVE_REAL_IMPORT') == '1', 'requires existing AReaL image, CPU only')
    def test_real_runtime_signatures(self):
        from areal import PPOTrainer
        from areal.engine.megatron_engine import MegatronPPOActor
        self.assertEqual(list(inspect.signature(PPOTrainer._save_recover_checkpoint).parameters),
                         ['self', 'epoch', 'epoch_step', 'global_step'])
        self.assertEqual(list(inspect.signature(MegatronPPOActor.optimizer_step).parameters), ['self'])

    def test_wrapper_preserves_result_and_exception(self):
        class Trainer:
            def _save_recover_checkpoint(self, epoch, epoch_step, global_step):
                raise LookupError('native save failure')
        class Actor:
            def optimizer_step(self):
                return result
        result = {'update_successful': 1, 'lr': 1e-6}
        probe = Mock()
        install(probe, Trainer, Actor, Mock())
        self.assertIs(Actor().optimizer_step(), result)
        probe.updated.assert_called_once_with(result, None)
        with self.assertRaisesRegex(LookupError, 'native save failure'):
            Trainer()._save_recover_checkpoint(0, 0, 0)

    def test_profiles_and_config(self):
        config = ROOT / 'docs/experiments/rewardtxn-ft-20260916/native-trainer-control.json'
        with self.assertRaisesRegex(ValueError, 'environment'):
            read_config(config)
        value, digest = read_config(config, PROFILE)
        self.assertEqual(value['timeouts']['run'], 900)
        self.assertEqual(len(digest), 64)
        for values in ([], ['GPU-' + 'a'*36]*4):
            with self.assertRaises(ValueError): validate_uuids(values)
        ids = ['GPU-' + f'{i:08x}-0000-0000-0000-000000000000' for i in range(4)]
        validate_uuids(ids)
        info = {'Config': {'User': '1028:1029'}, 'HostConfig': {'ReadonlyRootfs': True,
                'RestartPolicy': {'Name': 'no'}, 'CapDrop': ['ALL'], 'SecurityOpt': ['no-new-privileges'],
                'DeviceRequests': [{'Count': 0, 'DeviceIDs': ids}]}}
        with self.assertRaises(RuntimeError): check_containment(info)
        check_containment(info, PROFILE, ids)
        bad = copy.deepcopy(info); bad['HostConfig']['DeviceRequests'][0]['Count'] = -1
        with self.assertRaises(RuntimeError): check_containment(bad, PROFILE, ids)


if __name__ == '__main__':
    unittest.main()
