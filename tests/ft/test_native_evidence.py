"""Synthetic negative evidence contracts; never native recovery/GPU evidence."""
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from scripts.ft.areal_native_trainer_probe import EVENT, EVIDENCE, METADATA, install_recover_load
from scripts.ft.verify_native_trainer_probe import verify_training_evidence, manifest


def fixture():
    identities = [{'pid': pid, 'ppid': 2, 'pgid': 2, 'start_time': str(pid * 100),
                   'boot_id': 'fixture-boot', 'cgroup': '/fixture', 'pid_namespace': 123, 'nspid': [pid]}
                  for pid in (10, 11)]
    journal, witness = [], []
    for index, status in enumerate(('pending', 'already_fired')):
        incarnation = f'inc-{index}'
        assignment = {'event_id': EVENT, 'status': status, 'event_nonce': 'nonce-fixture'}
        journal.extend([{'kind': 'descendant_registered', 'role': 'trainer', 'incarnation': incarnation,
                         'identity': copy.deepcopy(identities[index]), 'ancestry': [{'pid': 2}]},
                        {'kind': 'injection_assignment', 'role': 'trainer', 'incarnation': incarnation,
                         'injection': copy.deepcopy(assignment)}])
        witness.append({'event': 'registered', 'incarnation': incarnation, 'identity': copy.deepcopy(identities[index]),
                        'pid': identities[index]['pid'], 'assignment': assignment, 'pilot_incarnation': f'pilot-{index}'})
    for kind in ('ready', 'signal_sent', 'process_exit_observed'):
        journal.append({'kind': kind, 'identity': copy.deepcopy(identities[0]), 'incarnation': 'inc-0',
                        'event_id': EVENT, 'event_nonce': 'nonce-fixture', 'signal': 9, 'evidence': EVIDENCE})
    predecessor = {'checkpoint_path': '/output/checkpoint', 'checkpoint': [{'path': '.metadata', 'size': 1, 'sha256': 'fixture'}],
                   'metadata_path': '/output/metadata', 'metadata': [{'path': 'step_info.json', 'size': 1, 'sha256': 'fixture'}]}
    def event(index, name, when, **fields):
        witness.append({'event': name, 'incarnation': f'inc-{index}', 'identity': copy.deepcopy(identities[index]),
                        'pid': identities[index]['pid'], 'time_ns': when, **fields})
    event(0, 'successful_update', 1, ordinal=1, stats={'update_successful': 1})
    event(0, 'complete_save', 2, global_step=0, **copy.deepcopy(predecessor))
    event(0, 'successful_update', 3, ordinal=2, stats={'update_successful': 1})
    event(0, 'ready_witness', 4, predecessor=copy.deepcopy(predecessor), evidence=EVIDENCE)
    event(1, 'recover_info_loaded', 6, last_step_info={'global_step': 0},
          metadata_path=predecessor['metadata_path'], metadata=copy.deepcopy(predecessor['metadata']))
    event(1, 'successful_update', 7, ordinal=1, stats={'update_successful': 1})
    event(1, 'successful_update', 8, ordinal=2, stats={'update_successful': 1})
    event(1, 'complete_save', 9, global_step=2, **copy.deepcopy(predecessor))
    pilot = [{'event': 'checkpoint_load_returned', 'pid': 11, 'process_incarnation': 'pilot-1',
              'starttime': '1100', 'boot_id': 'fixture-boot', 'cgroup': '/fixture\n',
              'path': predecessor['checkpoint_path'], 'files': copy.deepcopy(predecessor['checkpoint']),
              'with_optim': True, 'weight_format': 'dcp', 'wall_time_ns': 5}]
    return journal, witness, pilot


class EvidenceContracts(unittest.TestCase):
    def test_complete_synthetic_contract(self):
        checks = verify_training_evidence(*fixture())
        self.assertTrue(all(checks.values()), checks)

    def test_wrong_signal_or_witness_identity_rejected(self):
        for mutation in ('signal', 'ready', 'witness_identity', 'witness_incarnation', 'pilot_identity', 'pilot_incarnation'):
            with self.subTest(mutation=mutation):
                journal, witness, pilot = fixture()
                if mutation in ('signal', 'ready'):
                    entry = next(e for e in journal if e['kind'] == ('signal_sent' if mutation == 'signal' else 'ready'))
                    entry['incarnation'] = 'inc-1'
                elif mutation == 'witness_identity':
                    next(e for e in witness if e['event'] == 'successful_update')['identity']['start_time'] = 'wrong'
                elif mutation == 'witness_incarnation':
                    next(e for e in witness if e['event'] == 'complete_save')['incarnation'] = 'inc-1'
                elif mutation == 'pilot_identity':
                    pilot[0]['starttime'] = 'wrong'
                else:
                    pilot[0]['process_incarnation'] = 'wrong'
                checks = verify_training_evidence(journal, witness, pilot)
                self.assertFalse(all(checks.values()), checks)

    def test_missing_wrong_or_late_recoverinfo_rejected(self):
        for mutation in ('missing', 'step', 'metadata', 'late'):
            with self.subTest(mutation=mutation):
                journal, witness, pilot = fixture()
                record = next(e for e in witness if e['event'] == 'recover_info_loaded')
                if mutation == 'missing': witness.remove(record)
                elif mutation == 'step': record['last_step_info']['global_step'] = 1
                elif mutation == 'metadata': record['metadata'][0]['sha256'] = 'wrong'
                else: record['time_ns'] = 10
                checks = verify_training_evidence(journal, witness, pilot)
                self.assertFalse(all(checks.values()), checks)

    def test_load_wrapper_preserves_original_result_none_and_exception(self):
        @dataclass
        class Step:
            global_step: int = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in METADATA: (path / name).write_text('{}')
            (path / 'step_info.json').write_text('{"global_step":0}')
            result = SimpleNamespace(last_step_info=Step())
            calls = []
            class Handler:
                config = SimpleNamespace(experiment_name='e', trial_name='t', fileroot='r')
                @staticmethod
                def recover_info_path(*args): return directory
                def load(self, mode):
                    calls.append(mode)
                    if mode == 'error': raise LookupError('original failure')
                    return result if mode == 'loaded' else None
            probe = Mock()
            install_recover_load(probe, Handler, manifest)
            handler = Handler()
            self.assertIs(handler.load('loaded'), result)
            fields = probe.observe.call_args.kwargs
            self.assertEqual(fields['last_step_info']['global_step'], 0)
            self.assertEqual(fields['metadata'], manifest(path))
            self.assertIsNone(handler.load('absent'))
            with self.assertRaisesRegex(LookupError, 'original failure'): handler.load('error')
            self.assertEqual(calls, ['loaded', 'absent', 'error'])
            self.assertEqual(probe.observe.call_count, 2)


if __name__ == '__main__': unittest.main()
