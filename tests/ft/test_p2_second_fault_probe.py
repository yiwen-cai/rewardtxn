"""Independent saved CPU chain/protocol checks; no production validators."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

EVIDENCE = None
GROUPS = ('bootstrap', 'primary', 'second')
COMPONENTS = ('model', 'optimizer_master', 'optimizer_moments', 'optimizer_step', 'scheduler',
              'rng_python', 'rng_numpy', 'rng_torch_cpu', 'rng_device', 'rng_tracker', 'policy')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def expected_inputs():
    return {'scope': 'CPU full-state fixture; no optimizer backend',
        'groups': {group: [{'prompt': [{'role': 'user', 'content': f'Return integer {n + 1}.'}],
            'label': str(n + 1), 'completion': str(n + 1), 'tokens': [100 + n, 200 + i], 'loss_mask': [0, 1],
            'logprobs': [0.0, -0.25], 'policy_version': n} for i in range(8)] for n, group in enumerate(GROUPS)},
        'ranks': ['actor:0', 'actor:1'], 'components': list(COMPONENTS),
        'logical_updates': ['bootstrap-update', 'primary-update', 'second-update'],
        'source_order': [f'{g}:{i}' for g in GROUPS for i in range(8)]}


class StaticContractTests(unittest.TestCase):
    def test_frozen_work_is_two_distinct_k8_targets(self):
        expected = expected_inputs()
        self.assertEqual(len(expected['groups']['primary']), 8)
        self.assertEqual(len(expected['groups']['second']), 8)
        self.assertNotEqual(expected['groups']['primary'], expected['groups']['second'])
        self.assertEqual(len(set(expected['source_order'])), 24)


class IsolatedChainTests(unittest.TestCase):
    def setUp(self):
        if EVIDENCE is None:
            self.skipTest('not executed: provide completed private CPU supervisor evidence')
        self.root = EVIDENCE
        supervisor = read(self.root / 'supervisor_result.json')
        self.assertIsNone(supervisor['failure'])
        self.assertTrue(supervisor['cleanup_confirmed'])
        self.assertEqual(supervisor['cleanup_errors'], [])
        final = read(self.root / 'inspect_final.json')
        self.assertFalse(final['State']['Running'])
        self.assertEqual(final['Id'], supervisor['container_id'])
        self.path = self.root / 'probes/two-faults'
        self.journal = rows(self.path / 'controller/events.jsonl')
        self.work = rows(self.path / 'worker-events.jsonl')
        self.inputs = read(self.path / 'inputs.json')
        self.assertEqual(self.inputs, expected_inputs())

    def test_two_real_exits_distinct_identities_nonces_and_three_attempts(self):
        self.assertEqual(read(self.path / 'controller-result.json')['classification'], 'execution_complete')
        registrations = [r for r in self.journal if r['kind'] == 'process_registered']
        self.assertEqual([r['attempt'] for r in registrations], [0, 1, 2])
        self.assertEqual(len({(r['identity']['pid'], r['identity']['start_time']) for r in registrations}), 3)
        for kind in ('armed', 'fired', 'observed'):
            events = [r for r in self.journal if r['kind'] == kind]
            self.assertEqual([(r['event_id'], r['attempt']) for r in events], [('primary-fault', 0), ('second-fault', 1)])
        observed = [r for r in self.journal if r['kind'] == 'observed']
        self.assertTrue(all(r['exit_code'] == -9 for r in observed))
        self.assertEqual(len({r['event_nonce'] for r in observed}), 2)
        for event in observed:
            self.assertEqual(event['event_nonce'], sha(json.dumps([event['run_nonce'], event['event_id'], event['attempt']]).encode()))
            cut = next(r for r in self.work if r['kind'] == 'manifest_cut' and r['event_id'] == event['event_id'])
            self.assertTrue(cut['token_absent'])
            generation_dir = self.path / 'state/generations' / cut['generation']
            self.assertEqual(cut['manifest_sha256'], sha((generation_dir / 'manifest.json').read_bytes()))
            self.assertEqual(cut['files'], read(generation_dir / 'manifest.json')['files'])
            ready = next(r for r in self.journal if r['kind'] == 'ready' and r['event_id'] == event['event_id'])
            self.assertEqual(ready['evidence']['input_sha256'], sha((self.path / 'inputs.json').read_bytes()))
            self.assertEqual(cut['event_nonce'], event['event_nonce'])
            registered = registrations[event['attempt']]['identity']
            self.assertTrue(all(cut['identity'][key] == value for key, value in registered.items()))
            self.assertLess(cut['monotonic_ns'], event['controller_monotonic_ns'])
        observations = [r for r in self.journal if r['kind'] == 'method_observation']
        self.assertEqual(observations[-1]['exit_codes'], {'target': 0})

    def test_promote_actual_load_precedes_second_work_and_final_load(self):
        selected = [r for r in self.work if r['kind'] == 'recovery_selected']
        self.assertEqual([r['epoch'] for r in selected], [0, 1, 2])
        self.assertEqual([r['decision']['reason'] for r in selected[1:]], ['completed durable candidate'] * 2)
        loads = [r for r in self.work if r['kind'] == 'checkpoint_loaded']
        self.assertEqual([r['expected_group'] for r in loads], list(GROUPS))
        cuts = [r for r in self.work if r['kind'] == 'manifest_cut']
        self.assertEqual(selected[1]['decision']['generation'], cuts[0]['generation'])
        self.assertEqual(selected[2]['decision']['generation'], cuts[1]['generation'])
        second_prepare = next(r for r in self.work if r['kind'] == 'prepared' and r['group'] == 'second')
        first_exit = next(r for r in self.journal if r['kind'] == 'observed' and r['attempt'] == 0)
        self.assertLess(first_exit['controller_monotonic_ns'], loads[1]['monotonic_ns'])
        self.assertLess(loads[1]['monotonic_ns'], second_prepare['monotonic_ns'])
        self.assertLess(second_prepare['monotonic_ns'], cuts[1]['monotonic_ns'])
        owners = [r for r in self.work if r['kind'] == 'owner_acquired']
        for previous, current in zip(owners, owners[1:]):
            self.assertEqual(current['prior_processes'], [{k: previous['identity'][k] for k in ('pid', 'start_time', 'boot_id')}])
            self.assertEqual(current['epoch'], previous['epoch'] + 1)

    def test_independent_final_chain_all_components_payloads_and_cursor_pending(self):
        control = read(self.path / 'state/control.json')
        head, chain = control['head'], []
        while head is not None:
            path = self.path / 'state/generations' / head['generation']
            token, manifest = read(path / 'token.json'), read(path / 'manifest.json')
            self.assertEqual(sha(canonical(token)), head['token_sha256'])
            self.assertEqual(token['manifest_sha256'], sha((path / 'manifest.json').read_bytes()))
            self.assertEqual(token['intent_sha256'], sha((path / 'intent.json').read_bytes()))
            chain.append((path, token, manifest)); head = token['parent']
            self.assertLessEqual(len(chain), 3)
        chain.reverse()
        self.assertEqual(len(chain), 3)
        for n, (path, token, manifest) in enumerate(chain):
            group = GROUPS[n]
            self.assertEqual((token['execution_epoch'], token['commit_epoch']), ((0, 0), (0, 1), (1, 2))[n])
            self.assertEqual(manifest['expected_ranks'], ['actor:0', 'actor:1'])
            self.assertEqual(set(manifest['components']), set(COMPONENTS))
            self.assertEqual(len(manifest['files']), 22)
            for component in COMPONENTS:
                for rank in ('actor:0', 'actor:1'):
                    name = rank.replace(':', '-') + '/' + component + '.bin'
                    expected = canonical({'component': component, 'rank': rank, 'retained_groups': list(GROUPS[:n + 1]),
                        'source_sha256': sha(canonical(self.inputs)), 'cpu_fixture_state': [n + 1] * 128})
                    blob = (path / 'checkpoint' / name).read_bytes()
                    self.assertEqual(blob, expected)
                    self.assertEqual(manifest['files'][name], {'sha256': sha(blob), 'size': len(blob)})
            consumed = [f'{g}:{i}' for g in GROUPS[:n + 1] for i in range(8)]
            pending = [] if n == 2 else [f'{GROUPS[n + 1]}:{i}' for i in range(8)]
            self.assertEqual(manifest['data']['consumed'], consumed)
            self.assertEqual([x['sample'] for x in manifest['data']['pending']], pending)
            self.assertEqual(manifest['data']['drawn'], consumed + pending)
            self.assertEqual(manifest['data']['cursor'], (16, 24, 24)[n])
            self.assertEqual(manifest['data']['source_sha256'], sha(canonical(self.inputs)))
            for entry in manifest['data']['pending']:
                index = int(entry['sample'].split(':')[1])
                raw = self.inputs['groups'][GROUPS[n + 1]][index]
                self.assertEqual(entry['prompt'], {'messages': raw['prompt'], 'label': raw['label']})
                self.assertEqual(entry['prompt_sha256'], sha(canonical(entry['prompt'])))
                self.assertEqual((entry['action'], entry['k']), ('regenerate', 8))
            update = manifest['updates'][0]
            self.assertEqual(update['logical_update_id'], group + '-update')
            samples = update['groups'][0]['samples']
            self.assertEqual([e['sample'] for e in samples], [f'{group}:{i}' for i in range(8)])
            self.assertEqual(update['train_input_sha256'], sha(canonical(samples)))
            for i, entry in enumerate(samples):
                raw = self.inputs['groups'][group][i]
                payload = entry['receipt']['payload']
                self.assertEqual(payload['response_sha256'], sha(canonical(raw)))
                self.assertEqual(payload['reward_sha256'], sha(canonical(1.0)))
                self.assertEqual(payload['tensor_input_sha256'], sha(canonical({'raw': raw, 'reward': 1.0})))
        for load in (r for r in self.work if r['kind'] == 'checkpoint_loaded'):
            manifest = read(self.path / 'state/generations' / load['generation'] / 'manifest.json')
            self.assertEqual({x['path']: {'sha256': x['sha256'], 'size': x['size']} for x in load['reads']}, manifest['files'])

    def test_early_second_and_wrong_target_do_not_fire(self):
        for scenario, reason in (('second-early', 'unknown event'), ('wrong-target', 'ready identity/nonce mismatch')):
            path = self.root / 'probes' / scenario
            self.assertEqual(read(path / 'controller-result.json')['classification'], 'technical_invalid')
            self.assertIn(reason, read(path / 'controller-result.json')['reason'])
            self.assertFalse(any(r['kind'] in ('fired', 'observed') for r in rows(path / 'controller/events.jsonl')))

    def test_incomplete_second_never_promoted_and_live_owner_rejected(self):
        path = self.root / 'probes/second-incomplete'
        journal = rows(path / 'controller/events.jsonl')
        self.assertEqual([(r['event_id'], r['attempt']) for r in journal if r['kind'] == 'fired'], [('primary-fault', 0)])
        work = rows(path / 'worker-events.jsonl')
        second = next(r for r in work if r['kind'] == 'prepared' and r['group'] == 'second')
        folder = path / 'state/generations' / second['generation']
        self.assertFalse((folder / 'token.json').exists())
        self.assertFalse((folder / 'manifest.json').exists())
        self.assertTrue((folder / 'abandoned.json').exists())
        self.assertTrue(any(r['kind'] == 'incomplete_rejected' for r in work))
        recovery = [r for r in work if r['kind'] == 'recovery_selected'][-1]['decision']
        self.assertIn(str(folder / 'intent.json').replace(str(self.root), '/output'), recovery['rollback_intents'])
        self.assertEqual([p['sample'] for p in recovery['pending']], [f'second:{i}' for i in range(8)])
        live = self.root / 'probes/old-owner-live'
        self.assertEqual(read(live / 'rejection.json')['reason'], 'previous registered process still alive')
        self.assertEqual(read(live / 'state/control.json')['epoch'], 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence', type=Path)
    args, rest = parser.parse_known_args(); EVIDENCE = args.evidence
    unittest.main(argv=[sys.argv[0], *rest])
