"""Verify saved isolated probe evidence. Host execution never sends a signal."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('controller_probe', HERE / 'p2_controller_probe.py')
probe = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(probe)
EVIDENCE = None


def read(path):
    return json.loads(path.read_text())


class GuardTests(unittest.TestCase):
    def test_no_private_namespace_proof_cannot_reach_kill_path(self):
        with patch.object(probe.os, 'pidfd_open', create=True), patch.object(probe.signal, 'pidfd_send_signal', create=True), patch.object(probe.Path, 'read_text', side_effect=FileNotFoundError('no isolated launch')):
            with patch.object(probe.subprocess, 'Popen') as launch:
                with self.assertRaises(FileNotFoundError):
                    probe.one_probe(Path('/unused'), 'before-send-crash')
                launch.assert_not_called()

    def test_unknown_mode_is_not_an_entrypoint(self):
        self.assertEqual(set(probe.MODES), {'before-send-crash', 'after-wait-crash', 'before-send-control', 'after-wait-control'})


class IsolatedEvidenceTests(unittest.TestCase):
    def setUp(self):
        if EVIDENCE is None:
            self.skipTest('not executed: supply --evidence from completed isolated CPU supervisor')
        self.root = EVIDENCE
        supervisor = read(self.root / 'supervisor_result.json')
        self.assertTrue(supervisor['cleanup_confirmed'])
        self.assertIsNone(supervisor['failure'])
        self.assertEqual(supervisor['cleanup_errors'], [])
        self.assertFalse(read(self.root / 'inspect_final.json')['State']['Running'])
        self.assertEqual(read(self.root / 'inspect_final.json')['Id'], supervisor['container_id'])

    def modes(self, suffix):
        return [m for m in probe.MODES if m.endswith(suffix)]

    def test_two_real_controller_kills_and_independent_target_liveness(self):
        for mode in self.modes('crash'):
            path = self.root / 'probes' / mode
            cut = read(path / 'cut-witness.json')
            exits = read(path / 'process-exits.json')
            self.assertEqual(exits['controller_wait'], -9)
            self.assertEqual(cut['target_pidfd_readable'], mode.startswith('after-wait'))
            self.assertTrue(exits['target_pidfd_readable_after'])
            self.assertNotEqual(exits['guardian_still_alive']['pid'], exits['controller']['pid'])
            observed = cut['observations']
            self.assertTrue(observed[0]['independently_live'])
            self.assertEqual(observed[0]['actual']['ppid'], exits['controller']['pid'])
            sends = [x for x in observed if x['kind'] == 'signal_returned']
            self.assertEqual(len(sends), int(mode.startswith('after-wait')))
            if sends:
                self.assertEqual(sends[0]['identity'], observed[0]['identity'])
                self.assertEqual(sends[0]['signal'], 9)
                self.assertEqual(observed[-1]['fields']['exit_code'], -9)
            self.assertEqual(sum(r['kind'] == 'fired' for r in cut['records']), 1)
            self.assertFalse(any(r['kind'] == 'observed' for r in cut['records']))

    def test_real_prepare_hash_identity_and_namespace(self):
        for mode in probe.MODES:
            path = self.root / 'probes' / mode
            proof = read(path / 'target-boundary.json')
            cut = read(path / 'cut-witness.json')
            self.assertEqual(hashlib.sha256((path / proof['intent_file']).read_bytes()).hexdigest(), proof['intent_sha256'])
            self.assertEqual(proof['intent_sha256'], cut['target_intent_sha256'])
            registered = cut['observations'][0]['identity']
            self.assertTrue(all(proof['identity'][key] == value for key, value in registered.items()))
            containment = read(path / 'containment.json')
            self.assertNotEqual(containment['guardian']['pid_namespace'], containment['host_pid_namespace'])
            self.assertEqual(proof['identity']['pid_namespace'], containment['guardian']['pid_namespace'])
            self.assertEqual(proof['boundary'], 'real_state_prepare_returned')

    def test_fresh_resume_original_journal_adds_only_uncertain_result(self):
        for mode in self.modes('crash'):
            path = self.root / 'probes' / mode
            before, after = read(path / 'journal-before-resume.json'), read(path / 'journal-after-resume.json')
            self.assertEqual(after['records'][:-1], before['records'])
            result = after['records'][-1]
            self.assertEqual(result['kind'], 'result')
            self.assertEqual(result['classification'], 'technical_invalid')
            self.assertEqual(result['reason'], 'uncertain interrupted attempt; no replay')
            self.assertEqual(result['oracle_status'], 'not_evaluated')
            self.assertEqual(read(path / 'resume-result.json')['result'], result)
            self.assertNotEqual(read(path / 'resume-result.json')['identity']['pid'], read(path / 'process-exits.json')['controller']['pid'])
            self.assertEqual(hashlib.sha256((path / 'run/events.jsonl').read_bytes()).hexdigest(), after['sha256'])
            self.assertEqual(sum(r['kind'] == 'process_registered' for r in after['records']), 1)
            self.assertEqual(sum(r['kind'] == 'fired' for r in after['records']), 1)

    def test_both_uninterrupted_controls_observe_real_target_kill(self):
        for mode in self.modes('control'):
            path = self.root / 'probes' / mode
            self.assertEqual(read(path / 'process-exits.json')['controller_wait'], 0)
            self.assertEqual(read(path / 'controller-result.json')['classification'], 'execution_complete')
            before, after = read(path / 'journal-before-resume.json'), read(path / 'journal-after-resume.json')
            self.assertEqual(before, after)
            observed = [r for r in before['records'] if r['kind'] == 'observed']
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]['exit_code'], -9)

    def test_real_negative_restarts_and_invalid_ready_reject(self):
        for mode in probe.MODES:
            path = self.root / 'probes' / mode
            self.assertIn('immutable', read(path / 'repeat-result.json')['message'])
            self.assertIn('frozen inputs changed', read(path / 'changed-result.json')['message'])
            self.assertEqual(read(path / 'bad-tail/rejection.json')['error'], 'JSONDecodeError')
        for kind in ('nonce', 'pid'):
            path = self.root / 'probes' / ('bad-' + kind)
            self.assertIn('ready identity/nonce mismatch', read(path / 'result.json')['reason'])
            rows = [json.loads(x) for x in (path / 'run/events.jsonl').read_text().splitlines()]
            self.assertFalse(any(r['kind'] in ('fired', 'observed') for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence', type=Path)
    args, rest = parser.parse_known_args()
    EVIDENCE = args.evidence
    unittest.main(argv=[sys.argv[0], *rest])
