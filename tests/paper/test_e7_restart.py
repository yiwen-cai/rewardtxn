import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import e7_restart_checks as checks


class RestartChecks(unittest.TestCase):
    def setUp(self):
        self.spec = checks.read(checks.SPEC)

    def records(self, differences, seeds=None):
        seeds = seeds or self.spec['formal_seeds']
        return [r for s, d in zip(seeds, differences) for r in
                [{'seed': s, 'group': 'group_rm', 'accuracy_pp': 70, 'quality_pass': True},
                 {'seed': s, 'group': 'b6', 'accuracy_pp': 70 + d, 'quality_pass': True}]]

    def test_paired_direction_units_and_order(self):
        records = self.records([.1, -.1, .2, -.2, 0])
        result = checks.paired_report(list(reversed(records)), self.spec['formal_seeds'], self.spec)
        self.assertAlmostEqual(result['tost']['estimate'], 0)
        self.assertTrue(result['tost']['equivalent'])
        self.assertEqual(result['tost']['n_pairs'], 5)
        self.assertEqual(result['unit'], 'pp')
        # A numerical TOST alone must not enable the final claim before pilot gating.
        self.assertFalse(result['equivalence_claim_enabled'])

    def test_large_effect_fails(self):
        result = checks.paired_report(self.records([2, 2.1, 2.2, 2.3, 2.4]), self.spec['formal_seeds'], self.spec)
        self.assertFalse(result['tost']['equivalent'])
        self.assertAlmostEqual(result['tost']['estimate'], 2.2)

    def test_missing_duplicate_and_nan_rejected(self):
        records = self.records([0, .1, .2, .3, .4])
        for invalid in [records[:-1], records + [records[0]]]:
            with self.assertRaises(ValueError):
                checks.paired_report(invalid, self.spec['formal_seeds'], self.spec)
        records[0]['accuracy_pp'] = float('nan')
        with self.assertRaises(ValueError):
            checks.paired_report(records, self.spec['formal_seeds'], self.spec)

    def test_zero_variance_never_enables_claim(self):
        for difference in [0, 1, 2]:
            r = checks.paired_report(self.records([difference] * 5), self.spec['formal_seeds'], self.spec)
            self.assertFalse(r['equivalence_claim_enabled'])
            self.assertFalse(r['nondegenerate'])

    def test_pilot_fixed_n_and_quality(self):
        seeds = self.spec['pilot_seeds']
        records = self.records([-.1, 0, .1], seeds)
        r = checks.paired_report(records, seeds, self.spec, pilot=True)
        self.assertTrue(r['equivalence_claim_enabled'])
        self.assertEqual([x['n_pairs'] for x in r['planning']['candidates']], [5])
        records[0]['quality_pass'] = False
        self.assertFalse(checks.paired_report(records, seeds, self.spec, pilot=True)['equivalence_claim_enabled'])
        r = checks.paired_report(self.records([-10, 0, 10], seeds), seeds, self.spec, pilot=True)
        self.assertFalse(r['equivalence_claim_enabled'])
        r = checks.paired_report(self.records([0, 0, 0], seeds), seeds, self.spec, pilot=True)
        self.assertFalse(r['equivalence_claim_enabled'])

    def test_actual_data_isolated(self):
        self.assertEqual(checks.check_data(self.spec)['train_count'], 6373)

    def test_actual_eval_tampering_detected(self):
        path = ROOT / self.spec['base_validation']
        split = ROOT / self.spec['validation_split']
        checks.check_eval(path, split)
        original = checks.read
        for field, value in [('n_correct', -1), ('truncated_fraction', .5), ('split_sha256', 'bad')]:
            damaged = copy.deepcopy(original(path))
            damaged[field] = value
            with patch.object(checks, 'read', side_effect=lambda p: damaged if Path(p) == path else original(p)):
                with self.assertRaises(ValueError):
                    checks.check_eval(path, split)

    def test_formal_preflight_refuses_missing_pilot_without_docker(self):
        with patch.object(checks, 'check_data'), patch.object(checks, 'manifest_report', side_effect=FileNotFoundError('pilot')), patch.object(checks.subprocess, 'run') as run:
            env = {'RTX_LR': '1e-6', 'RTX_MODEL_DIR': '/root/models/' + self.spec['model'],
                   'RTX_DATA_PATH': '/workspace/' + self.spec['train_file'], 'RTX_NUM_ROLLOUT': '500',
                   'RTX_SAVE_HF': '1', 'RTX_SAVE_INTERVAL': '50', 'RTX_TRAIN_ENTRY': 'train_async.py',
                   'RTX_FULLY_ASYNC': '1', 'RTX_PAPER_MODE': '1', 'RTX_EXTRA_MODEL_ARGS': '--use-rollout-logprobs'}
            with patch.dict(os.environ, env):
                with self.assertRaises(FileNotFoundError):
                    checks.preflight(self.spec, 'formal', 17)
            run.assert_not_called()

    def test_shell_syntax(self):
        for name in ['start_e7_clean.sh', 'scripts/e7_formal_launch.sh', 'scripts/e7_restart_run.sh']:
            subprocess.run(['bash', '-n', str(ROOT / name)], check=True)

    def test_new_launcher_forwards_fixed_recipe_without_training(self):
        script = (ROOT / 'scripts/e7_restart_run.sh').read_text()
        # Shell functions intercept both external entrypoints; no Docker, files or training.
        mocks = '''
python3() {
 if [[ "$2" == preflight ]]; then
   [[ "$RTX_DATA_PATH" == /workspace/runs/diagnosis-20260910/train.jsonl ]] || exit 91
   [[ "$RTX_PAPER_MODE" == "$EXPECT_PAPER" && "$RTX_SAVE_HF" == 1 ]] || exit 92
   [[ "$RTX_EXTRA_MODEL_ARGS" == *--use-rollout-logprobs* ]] || exit 93
 fi
}
bash() { [[ "$RTX_SKIP_GATE" == 0 && "$RTX_ALLOW_REUSE" == 0 ]] || exit 94; }
'''
        for stage, paper in [('pilot', '0'), ('formal', '1')]:
            env = dict(os.environ, EXPECT_PAPER=paper)
            env.pop('RTX_EXTRA_MODEL_ARGS', None)
            env.pop('RTX_PROFILE', None)
            subprocess.run(['bash', '-c', mocks + script, 'test', stage,
                            '11' if stage == 'pilot' else '17', 'b6', 'e7restart-mock'], env=env, check=True)


if __name__ == '__main__':
    unittest.main()
