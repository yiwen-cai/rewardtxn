"""CPU checks for paired smoke preparation and real scoring observation."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from check_ft1_smoke import natural_f4
from run_ft1 import REPO, smoke_config


class Preparation(unittest.TestCase):
    def test_config_preserves_common_async_and_sampling(self):
        from areal.api.cli_args import GRPOConfig, load_expr_config
        base = (REPO/'docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml').read_text()
        base = base.replace('async_save: false', 'async_save: true')
        for seed in (401, 409):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'config.yaml'
                path.write_text(smoke_config(base, seed))
                cfg, _ = load_expr_config(['--config', str(path)], GRPOConfig)
                self.assertEqual((cfg.seed, cfg.total_train_steps, cfg.recover.retries), (seed, 10, 0))
                self.assertEqual((cfg.gconfig.n_samples, cfg.train_dataset.batch_size), (8, 4))
                self.assertTrue(cfg.actor.megatron.async_save)
                self.assertFalse(cfg.recover.no_save_optim)
                self.assertTrue(cfg.actor.megatron.use_checkpoint_opt_param_scheduler)

    def test_changed_base_and_unfrozen_seed_are_rejected(self):
        with self.assertRaises(ValueError):
            smoke_config('', 401)
        with self.assertRaises(ValueError):
            smoke_config('', 211)

    def test_f4_requires_all_generation_before_fourth_distinct_execution(self):
        def event(kind, idx, clock):
            return dict(event=kind, sample_idx=idx, monotonic_ns=clock, source_row_id=2, task_id=7)
        generations = [event('generation_complete', i, i) for i in range(8)]
        starts = [event('score_execution_started', i, 10+i) for i in range(4)]
        self.assertEqual(len(natural_f4(generations+starts)), 1)
        self.assertEqual(natural_f4(generations[:-1]+starts), [])
        self.assertEqual(natural_f4(generations[:-1]+starts+[event('generation_complete', 7, 99)]), [])
        starts[-1]['sample_idx'] = 0
        self.assertEqual(natural_f4(generations+starts), [])

    def test_real_reward_child_preserves_identity_and_is_reaped(self):
        from areal import workflow_context
        from areal.infra.workflow_context import WorkflowContext
        from areal.api import AsyncRewardWrapper
        from areal.reward.gsm8k import gsm8k_reward_fn
        from areal.workflow.rlvr import RLVRWorkflow
        from areal.utils import strict_reward
        from scripts.ft.areal_ft1 import install_observer
        original_rewards, original_worker = RLVRWorkflow._compute_rewards, strict_reward._worker
        context = workflow_context.get()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'FT1_OBSERVE': directory}):
                install_observer()
                workflow_context.set(WorkflowContext(task_id=7, sample_idx=3))
                workflow = object.__new__(RLVRWorkflow)
                workflow.tokenizer = SimpleNamespace(decode=lambda tokens: '4')
                workflow.async_reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)
                response = SimpleNamespace(input_tokens=[1], output_tokens=[4], output_versions=[0])
                value = asyncio.run(workflow._compute_rewards(response, '2+2', {'source_row_id': 23, 'answer': '4'}))
                self.assertEqual(value, 1)
                events = [json.loads(line) for path in Path(directory).glob('*.jsonl') for line in path.read_text().splitlines()]
                generated, started = sorted(events, key=lambda e: e['monotonic_ns'])
                self.assertEqual([generated['event'], started['event']], ['generation_complete', 'score_execution_started'])
                self.assertEqual((started['source_row_id'], started['task_id'], started['sample_idx']), (23, 7, 3))
                self.assertNotEqual(started['pid'], os.getpid())
                self.assertEqual(started['pid'], started['identity']['pid'])
                self.assertFalse(Path(f"/proc/{started['pid']}").exists())
        finally:
            RLVRWorkflow._compute_rewards, strict_reward._worker = original_rewards, original_worker
            workflow_context.set(context)


if __name__ == '__main__':
    unittest.main()
