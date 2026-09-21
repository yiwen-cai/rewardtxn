"""Real PPO dispatch into installed hooks; CPU terminals, no GPU claim."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class HookContract(unittest.TestCase):
    def test_native_keyword_forward_and_separate_optimizer_scheduler(self):
        import torch
        from areal.api.cli_args import PPOActorConfig
        from areal.engine.megatron_engine import MegatronPPOActor
        from areal.trainer.ppo.actor import PPOActor
        import areal.trainer.rl_trainer as trainer_module
        from scripts.ft.batch_identity import KEYS
        from scripts.ft.training_adapter import install
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        calls = []
        def terminal_forward(self, input_, *args, **kwargs):
            self_test.assertFalse(set(KEYS) & set(input_))
            calls.append('forward')
            return torch.zeros_like(input_['input_ids'], dtype=torch.float32)
        def terminal_optimizer(self):
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
            return {'update_successful': 1.0, 'grad_norm': 1.0, 'lr': 0.01}
        def terminal_scheduler(self):
            scheduler.step()
        self_test = self
        runtime = SimpleNamespace(optimizer_stats=None, scheduler_done=False, generation='cpu-hook',
                                  event=lambda name, **fields: calls.append(name))
        with ExitStack() as stack:
            for cls, names in ((MegatronPPOActor, ('ppo_update', 'train_batch', 'optimizer_step', 'lr_scheduler_step', 'forward')),
                               (trainer_module.PPOTrainer, ('_create_dataloader',))):
                for name in names:
                    stack.enter_context(patch.object(cls, name, getattr(cls, name)))
            stack.enter_context(patch.object(trainer_module, 'RecoverHandler', trainer_module.RecoverHandler))
            MegatronPPOActor.forward = terminal_forward
            MegatronPPOActor.optimizer_step = terminal_optimizer
            MegatronPPOActor.lr_scheduler_step = terminal_scheduler
            install(runtime)
            actor = MegatronPPOActor.__new__(MegatronPPOActor)
            actor.eval = lambda: None
            actor.actor = PPOActor(PPOActorConfig(recompute_logprob=False), actor)
            batch = {'input_ids': torch.ones((1, 4), dtype=torch.int32),
                     'attention_mask': torch.ones((1, 4), dtype=torch.bool)}
            batch.update({k: torch.ones((1, 4), dtype=torch.int64) for k in KEYS})
            result = actor.compute_logp([batch])  # original PPO uses input_= keyword
            self.assertEqual(len(result), 1)
            self.assertTrue(all(k in batch for k in KEYS))
            with self.assertRaisesRegex(RuntimeError, 'scheduler out'):
                actor.step_lr_scheduler()
            actor.optimizer_step()
            self.assertLess(parameter.item(), 1.0)
            self.assertFalse(runtime.scheduler_done)
            actor.step_lr_scheduler()
            self.assertTrue(runtime.scheduler_done)
            self.assertEqual(optimizer.param_groups[0]['lr'], 0.005)
            self.assertEqual(calls, ['forward', 'optimizer_applied', 'scheduler_applied'])
            with self.assertRaisesRegex(RuntimeError, 'more than one'):
                actor.optimizer_step()


if __name__ == '__main__':
    unittest.main()
