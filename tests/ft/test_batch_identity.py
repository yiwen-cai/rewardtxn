"""Real async AReaL executor/PPO CPU bridge; synthetic generation, no optimizer."""
import asyncio
import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('FT_BATCH_IDENTITY_CPU') == '1', 'requires real AReaL CPU image')
class BatchBridgeContract(unittest.TestCase):
    def test_actual_async_executor_ppo_boundary_and_negative_contracts(self):
        import torch
        from transformers import AutoTokenizer
        from areal import workflow_context
        from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig, PPOActorConfig, MicroBatchSpec
        from areal.infra.workflow_context import WorkflowContext
        from areal.infra.workflow_executor import WorkflowExecutor
        from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
        from areal.trainer.ppo.actor import PPOActor
        from areal.utils.data import concat_batch, split_padded_tensor_dict_into_mb_list
        from scripts.ft import state
        from scripts.ft.rlvr_replay import CallReturnRLVR, ArtifactError
        from scripts.ft.batch_identity import IdentityBridge, PreparedBoundary, KEYS, strip_identity
        from replay_fixture import opened
        from rlvr_replay_fixture import FakeEngine, counted_gsm8k, gsm8k_reward_fn, asset_hash

        evidence = os.environ.get('FT_BATCH_IDENTITY_EVIDENCE_DIR')
        if evidence:
            Path(evidence).mkdir(parents=True, exist_ok=True)
            root = Path(tempfile.mkdtemp(prefix='bridge-', dir=evidence))
        else:
            tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); root = Path(tmp.name)
        tokenizer_path = os.environ.get('FT_RLVR_TOKENIZER', '/workspace/models/Qwen2.5-0.5B-Instruct')
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        verifier = hashlib.sha256(Path(gsm8k_reward_fn.__code__.co_filename).read_bytes()).hexdigest()
        owner = state.acquire_owner(root / 'state', -1, run_nonce='cpu-run', config_sha256='b'*64, verifier_version=verifier)
        self.addCleanup(owner.close)
        workflow = CallReturnRLVR(owner=owner, root=root / 'artifacts', attempts={},
                                  tokenizer_sha256=asset_hash(tokenizer_path), reward_fn=counted_gsm8k,
                                  gconfig=GenerationHyperparameters(n_samples=1), tokenizer=tokenizer)
        self.addCleanup(workflow.close)
        bridge = IdentityBridge(workflow, root / 'bridge', policy_version=3); self.addCleanup(bridge.close)
        def enrich(item):
            return dict(item, answer='4', _cpu_log=str(root / 'score.jsonl'))
        # Separate loader instance owns warm-up iteration. Executor's new instance
        # has only its producer thread, while durable draws retain identity.
        with opened(root / 'draw') as warm_loader:
            first = [enrich(item) for item in next(warm_loader)]
        for item in first: bridge.authorize(item)
        async def warm():
            workflow_context.set(WorkflowContext(task_id=1, sample_idx=0))
            await workflow.arun_episode(FakeEngine(tokenizer), first[0])
            async def stop(*args): raise InterruptedError('response durable')
            from unittest.mock import patch
            workflow_context.set(WorkflowContext(task_id=2, sample_idx=1))
            with patch.object(workflow, '_compute_rewards', stop):
                with self.assertRaises(InterruptedError):
                    await workflow.arun_episode(FakeEngine(tokenizer), first[0])
        asyncio.run(warm())
        draw = opened(root / 'draw'); self.addCleanup(draw.close)
        class AuthorizedLoader:
            batch_size = draw.batch_size
            sampler = draw.sampler
            def __len__(self): return len(draw)
            def __iter__(self):
                for batch in draw:
                    result = [enrich(item) for item in batch]
                    for item in result: bridge.authorize(item)
                    yield result
        class AsyncEngine(FakeEngine):
            def __init__(self):
                super().__init__(tokenizer, answer={i: '4' if i % 2 == 0 else '5' for i in range(8)})
                self.version = 3
            def get_version(self): return self.version
            async def agenerate(self, request):
                # Test-only variable delay, preserving generate -> immediate score.
                await asyncio.sleep(0.002 * (7 - workflow_context.get().sample_idx))
                return await super().agenerate(request)
        engine = AsyncEngine()
        grouped = GroupedRolloutWorkflow(bridge, 8, logging.getLogger('bridge'), drop_incomplete_group=True)
        executor = WorkflowExecutor(InferenceEngineConfig(consumer_batch_size=4, max_concurrent_rollouts=4,
                                                         max_head_offpolicyness=2, check_trajectory_format=True), engine)
        executor.initialize(train_data_parallel_size=1)
        rejected_trajectories = []
        def accept(trajectory):
            # Test filter rejects one whole arrival without reading artifacts
            # on the async event loop. Identities are verified after shutdown.
            if not rejected_trajectories:
                rejected_trajectories.append(trajectory)
                return False
            return True
        try:
            batch = executor.prepare_batch(AuthorizedLoader(), grouped, should_accept_fn=accept)
            # Different args are intentionally invalid: native cached generator
            # must continue to use the original loader/workflow/filter.
            class NotUsed:
                batch_size = 4
                def __iter__(self): raise AssertionError('cached generator replaced')
            second = executor.prepare_batch(NotUsed(), None)
        finally:
            executor.destroy()
        self.assertEqual(len(batch), 4); self.assertEqual(len(second), 4)
        all_rows = [r for t in batch + second for r in bridge.validate_rows(t)]
        self.assertEqual(len(all_rows), 64)
        self.assertEqual(len({r['sample'] for r in all_rows}), 64)
        rejected = [r['sample'] for t in rejected_trajectories for r in bridge.validate_rows(t)]
        self.assertEqual(len(rejected), 8)
        # Identical token rows from separate occurrences must keep different identities.
        self.assertTrue(torch.equal(batch[0]['input_ids'][0][batch[0]['attention_mask'][0]],
                                    batch[1]['input_ids'][0][batch[1]['attention_mask'][0]]))
        self.assertNotEqual(bridge.validate_rows(batch[0])[0]['identity'],
                            bridge.validate_rows(batch[1])[0]['identity'])
        self.assertFalse(set(rejected) & {r['sample'] for r in all_rows})
        # Reject is still an obligation, not a consumed receipt or state token.
        with state._locked(owner) as control: self.assertIsNone(control['head'])

        class BoundaryEngine:
            def get_version(self): return 3
            def train(self): pass
            def eval(self): pass
            def forward(self, input_, aggregate_fn):
                clean = strip_identity(input_)
                self.assert_no_identity(clean)
                # Explicit synthetic forward result, no model/optimizer claim.
                return torch.zeros_like(clean['input_ids'], dtype=torch.float32)
            def assert_no_identity(self, data):
                if any(key in data for key in KEYS): raise AssertionError('identity leaked to model')
            def train_batch(self, data, **kwargs):
                clean = strip_identity(data)
                self.assert_no_identity(clean)
                if not all(key in data for key in KEYS): raise AssertionError('caller carrier mutated')
                return bridge.at_train_batch(data)
        config = PPOActorConfig(ppo_n_minibatches=1, recompute_logprob=False,
                                mask_no_eos_with_zero=False, reward_scaling=2.0, reward_bias=-0.5)
        actor = PPOActor(config, BoundaryEngine())
        # Real batched_call/forward entry verifies stripping leaves original intact.
        actor.compute_logp(batch)
        self.assertTrue(all(all(k in t for k in KEYS) for t in batch))
        advantages = actor.compute_advantages(batch)
        frozen = dataclasses.asdict(config)
        intent = bridge.begin_update(advantages, frozen)
        reordered = bridge.begin_update(list(reversed(advantages)), frozen)
        self.assertEqual(intent['logical_update_id'], reordered['logical_update_id'])
        self.assertNotEqual(intent['physical_invocation_id'], reordered['physical_invocation_id'])
        # Call the actual PPO update and its real splitter; terminal explicitly stops.
        with self.assertRaises(PreparedBoundary): actor.ppo_update(advantages)
        intents = list((root / 'bridge').glob('intent-*.json'))
        self.assertEqual(len(intents), 1)
        persisted = json.loads(intents[0].read_text())
        self.assertEqual(persisted['status'], 'prepared_not_applied')
        self.assertEqual(len(persisted['microbatch_rows']), 32)
        self.assertFalse(set(KEYS) & set(persisted['actual_train_tensors']))

        packed, _ = concat_batch(advantages)
        mbs = split_padded_tensor_dict_into_mb_list(packed, MicroBatchSpec(n_mbs=2)).mbs
        rows = [r for mb in mbs for r in bridge.validate_rows(mb)]
        self.assertEqual(sorted(r['sample'] for r in rows), sorted(r['sample'] for r in intent['rows']))
        self.assertEqual(len(mbs), 2)
        # Real splitter regression demonstration: a 1D row ID is copied whole.
        bad_sidecar = dict(packed, _bad_row_id=torch.arange(packed['input_ids'].shape[0]))
        bad_mbs = split_padded_tensor_dict_into_mb_list(bad_sidecar, MicroBatchSpec(n_mbs=2)).mbs
        self.assertTrue(any(m['_bad_row_id'].shape[0] != m['input_ids'].shape[0] for m in bad_mbs))
        # Partial microbatch groups are legal; complete K is checked once above.
        with self.assertRaises(ArtifactError): bridge.begin_update([dict((k,v[:1]) for k,v in advantages[0].items())], frozen)
        with self.assertRaises(ArtifactError): bridge.begin_update(advantages + advantages[:1], frozen)
        bad = {k:v.clone() for k,v in packed.items()}
        del bad[KEYS[0]]
        with self.assertRaises(ArtifactError): bridge.validate_rows(bad)
        bad = {k:v.clone() for k,v in packed.items()}
        # Different sample lengths/tokens: swap all four identities, not content.
        for key in KEYS:
            a, b = bad[key][0,0].item(), bad[key][1,0].item()
            bad[key][0][bad['attention_mask'][0]] = b
            bad[key][1][bad['attention_mask'][1]] = a
        with self.assertRaises(ArtifactError): bridge.validate_rows(bad)
        bad = {k:v.clone() for k,v in packed.items()}
        pad = (~bad['attention_mask']).nonzero()
        self.assertGreater(len(pad), 0)
        bad[KEYS[0]][tuple(pad[0])] = 1
        with self.assertRaises(ArtifactError): bridge.validate_rows(bad)
        bad = {k:v.clone() for k,v in packed.items()}
        bad['advantages'][0,0] += 1
        with self.assertRaisesRegex(ArtifactError, 'PPO input changed'):
            bridge.at_train_batch(bad)
        sample = intent['rows'][0]['sample']; attempt = workflow.attempts[sample]
        state.authorize_attempt(owner, sample, attempt['attempt'], 'superseded', expected_policy_version=3)
        with self.assertRaises(ArtifactError): bridge.validate_rows(packed)
        (root / 'result.json').write_text(json.dumps({'synthetic_generation_and_forward': True,
            'real_async_executor_grouped_ppo_and_split': True, 'optimizer_executed': False,
            'consumed_authority': False, 'groups_delivered': 8, 'rejected_pending_samples': rejected,
            'microbatch_sizes': [m['input_ids'].shape[0] for m in mbs],
            'logical_update_id': intent['logical_update_id'],
            'sources': {inspect.getfile(obj): hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
                        for obj in (IdentityBridge, WorkflowExecutor, PPOActor, CallReturnRLVR)}}, indent=2))


if __name__ == '__main__': unittest.main()
