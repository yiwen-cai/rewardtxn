"""Opt-in real AReaL CPU contract, with explicitly synthetic generation."""
import asyncio
import copy
import dataclasses
import pickle
import hashlib
import json
import inspect
import logging
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

ENABLED = os.environ.get('FT_RLVR_REPLAY_CPU') == '1'


@unittest.skipUnless(ENABLED, 'requires real AReaL CPU image; FT_RLVR_REPLAY_CPU=1')
class RLVRContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, state, replay, R, F, workflow_context, WorkflowContext, Grouped, RLVR, Wrapper, Gconfig
        import torch
        from scripts.ft import state, replay
        from scripts.ft import rlvr_replay as R
        import rlvr_replay_fixture as F
        from areal import workflow_context
        from areal.infra.workflow_context import WorkflowContext
        from areal.infra.remote_inf_engine import GroupedRolloutWorkflow as Grouped
        from areal.workflow.rlvr import RLVRWorkflow as RLVR
        from areal.api import AsyncRewardWrapper as Wrapper
        from areal.api.cli_args import GenerationHyperparameters as Gconfig
        from transformers import AutoTokenizer
        path = os.environ.get('FT_RLVR_TOKENIZER', '/workspace/models/Qwen2.5-0.5B-Instruct')
        cls.tokenizer_path = path
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        cls.tokenizer_hash = F.asset_hash(path)
        cls.verifier = hashlib.sha256(Path(F.gsm8k_reward_fn.__code__.co_filename).read_bytes()).hexdigest()

    def setUp(self):
        evidence = os.environ.get('FT_RLVR_EVIDENCE_DIR')
        if evidence:
            Path(evidence).mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName + '-', dir=evidence))
        else:
            tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
            self.root = Path(tmp.name)
        self.owner = state.acquire_owner(self.root / 'state', -1, run_nonce='cpu-run',
                                         config_sha256='b' * 64, verifier_version=self.verifier)
        self.addCleanup(self.owner.close)
        from replay_fixture import opened
        with opened(self.root / 'draw') as loader:
            self.data = next(loader)[0]
        self.data.update(answer='4', _cpu_log=str(self.root / 'score.jsonl'))
        self.attempts = {f"{self.data['_r_draw']['group_id']}:{i}":
                         state.authorize_attempt(self.owner, f"{self.data['_r_draw']['group_id']}:{i}",
                                                 None, f'a{i}', expected_policy_version=3)
                         for i in range(8)}
        (self.root / 'provenance.json').write_text(json.dumps({
            'synthetic_generation': True, 'tokenizer_sha256': self.tokenizer_hash,
            'gsm8k_sha256': self.verifier, 'max_workers': 1, 'max_retries': 1,
            'torch_version': torch.__version__,
            'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in {Path(inspect.getfile(obj)) for obj in
                                     (R.CallReturnRLVR, R.ReturningReward, RLVR, Grouped, Wrapper, F.counted_gsm8k, F.gsm8k_reward_fn)}},
            'scope': 'strict GSM8K scoring; synthetic generation CPU'}, indent=2))

    def workflow(self, **overrides):
        args = dict(owner=self.owner, root=self.root / 'artifacts', attempts=self.attempts,
                    tokenizer_sha256=self.tokenizer_hash, reward_fn=F.counted_gsm8k,
                    gconfig=Gconfig(n_samples=1), tokenizer=self.tokenizer)
        args.update(overrides)
        workflow = R.CallReturnRLVR(**args)
        self.addCleanup(workflow.close)
        return workflow

    async def episode(self, workflow, engine, index=0, data=None):
        workflow_context.set(WorkflowContext(task_id=1, sample_idx=index))
        return await workflow.arun_episode(engine, self.data if data is None else data)

    def count(self):
        path = Path(self.data['_cpu_log'])
        return len(path.read_text().splitlines()) if path.exists() else 0

    def equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertEqual(left[key].dtype, right[key].dtype)
            self.assertEqual(left[key].shape, right[key].shape)
            self.assertEqual(left[key].contiguous().numpy().tobytes(), right[key].contiguous().numpy().tobytes())

    def baseline(self):
        workflow = RLVR(F.counted_gsm8k, Gconfig(n_samples=1), self.tokenizer)
        workflow.async_reward_fn = Wrapper(F.counted_gsm8k, max_workers=1, max_retries=0)
        return workflow

    def test_four_stage_paths_and_native_builder_bytes(self):
        async def run():
            for index, cut in ((0, 'none'), (1, 'response'), (2, 'reward')):
                with self.subTest(cut=cut):
                    wf = self.workflow(); engine = F.FakeEngine(self.tokenizer)
                    original = wf._put
                    phases = []
                    def put(context, name, payload):
                        if (cut == 'response' and name == 'reward') or (cut == 'reward' and name == 'tensor'):
                            raise InterruptedError('fixture interruption before ' + name)
                        result = original(context, name, payload); phases.append(name); return result
                    with patch.object(wf, '_put', put):
                        if cut == 'none':
                            await self.episode(wf, engine, index)
                        else:
                            with self.assertRaises(InterruptedError): await self.episode(wf, engine, index)
                    self.assertEqual(engine.calls, [index])
                    self.assertEqual(phases, {'none': ['response', 'reward', 'tensor'],
                                             'response': ['response'], 'reward': ['response', 'reward']}[cut])
                    fresh = self.workflow(); newengine = F.FakeEngine(self.tokenizer)
                    scores = self.count()
                    actual = await self.episode(fresh, newengine, index)
                    self.assertEqual(newengine.calls, [])
                    self.assertEqual(self.count() - scores, int(cut == 'response'))
                    native = await self.episode(self.baseline(), F.FakeEngine(self.tokenizer), index)
                    self.equal(actual, native)
                    actual['input_ids'].zero_()
                    self.equal(await self.episode(self.workflow(), newengine, index), native)
            self.assertEqual(len(list((self.root / 'artifacts/samples').glob('*/tensor.json'))), 3)
        asyncio.run(run())

    def test_grouped_mixed_cache_padding_order_and_immediate_score(self):
        async def run():
            await self.episode(self.workflow(), F.FakeEngine(self.tokenizer), 0)
            partial = self.workflow()
            async def stop(*args): raise InterruptedError('after durable response')
            with patch.object(partial, '_compute_rewards', stop):
                with self.assertRaises(InterruptedError): await self.episode(partial, F.FakeEngine(self.tokenizer, answer='5'), 1)
            scores = self.count(); engine = F.FakeEngine(self.tokenizer, answer={i: ('4' if i % 2 == 0 else '5') for i in range(8)})
            grouped = Grouped(self.workflow(), 8, logging.getLogger('cpu-contract'))
            actual = await grouped.arun_episode(engine, self.data)
            self.assertEqual(sorted(engine.calls), list(range(2, 8)))
            self.assertEqual(self.count() - scores, 7)
            native = await Grouped(self.baseline(), 8, logging.getLogger('cpu-contract')).arun_episode(F.FakeEngine(self.tokenizer, answer={i: ('4' if i % 2 == 0 else '5') for i in range(8)}), self.data)
            self.assertEqual(actual['rewards'].tolist(), [1., 0.] * 4)
            self.equal(actual, native)
        asyncio.run(run())

    def test_no_group_generation_barrier(self):
        async def run():
            gate = asyncio.Event(); engine = F.FakeEngine(self.tokenizer, gate=gate)
            wf = self.workflow(); completed = asyncio.Event(); original = wf._compute_rewards
            async def observe(*args):
                result = await original(*args)
                if workflow_context.get().sample_idx == 0: completed.set()
                return result
            with patch.object(wf, '_compute_rewards', observe):
                task = asyncio.create_task(Grouped(wf, 2, logging.getLogger('cpu')).arun_episode(engine, self.data))
                try:
                    await asyncio.wait_for(engine.entered.wait(), 10)
                    await asyncio.wait_for(completed.wait(), 30)
                    self.assertFalse(gate.is_set())
                finally:
                    gate.set()
                    await task
        asyncio.run(run())

    def test_corruption_and_frozen_input_mismatch_fail_closed(self):
        async def run():
            await self.episode(self.workflow(), F.FakeEngine(self.tokenizer))
            changed = copy.deepcopy(self.data); changed['answer'] = '5'
            for kwargs, data in (({}, changed), ({'tokenizer_sha256': 'c' * 64}, self.data),
                                 ({'gconfig': Gconfig(n_samples=1, temperature=0.123)}, self.data)):
                engine = F.FakeEngine(self.tokenizer)
                with self.assertRaises(R.ArtifactError): await self.episode(self.workflow(**kwargs), engine, data=data)
                self.assertEqual(engine.calls, [])
            path = next((self.root / 'artifacts/samples').glob('*/response.json'))
            record = json.loads(path.read_text())
            blob = self.root / 'artifacts/blobs' / record['blob']['sha256']
            blob.write_bytes(b'truncated')
            with self.assertRaises(replay.ReplayError): await self.episode(self.workflow(), F.FakeEngine(self.tokenizer))
        asyncio.run(run())

    def test_authority_mixed_versions_and_await_supersession(self):
        async def run():
            for key in ('epoch', 'verifier'):
                attempts = copy.deepcopy(self.attempts)
                for attempt in attempts.values():
                    if key == 'epoch': attempt['epoch'] += 1
                    else: attempt['versions']['verifier_version'] = 'e' * 64
                with self.assertRaises(R.ArtifactError): await self.episode(self.workflow(attempts=attempts), F.FakeEngine(self.tokenizer))
            with self.assertRaises(R.ArtifactError): await self.episode(self.workflow(), F.FakeEngine(self.tokenizer, mixed=True))
            gate = asyncio.Event(); engine = F.FakeEngine(self.tokenizer, gate=gate, gated_index=0)
            task = asyncio.create_task(self.episode(self.workflow(), engine))
            await engine.entered.wait()
            sample = next(iter(self.attempts))
            state.authorize_attempt(self.owner, sample, 'a0', 'replacement', expected_policy_version=3)
            gate.set()
            with self.assertRaises(R.ArtifactError): await task
            with state._locked(self.owner) as control: self.assertEqual(control['accepted'], {})
            self.assertFalse(list((self.root / 'artifacts/samples').glob('*/response.json')))
        asyncio.run(run())

    def test_real_new_owner_rejects_old_adoption_and_regenerates(self):
        root = self.root / 'new-owner'
        process = subprocess.run([sys.executable, str(Path(F.__file__)), str(root), self.tokenizer_path],
                                 capture_output=True, text=True, timeout=60)
        (self.root / 'owner-child.log').write_text(process.stdout + process.stderr)
        self.assertEqual(process.returncode, 0, process.stderr)
        old = json.loads((root / 'old.json').read_text())
        control = json.loads((root / 'state/control.json').read_text())
        owner = state.acquire_owner(root / 'state', 0, control['processes'])
        self.addCleanup(owner.close)
        sample = old['attempt']['sample']
        args = dict(owner=owner, root=root / 'artifacts', attempts={sample: old['attempt']})
        async def run():
            with self.assertRaises(R.ArtifactError):
                await self.episode(self.workflow(**args), F.FakeEngine(self.tokenizer), data=old['data'])
            new = state.authorize_attempt(owner, sample, 'old', 'new', expected_policy_version=3)
            args['attempts'] = {sample: new}
            engine = F.FakeEngine(self.tokenizer)
            await self.episode(self.workflow(**args), engine, data=old['data'])
            self.assertEqual(engine.calls, [0])
            self.assertEqual(len(list((root / 'artifacts/samples').glob('*/response.json'))), 2)
        asyncio.run(run())

    def test_legitimate_zero_is_cached_and_reused(self):
        self.assertEqual(F.gsm8k_reward_fn('', r'The answer is \boxed{5}.', [], [], answer='4'), 0.0)
        async def run():
            result = await self.episode(self.workflow(), F.FakeEngine(self.tokenizer, answer='5'))
            self.assertEqual(result['rewards'].item(), 0.0)
            engine = F.FakeEngine(self.tokenizer)
            self.equal(await self.episode(self.workflow(), engine), result)
            self.assertEqual(engine.calls, [])
        asyncio.run(run())
        self.assertEqual(self.count(), 1)
        with state._locked(self.owner) as control: self.assertEqual(len(control['accepted']), 1)

    def test_inner_exception_is_terminal_and_not_cached(self):
        from areal.utils.reward_status import RewardEvaluationError
        self.data['_cpu_inner_exception'] = True
        async def run():
            with self.assertRaises(RewardEvaluationError) as error:
                await self.episode(self.workflow(), F.FakeEngine(self.tokenizer))
            self.assertEqual([a['status'] for a in error.exception.attempts], ['error', 'error'])
        asyncio.run(run())
        self.assertFalse(list((self.root / 'artifacts/samples').glob('*/reward.json')))
        self.assertFalse(list((self.root / 'artifacts/samples').glob('*/tensor.json')))
        self.assertEqual(self.count(), 0)

    def test_envelope_binding_and_pickle(self):
        async def run():
            for i, field in enumerate(('invocation_nonce', 'input_sha256', 'verifier_sha256')):
                wf = self.workflow(); original = wf.async_reward_fn
                async def wrong(*args, **kwargs):
                    packet = await original(*args, **kwargs)
                    self.assertEqual(pickle.loads(pickle.dumps(packet)), packet)
                    return dataclasses.replace(packet, **{field: 'f' * 64})
                wf.async_reward_fn = wrong
                with self.assertRaises(R.UncertainReward):
                    await self.episode(wf, F.FakeEngine(self.tokenizer), i)
            with state._locked(self.owner) as control: self.assertEqual(control['accepted'], {})
            self.assertEqual(list((self.root / 'artifacts/samples').glob('*/reward.json')), [])
        asyncio.run(run())

    def test_cached_nonce_and_old_schema_cannot_self_authorize(self):
        async def run():
            await self.episode(self.workflow(), F.FakeEngine(self.tokenizer))
            store = replay.BlobStore(self.root / 'artifacts/blobs')
            reward_path = next((self.root / 'artifacts/samples').glob('*/reward.json'))
            record = json.loads(store.get(json.loads(reward_path.read_text())['blob']))
            original = copy.deepcopy(record)
            record['payload']['return']['invocation_nonce'] = 'f' * 64
            changed_reward = store.put(replay._encode(record))
            reward_path.write_text(json.dumps({'blob': changed_reward}))
            # First retain the independent upstream-hash rejection contract.
            with self.assertRaisesRegex(R.ArtifactError, 'tensor upstream binding mismatch'):
                await self.episode(self.workflow(), F.FakeEngine(self.tokenizer))
            # Adversarial fixture: rebuild every affected reference so that the
            # complete chain is hash-consistent, but its nonce is still wrong.
            tensor_path = reward_path.with_name('tensor.json')
            tensor_index = json.loads(tensor_path.read_text())
            tensor = json.loads(store.get(tensor_index['blob']))
            tensor['payload']['reward_sha256'] = changed_reward['sha256']
            tensor_path.write_text(json.dumps({'blob': store.put(replay._encode(tensor))}))
            wf = self.workflow(); engine = F.FakeEngine(self.tokenizer); scores = self.count()
            with patch.object(wf, '_validate_return', wraps=wf._validate_return) as validate:
                with self.assertRaisesRegex(R.UncertainReward, 'mismatched official-call-return envelope'):
                    await self.episode(wf, engine)
                self.assertEqual(validate.call_count, 1)
                context, response, envelope = validate.call_args.args
                self.assertEqual(envelope.invocation_nonce, 'f' * 64)
                self.assertNotEqual(envelope.invocation_nonce, wf._invocation_nonce(context, response))
            self.assertEqual(engine.calls, [])
            self.assertEqual(self.count(), scores)
            reward_path.write_text(json.dumps({'blob': store.put(replay._encode(original))}))
            tensor_path.write_text(json.dumps(tensor_index))
            response_path = reward_path.with_name('response.json')
            record = json.loads(store.get(json.loads(response_path.read_text())['blob']))
            record['binding']['schema'] = 1
            response_path.write_text(json.dumps({'blob': store.put(replay._encode(record))}))
            with self.assertRaises(R.ArtifactError): await self.episode(self.workflow(), F.FakeEngine(self.tokenizer))
        asyncio.run(run())

    def test_owner_cas_during_real_reward_await(self):
        async def run():
            wf = self.workflow(); original = wf.async_reward_fn; entered = asyncio.Event()
            async def observed(*args, **kwargs):
                entered.set()
                return await original(*args, **kwargs)
            wf.async_reward_fn = observed
            self.data['_cpu_delay'] = 0.2
            task = asyncio.create_task(self.episode(wf, F.FakeEngine(self.tokenizer)))
            await entered.wait()
            sample = next(iter(self.attempts))
            state.authorize_attempt(self.owner, sample, 'a0', 'new-after-reward-start', expected_policy_version=3)
            with self.assertRaises(R.ArtifactError): await task
            self.assert_no_complete_result()
        asyncio.run(run())

    def assert_no_complete_result(self):
        self.assertEqual(len(list((self.root / 'artifacts/samples').glob('*/response.json'))), 1)
        for phase in ('reward', 'tensor'):
            self.assertEqual(list((self.root / 'artifacts/samples').glob('*/' + phase + '.json')), [])
        with state._locked(self.owner) as control: self.assertEqual(control['accepted'], {})

    def test_z_actual_wrapper_timeout_is_terminal_without_late_score(self):
        from areal.utils.reward_status import RewardEvaluationError
        self.data['_cpu_delay'] = 0.5
        async def run():
            with self.assertRaises(RewardEvaluationError) as error:
                await self.episode(self.workflow(timeout_seconds=0.02), F.FakeEngine(self.tokenizer))
            self.assertEqual([a['status'] for a in error.exception.attempts],['timeout','timeout'])
        asyncio.run(run())
        self.assert_no_complete_result()
        time.sleep(.6)
        self.assertEqual(self.count(), 0)
        self.assert_no_complete_result()


if __name__ == '__main__': unittest.main()
