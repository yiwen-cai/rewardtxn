"""Real strict score processes and native rollout failure propagation on CPU."""
import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch


def controlled_score(prompt, completion, prompt_ids, completion_ids, answer, record, mode, **extra):
    from areal.reward.gsm8k import gsm8k_reward_fn
    from scripts.ft.state import process_identity
    path = Path(record)
    prior = path.read_text().splitlines() if path.exists() else []
    with path.open('a') as f:
        f.write(json.dumps(process_identity(os.getpid()))+'\n'); f.flush(); os.fsync(f.fileno())
    if mode == 'hang':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        while True: time.sleep(1)
    if mode == 'error' or (mode == 'once' and not prior):
        raise RuntimeError('injected scoring execution failure')
    return gsm8k_reward_fn(prompt, completion, prompt_ids, completion_ids, answer=answer)


controlled_score.strict_scoring = True


class VerifierStatus(unittest.TestCase):
    def test_normal_scores_match_legacy_rules(self):
        from areal.reward import MathVerifyWorker
        from areal.reward.gsm8k import gsm8k_reward_fn
        worker=MathVerifyWorker(timeout=None)
        for completion,answer,expected in [('4','4',1),('5','4',0),('','4',0),
                ('nonsense','4',0),(r'\boxed{1/2}','0.5',1),('!!!invalid!!!','4',0)]:
            with self.subTest(completion=completion):
                self.assertEqual(gsm8k_reward_fn('',completion,[],[],answer=answer),expected)
                self.assertEqual(worker.verify(completion,answer),expected)

    def test_gold_and_exposed_parse_compare_errors_are_distinct(self):
        from areal.reward import MathVerifyWorker
        from areal.utils.reward_status import ScoringFailure
        worker=MathVerifyWorker(timeout=None)
        with self.assertRaises(ScoringFailure) as error: worker.verify_strict('4','None')
        self.assertEqual(error.exception.status,'invalid_gold')
        for target in ['areal.reward.parse','areal.reward.math_verify_verify']:
            with patch(target,side_effect=RuntimeError('injected backend error')):
                with self.assertRaises(ScoringFailure) as error: worker.verify_strict('4','4')
                self.assertEqual(error.exception.status,'error')

    def test_data_preflight_reports_original_bad_row_without_replacing_it(self):
        from scripts.ft.validate_scoring_data import validate
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.jsonl'
            path.write_text(json.dumps({'label':'4'})+'\n'+json.dumps({'label':'None'})+'\n')
            before=path.read_bytes()
            result=asyncio.run(validate(path))
            self.assertFalse(result['verified'])
            self.assertEqual(result['attempts'][0]['status'],'invalid_gold')
            self.assertIn('row 2',result['attempts'][0]['detail'])
            self.assertEqual(path.read_bytes(),before)


class StrictProcesses(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'attempts.jsonl'

    def wrapper(self, scorer=controlled_score, **kw):
        from areal.api import AsyncRewardWrapper
        return AsyncRewardWrapper(scorer,**kw)

    def data(self,mode,answer='4'):
        return dict(answer=answer,record=str(self.path),mode=mode)

    def assert_reaped(self):
        from scripts.ft.state import _exited
        rows=[json.loads(x) for x in self.path.read_text().splitlines()]
        self.assertTrue(all(_exited(p) for p in rows));return rows

    async def test_real_retry_succeeds_on_same_input_and_default_matches_r(self):
        from scripts.ft.reward_return import ReturningReward,input_digest
        args=('', '4', [], []);data=self.data('once')
        a=self.wrapper();r=self.wrapper(ReturningReward(controlled_score))
        self.assertEqual((a.max_workers,a.max_retries,a.timeout_seconds),
                         (r.max_workers,r.max_retries,r.timeout_seconds))
        self.assertEqual(a.max_retries,1)
        self.assertEqual(await a(*args,**data),1)
        self.assertEqual(len(self.assert_reaped()),2)
        self.path.unlink()
        request={'invocation_nonce':'b'*64,'input_sha256':input_digest(*args,data),
                 'verifier_sha256':'a'*64}
        result=await r(*args,**data,_r_reward_request=request)
        self.assertEqual((result.schema,result.status,result.score),(2,'scored',1))
        self.assertEqual(len(self.assert_reaped()),2)

    async def test_permanent_error_retains_two_failed_attempts(self):
        from areal.utils.reward_status import RewardEvaluationError
        with self.assertRaises(RewardEvaluationError) as error:
            await self.wrapper()('', '4', [], [], **self.data('error'))
        attempts=error.exception.attempts
        self.assertEqual([p['status'] for p in attempts],['error','error'])
        self.assertEqual(len({p['input_sha256'] for p in attempts}),1)
        self.assertEqual(len(self.assert_reaped()),2)

    async def test_timeout_kills_stubborn_worker_before_retry_and_return(self):
        from areal.utils.reward_status import RewardEvaluationError
        started=time.monotonic()
        with self.assertRaises(RewardEvaluationError) as error:
            await self.wrapper(timeout_seconds=.15)('', '4', [], [], **self.data('hang'))
        self.assertEqual([p['status'] for p in error.exception.attempts],['timeout','timeout'])
        self.assertLess(time.monotonic()-started,3)
        self.assertEqual(len(self.assert_reaped()),2)

    async def test_invalid_gold_is_not_retried_or_zeroed(self):
        from areal.utils.reward_status import RewardEvaluationError
        with self.assertRaises(RewardEvaluationError) as error:
            await self.wrapper()('', '4', [], [], **self.data('normal',answer='None'))
        self.assertEqual([p['status'] for p in error.exception.attempts],['invalid_gold'])
        self.assertEqual(len(self.assert_reaped()),1)

    async def test_cancellation_cleans_worker_and_releases_capacity(self):
        wrapper=self.wrapper(timeout_seconds=10)
        task=asyncio.create_task(wrapper('', '4', [], [], **self.data('hang')))
        while not self.path.exists():await asyncio.sleep(.01)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertEqual(len(self.assert_reaped()),1)
        self.assertEqual(await wrapper('', '5', [], [], **self.data('normal')),0)
        self.assertEqual(len(self.assert_reaped()),2)

    async def test_process_start_and_serialization_failures_are_terminal(self):
        from areal.utils.reward_status import RewardEvaluationError
        from multiprocessing.context import ForkProcess
        with patch.object(ForkProcess,'start',side_effect=OSError('injected start failure')):
            with self.assertRaises(RewardEvaluationError) as error:
                await self.wrapper()('', '4', [], [], **self.data('normal'))
        self.assertEqual(error.exception.attempts[0]['status'],'infrastructure_error')
        self.assertFalse(self.path.exists())
        with self.assertRaises(RewardEvaluationError) as error:
            await self.wrapper()('', '4', [], [], unsupported=object(), **self.data('normal'))
        self.assertEqual(error.exception.attempts[0]['status'],'infrastructure_error')

    async def test_repeated_calls_close_process_handles(self):
        before=len(list(Path('/proc/self/fd').iterdir()))
        for _ in range(6):
            self.assertEqual(await self.wrapper()('', '4', [], [], **self.data('normal')),1)
        self.assertLessEqual(len(list(Path('/proc/self/fd').iterdir())),before+1)
        self.assertEqual(len(self.assert_reaped()),6)


class NativeFailurePropagation(unittest.TestCase):
    def test_grouped_executor_stops_batch_instead_of_replacing_failed_sample(self):
        from areal.api.cli_args import GenerationHyperparameters,InferenceEngineConfig
        from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
        from areal.infra.workflow_executor import WorkflowExecutor
        from areal.workflow.rlvr import RLVRWorkflow
        from areal.utils.reward_status import RewardEvaluationError
        from rlvr_replay_fixture import FakeEngine
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(os.environ['FT_RLVR_TOKENIZER'],local_files_only=True)
        engine=FakeEngine(tokenizer);engine.get_version=lambda:3
        finished=[]
        class TrackedWorkflow(RLVRWorkflow):
            async def arun_episode(self,*args,**kwargs):
                from areal import workflow_context
                try:return await super().arun_episode(*args,**kwargs)
                finally:finished.append(workflow_context.get().sample_idx)
        wf=TrackedWorkflow(controlled_score,GenerationHyperparameters(n_samples=1),tokenizer)
        grouped=GroupedRolloutWorkflow(wf,8,logging.getLogger('strict-test'),drop_incomplete_group=True)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'attempts.jsonl'
            data={'messages':[{'role':'user','content':'What is 2+2?'}],
                  'answer':'4','record':str(path),'mode':'error'}
            class Loader:
                batch_size=1
                def __len__(self):return 1
                def __iter__(self):yield [data]
            executor=WorkflowExecutor(InferenceEngineConfig(consumer_batch_size=1,
                max_concurrent_rollouts=1,max_head_offpolicyness=0,check_trajectory_format=True),engine)
            executor.initialize(train_data_parallel_size=1)
            started=time.monotonic()
            try:
                with self.assertRaises(RuntimeError) as error:
                    executor.prepare_batch(Loader(),grouped)
                self.assertIsInstance(error.exception.__cause__,RewardEvaluationError)
                self.assertLess(time.monotonic()-started,10)
                self.assertEqual(executor.staleness_manager.get_stats().accepted,0)
                self.assertEqual(sorted(finished),list(range(8)))
            finally:executor.destroy()
            from scripts.ft.state import _exited
            self.assertTrue(all(_exited(json.loads(x)) for x in path.read_text().splitlines()))


if __name__=='__main__':unittest.main()
