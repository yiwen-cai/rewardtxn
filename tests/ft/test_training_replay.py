"""CPU lineage/consumption checks; GPU integration is a separate probe."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.ft import state


class DirectoryMapping(unittest.TestCase):
    def test_unresolved_native_writer_blocks_before_owner_takeover(self):
        from scripts.ft.training_adapter import Runtime
        with tempfile.TemporaryDirectory() as tmp:
            runtime = Runtime.__new__(Runtime)
            runtime.root = Path(tmp)
            (runtime.root / 'writer.json').write_text(json.dumps({'pending': True,
                'pid_namespace': os.stat('/proc/self/ns/pid').st_ino}))
            with self.assertRaisesRegex(RuntimeError, 'pending writer lacks bound owner/job evidence'):
                runtime.make_loader(None)
            self.assertFalse((runtime.root / 'state').exists())

    def test_consumption_comes_only_from_retained_chain_and_repacks_groups(self):
        from test_state import new_owner, COMPONENTS, HASH
        from replay_fixture import opened
        from scripts.ft.training_replay import RetainedLoader
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with opened(root / 'draw') as draw:
                original = next(draw)
                next(draw)  # durable prefetch beyond the commit snapshot
            with new_owner(root / 'state') as owner:
                groups = []
                for item in original[:2]:
                    slot = item['_r_draw']; entries = []
                    for i in range(8):
                        sample = slot['group_id'] + ':' + str(i)
                        attempt = state.authorize_attempt(owner, sample, None, 'a', expected_policy_version=0)
                        receipt = state.accept_result(owner, attempt, {'response_sha256': HASH,
                            'reward_sha256': HASH, 'tensor_input_sha256': HASH, **attempt['versions']})
                        entries.append({'sample_index': i, 'sample': sample, 'receipt': receipt})
                    prompt = {'messages': item['messages']}
                    groups.append({'logical_group_id': slot['group_id'], 'k': 8, 'samples': entries,
                        'prompt': prompt, 'prompt_sha256': state._hash(state._bytes(prompt))})
                consumed = [e['sample'] for g in groups for e in g['samples']]
                with opened(root / 'draw') as draw:
                    loader = RetainedLoader(draw, owner, lambda item: None)
                    data = loader.snapshot(consumed)
                intent = {'parent': None, 'ack_capability': 'none', 'config_sha256': HASH,
                    'expected_ranks': ['actor:0'], 'data_snapshot_id': 'snapshot',
                    'components': {c: {'actor:0': ['native/']} for c in COMPONENTS},
                    'updates': [{'logical_update_id': 'u', 'physical_update_id': 'p',
                                 'train_input_sha256': HASH, 'groups': groups}]}
                gid = state.prepare_generation(owner, intent, data)
                with opened(root / 'draw') as draw:
                    loader = RetainedLoader(draw, owner, lambda item: None)
                    self.assertEqual([x['source_row_id'] for x in next(iter(loader))], [0, 1, 2, 3])
                path = owner.root / 'generations' / gid / 'checkpoint/native'
                path.mkdir(parents=True); (path / 'fixture').write_bytes(b'not a backend state')
                state.record_evidence(owner, gid, {'kind': 'optimizer', 'snapshot_id': 'snapshot',
                    'physical_updates': ['p'], 'successful': True, 'scheduler_applied': True})
                state.record_evidence(owner, gid, {'kind': 'finalize', 'snapshot_id': 'snapshot',
                    'rank': 'actor:0', 'writer_closed': True})
                state.commit_generation(owner, gid)
                with opened(root / 'draw') as draw:
                    loader = RetainedLoader(draw, owner, lambda item: None)
                    self.assertEqual([x['source_row_id'] for x in next(iter(loader))], [2, 3, 4, 5])

    def test_native_shards_are_all_hashed_and_corruption_rejected(self):
        from test_state import new_owner, fixture, COMPONENTS
        with tempfile.TemporaryDirectory() as tmp, new_owner(Path(tmp)) as owner:
            intent, data = fixture(owner)
            intent['components'] = {c: {'actor:0': ['native/']} for c in COMPONENTS}
            gid = state.prepare_generation(owner, intent, data)
            path = owner.root / 'generations' / gid / 'checkpoint/native'
            path.mkdir(parents=True)
            (path / '__0_0.distcp').write_bytes(b'CPU fixture, not native model state')
            state.record_evidence(owner, gid, {'kind': 'optimizer', 'snapshot_id': 'cut0',
                'physical_updates': ['p0'], 'successful': True, 'scheduler_applied': True})
            with self.assertRaises(state.StateError):
                state.commit_generation(owner, gid)
            state.record_evidence(owner, gid, {'kind': 'finalize', 'snapshot_id': 'cut0',
                'rank': 'actor:0', 'writer_closed': True})
            state.commit_generation(owner, gid)
            (path / '__0_0.distcp').write_bytes(b'corrupted')
            with self.assertRaises(state.StateError):
                state.select_recovery(owner)


def stage(root, version=3):
    from transformers import AutoTokenizer
    from areal import workflow_context
    from areal.api.cli_args import GenerationHyperparameters
    from areal.infra.workflow_context import WorkflowContext
    from replay_fixture import opened
    from rlvr_replay_fixture import counted_gsm8k, FakeEngine, asset_hash, gsm8k_reward_fn
    from scripts.ft.training_replay import TrainingRLVR, TrainingBridge
    root = Path(root)
    tokenizer_path = os.environ['FT_RLVR_TOKENIZER']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    verifier = hashlib.sha256(Path(gsm8k_reward_fn.__code__.co_filename).read_bytes()).hexdigest()
    path = root / 'state/control.json'
    control = json.loads(path.read_text()) if path.exists() else None
    owner = state.acquire_owner(root / 'state', -1 if control is None else control['epoch'],
        None if control is None else control['processes'], run_nonce='cpu-run', config_sha256='b'*64,
        verifier_version=verifier)
    wf = TrainingRLVR(owner=owner, root=root / 'artifacts', attempts={}, reward_fn=counted_gsm8k,
        tokenizer_sha256=asset_hash(tokenizer_path), gconfig=GenerationHyperparameters(n_samples=1), tokenizer=tokenizer)
    wf.current_version = lambda: version
    wf.max_lag = 2
    bridge = TrainingBridge(wf, root / 'bridge', policy_version=version, version=lambda: version, max_lag=2)
    try:
        with opened(root / 'draw') as draw:
            data = dict(next(draw)[0], answer='4', _cpu_log=str(root / 'scores.jsonl'))
        bridge.authorize(data)
        class VersionedEngine(FakeEngine):
            async def agenerate(self, request):
                response = await super().agenerate(request)
                response.output_versions = [version] * len(response.output_tokens)
                return response
        engine = VersionedEngine(tokenizer)
        async def invoke():
            workflow_context.set(WorkflowContext(task_id=99, sample_idx=0))
            result = await bridge.arun_episode(engine, data)
            return bridge.validate_rows(result)
        rows = asyncio.run(invoke())
        (root / f'epoch-{owner.epoch}.json').write_text(json.dumps({'generation_calls': engine.calls, 'rows': rows}))
    finally:
        bridge.close(); wf.close(); owner.close()


class Adoption(unittest.TestCase):
    def test_future_policy_artifact_is_regenerated_after_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            for version in (3, 2):
                subprocess.run([sys.executable, __file__, '--stage', tmp, str(version)], check=True, timeout=90)
            root = Path(tmp)
            self.assertEqual(json.loads((root / 'epoch-1.json').read_text())['generation_calls'], [0])
            self.assertEqual(len((root / 'scores.jsonl').read_text().splitlines()), 2)

    def test_two_new_processes_adopt_without_generation_or_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(3):
                subprocess.run([sys.executable, __file__, '--stage', tmp], check=True, timeout=90)
            root = Path(tmp)
            epochs = [json.loads((root / f'epoch-{i}.json').read_text()) for i in range(3)]
            self.assertEqual(epochs[0]['generation_calls'], [0])
            self.assertEqual(epochs[1]['generation_calls'], [])
            self.assertEqual(epochs[2]['generation_calls'], [])
            self.assertEqual(len((root / 'scores.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len({e['rows'][0]['sample'] for e in epochs}), 1)
            self.assertEqual([e['rows'][0]['attempt']['epoch'] for e in epochs], [0, 1, 2])
            self.assertEqual(len({e['rows'][0]['identity'] for e in epochs}), 3)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--stage':
        stage(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        unittest.main()
