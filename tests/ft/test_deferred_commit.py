"""R perf redesign D2a: deferred commit barrier (CPU, fake actor/DCP writer).

Container-only (imports megatron for the native snapshot fixture)."""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

from scripts.ft import state, training_adapter
from test_state import HASH, fixture, new_owner
from test_native_snapshot_r import fixture_state

MAIN = threading.main_thread()


class FakeCheckpointer:
    def __init__(self, runtime, seed=0, writer_seconds=0.2):
        self.runtime, self.seed, self.writer_seconds = runtime, seed, writer_seconds
        self.threads, self.fail_finalize, self.pending = [], False, []
        self._async_queue = types.SimpleNamespace(async_calls=collections.deque(), persistent=False)

    def generate_state_dict(self, with_optimizer, with_rng):
        self.threads.append(('generate', threading.current_thread()))
        return fixture_state(self.seed)

    def schedule(self, checkpoint):
        """Fork-like writer: a real child process that writes the shard, then exits."""
        native = checkpoint / 'native'
        native.mkdir()
        (checkpoint / 'recover').mkdir()
        (checkpoint / 'recover' / 'step_info.json').write_text('{}')
        script = ('import sys,time;time.sleep(float(sys.argv[2]));'
                  'open(sys.argv[1],"wb").write(bytes(range(256))*4096)')
        process = subprocess.Popen([sys.executable, '-c', script, str(native / '__0_0.distcp'),
                                    str(self.writer_seconds)])
        call = len(self.runtime.scheduled_ids)
        self._async_queue.async_calls.append(types.SimpleNamespace(
            async_caller=types.SimpleNamespace(process=process)))
        self.pending.append((call, process, native))
        self.runtime.scheduled_ids.append(call)

    def wait_async_saves(self):
        self.threads.append(('finalize', threading.current_thread()))
        while self.pending:
            call, process, native = self.pending.pop(0)
            self._async_queue.async_calls.popleft()
            if process.wait() != 0 or self.fail_finalize:
                raise RuntimeError('injected finalize failure')
            (native / '.metadata').write_bytes(b'metadata')
            self.runtime.finalized_ids.append(call)


class FakeActor:
    def __init__(self, runtime, **kwargs):
        self.checkpointer = FakeCheckpointer(runtime, **kwargs)

    def get_version(self):
        return 1


class DeferredCommit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.owner = new_owner(Path(self.tmp.name) / 'run' / 'state')
        self.addCleanup(self.owner.close)

    def runtime(self, **kwargs):
        runtime = training_adapter.Runtime.__new__(training_adapter.Runtime)
        runtime.root = Path(self.tmp.name) / 'run'
        runtime.owner, runtime.stop_after, runtime.pending = self.owner, None, None
        runtime.scheduled_ids, runtime.finalized_ids = [], []
        runtime.pin_first_commit, runtime.closed = False, False
        runtime.bridge = runtime.workflow = runtime.draw = None
        runtime.loader = types.SimpleNamespace(consumed=set())
        runtime.actor = FakeActor(runtime, **kwargs)
        runtime.gates = []
        runtime.writer_gate = lambda pending, generation=None: runtime.gates.append((pending, generation))
        return runtime

    def prepare(self, runtime, number=0, parent=None):
        intent, data = fixture(self.owner, number, parent)
        snapshot = f'p{number}'
        intent['data_snapshot_id'] = snapshot
        intent['components'] = {c: {'actor:0': ['native/', 'recover/', 'policy.json', 'native-state.json']}
                                 for c in training_adapter.COMPONENTS}
        runtime.generation = state.prepare_generation(self.owner, intent, data)
        runtime.update = {'physical_invocation_id': snapshot}
        runtime.optimizer_stats, runtime.scheduler_done = {'update_successful': 1.0}, True
        return runtime.generation

    def save(self, runtime, step=0):
        def dump(handler, engine, step_info):
            runtime.actor.checkpointer.schedule(runtime.io_checkpoint)
        step_info = types.SimpleNamespace(global_step=step)
        with patch.object(training_adapter.dataclasses, 'asdict', lambda value: {'global_step': value.global_step}):
            runtime.save(None, dump, runtime.actor, step_info)

    def events(self, runtime):
        path = runtime.root / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_save_returns_before_finalize_and_barrier_commits(self):
        runtime = self.runtime(writer_seconds=0.3)
        gid = self.prepare(runtime)
        self.save(runtime)
        self.assertIsNotNone(runtime.pending)
        self.assertFalse((self.owner.root / 'generations' / gid / 'token.json').exists())
        self.assertEqual(runtime.gates, [(True, None)])
        runtime.settle()
        directory = self.owner.root / 'generations' / gid
        self.assertTrue((directory / 'token.json').exists())
        self.assertEqual(runtime.gates[-1], (False, gid))
        signature = json.loads((directory / 'checkpoint' / 'native-state.json').read_text())
        expected = training_adapter.native_snapshot(runtime.actor)
        self.assertEqual(signature, json.loads(json.dumps(expected)))
        self.assertEqual([e['event'] for e in self.events(runtime)], ['committed'])
        self.assertEqual(state.select_recovery(self.owner)['generation'], gid)

    def test_collectives_and_state_capture_only_on_main_thread(self):
        runtime = self.runtime()
        self.prepare(runtime)
        self.save(runtime)
        runtime.settle()
        self.assertTrue(all(thread is MAIN for _, thread in runtime.actor.checkpointer.threads))
        self.assertEqual([name for name, _ in runtime.actor.checkpointer.threads], ['generate', 'finalize'])

    def test_speculative_shard_hash_reused(self):
        runtime = self.runtime(writer_seconds=0.0)
        gid = self.prepare(runtime)
        self.save(runtime)
        runtime.pending['hasher'].join(10)
        self.assertIn('native/__0_0.distcp', runtime.pending['hasher'].prehashed)
        mark = len(state.CONTENT_HASHED)
        with patch.object(state, 'CHUNK_THRESHOLD', 1 << 40):
            runtime.settle()
        manifest = json.loads((self.owner.root / 'generations' / gid / 'manifest.json').read_text())
        shard = self.owner.root / 'generations' / gid / 'checkpoint' / 'native' / '__0_0.distcp'
        self.assertEqual(manifest['files']['native/__0_0.distcp']['sha256'],
                         hashlib.sha256(shard.read_bytes()).hexdigest())
        self.assertEqual(len(state.CONTENT_HASHED) - mark, 1)  # only the small files at commit

    def test_changed_shard_after_speculation_is_rehashed(self):
        runtime = self.runtime(writer_seconds=0.0)
        gid = self.prepare(runtime)
        self.save(runtime)
        runtime.pending['hasher'].join(10)
        shard = self.owner.root / 'generations' / gid / 'checkpoint' / 'native' / '__0_0.distcp'
        data = shard.read_bytes()
        shard.unlink()
        shard.write_bytes(data[:-1] + b'x')
        runtime.settle()
        manifest = json.loads((self.owner.root / 'generations' / gid / 'manifest.json').read_text())
        self.assertEqual(manifest['files']['native/__0_0.distcp']['sha256'], hashlib.sha256(shard.read_bytes()).hexdigest())

    def test_finalize_failure_produces_no_token(self):
        runtime = self.runtime()
        gid = self.prepare(runtime)
        self.save(runtime)
        runtime.actor.checkpointer.fail_finalize = True
        with self.assertRaises(RuntimeError):
            runtime.settle()
        self.assertFalse((self.owner.root / 'generations' / gid / 'token.json').exists())
        self.assertIsNone(runtime.pending)
        self.assertIsNone(state.select_recovery(self.owner)['generation'])

    def test_hung_writer_does_not_deadlock_barrier_notify_path(self):
        runtime = self.runtime(writer_seconds=0.0)
        gid = self.prepare(runtime)
        with patch.object(training_adapter, '_writer_exited', lambda pid: False):
            self.save(runtime)
            done = threading.Event()
            def settle():
                runtime.settle(); done.set()
            worker = threading.Thread(target=settle); worker.start(); worker.join(20)
        self.assertTrue(done.is_set())
        self.assertTrue((self.owner.root / 'generations' / gid / 'token.json').exists())

    def test_begin_settles_first_and_abandon_leaves_uncommitted(self):
        runtime = self.runtime()
        gid = self.prepare(runtime)
        self.save(runtime)
        runtime.bridge = types.SimpleNamespace(begin_update=lambda trajectories, config: {'rows': []})
        runtime.config = types.SimpleNamespace(actor=None)
        with patch.object(training_adapter.dataclasses, 'asdict', lambda value: {}):
            runtime.begin([])
        self.assertTrue((self.owner.root / 'generations' / gid / 'token.json').exists())
        token = json.loads((self.owner.root / 'generations' / gid / 'token.json').read_text())
        parent = {'generation': gid, 'token_sha256': state._hash(state._bytes(token))}
        runtime.update = None
        second = self.prepare(runtime, 1, parent)
        self.save(runtime, step=1)
        runtime.abandon_pending()
        for _, process, _ in runtime.actor.checkpointer.pending:
            process.wait()
        self.assertIsNone(runtime.pending)
        self.assertFalse((self.owner.root / 'generations' / second / 'token.json').exists())
        self.assertEqual(state.select_recovery(self.owner)['generation'], gid)


class LeafHashing(unittest.TestCase):
    def test_memoryview_leaf_equals_shared_normalized(self):
        import numpy as np
        import torch
        values = [torch.randn(7, 5).t(), torch.arange(10, dtype=torch.int16), torch.randn(3).to(torch.bfloat16),
                  np.arange(12, dtype=np.float32).reshape(3, 4), np.arange(12, dtype=np.float32).reshape(3, 4).T,
                  torch.zeros(0)]
        for value in values:
            self.assertEqual(training_adapter._normalized_leaf_r(value), training_adapter.normalized(value))


if __name__ == '__main__':
    unittest.main()
