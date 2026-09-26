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


class FakeQueue:
    """AsyncCallsQueue stand-in: a real child process per call; finalize writes .metadata."""
    def __init__(self, writer_seconds):
        self.writer_seconds, self.async_calls, self.persistent = writer_seconds, collections.deque(), False
        self.fail_finalize, self.threads, self.next = False, [], 0

    def schedule_async_request(self, native):
        script = ('import sys,time;time.sleep(float(sys.argv[2]));'
                  'open(sys.argv[1],"wb").write(bytes(range(256))*4096)')
        process = subprocess.Popen([sys.executable, '-c', script, str(native / '__0_0.distcp'),
                                    str(self.writer_seconds)])
        call, self.next = self.next, self.next + 1
        self.async_calls.append(types.SimpleNamespace(call=call, native=native,
            async_caller=types.SimpleNamespace(process=process)))
        return call

    def maybe_finalize_async_calls(self, blocking=False):
        self.threads.append(threading.current_thread())
        done = []
        while self.async_calls:
            item = self.async_calls[0]
            process = item.async_caller.process
            if not blocking and process.poll() is None:
                break
            if process.wait() != 0 or self.fail_finalize:
                raise RuntimeError('injected finalize failure')
            (item.native / '.metadata').write_bytes(b'metadata')
            self.async_calls.popleft()
            done.append(item.call)
        return done


class FakeCheckpointer:
    def __init__(self, seed=0, writer_seconds=0.2):
        self.seed, self.threads, self.states = seed, [], []
        self._async_queue = FakeQueue(writer_seconds)

    def generate_state_dict(self, with_optimizer, with_rng):
        self.threads.append(threading.current_thread())
        self.states.append(fixture_state(self.seed))
        return self.states[-1]

    def wait_async_saves(self):
        self._async_queue.maybe_finalize_async_calls(blocking=True)


class FakeActor:
    def __init__(self, **kwargs):
        self.checkpointer = FakeCheckpointer(**kwargs)

    def get_version(self):
        return 1


class Pipelined(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.owner = new_owner(Path(self.tmp.name) / 'run' / 'state')
        self.addCleanup(self.owner.close)
        self.addCleanup(self.reap)
        self.runtimes = []

    def reap(self):
        for runtime in self.runtimes:
            runtime.abandon_pending()
            for item in list(runtime.actor.checkpointer._async_queue.async_calls):
                item.async_caller.process.wait()

    def runtime(self, **kwargs):
        runtime = training_adapter.Runtime.__new__(training_adapter.Runtime)
        runtime.root = Path(self.tmp.name) / 'run'
        runtime.owner, runtime.stop_after, runtime.pending = self.owner, None, None
        runtime.generation = runtime.update = None
        runtime.scheduled_ids, runtime.finalized_ids, runtime.calls = [], [], {}
        runtime.pin_first_commit, runtime.closed, runtime.prune_thread = False, False, None
        runtime.bridge = runtime.workflow = runtime.draw = None
        runtime.loader = types.SimpleNamespace(consumed=set())
        runtime.actor = FakeActor(**kwargs)
        runtime.gates = []
        runtime.writer_gate = lambda pending, generation=None: runtime.gates.append(
            (pending, runtime.generation if generation is None else generation))
        runtime._wrap_queue(runtime.actor.checkpointer._async_queue)
        self.runtimes.append(runtime)
        return runtime

    def gdir(self, gid):
        return self.owner.root / 'generations' / gid

    def prepare(self, runtime, number):
        intent, data = fixture(self.owner, number, None)
        consumed = sorted(runtime.loader.consumed) + [f'group{number}:0']
        data.update(drawn=consumed, consumed=consumed, cursor=len(consumed))
        if runtime.pending is not None:
            raw = (self.gdir(runtime.pending['generation']) / 'intent.json').read_bytes()
            intent['parent'] = {'generation': runtime.pending['generation'], 'intent_sha256': state._hash(raw)}
        else:
            with state._locked(self.owner) as control:
                intent['parent'], _ = state._head(self.owner, control)
        snapshot = f'p{number}'
        intent['data_snapshot_id'] = snapshot
        intent['components'] = {c: {'actor:0': ['native/', 'recover/', 'policy.json', 'native-state.json']}
                                for c in training_adapter.COMPONENTS}
        runtime.generation = state.prepare_generation(self.owner, intent, data)
        runtime.loader.consumed |= {f'group{number}:0'}
        runtime.update = {'physical_invocation_id': snapshot}
        runtime.optimizer_stats, runtime.scheduler_done = {'update_successful': 1.0}, True
        return runtime.generation

    def save(self, runtime, step):
        def dump(handler, engine, step_info):
            native = runtime.io_checkpoint / 'native'
            native.mkdir()
            (runtime.io_checkpoint / 'recover').mkdir()
            (runtime.io_checkpoint / 'recover' / 'step_info.json').write_text('{}')
            runtime.actor.checkpointer._async_queue.schedule_async_request(native)
        with patch.object(training_adapter.dataclasses, 'asdict', lambda value: {'global_step': value.global_step}):
            runtime.save(None, dump, runtime.actor, types.SimpleNamespace(global_step=step))

    def token(self, gid):
        return (self.gdir(gid) / 'token.json').exists()

    def events(self, runtime):
        path = runtime.root / 'events.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_lag1_sequence_commits_at_next_save(self):
        runtime = self.runtime()
        g0 = self.prepare(runtime, 0)
        self.save(runtime, 0)
        self.assertFalse(self.token(g0))
        self.assertTrue((self.gdir(g0) / 'receipts').exists())  # optimizer evidence already durable
        self.assertTrue((self.gdir(g0) / 'checkpoint' / 'policy.json').exists())
        g1 = self.prepare(runtime, 1)
        runtime.join_snapshot()  # B1 before optimizer
        self.assertTrue((self.gdir(g0) / 'checkpoint' / 'native-state.json').exists())
        self.save(runtime, 1)  # B2 commits g0 first
        self.assertTrue(self.token(g0))
        self.assertFalse(self.token(g1))
        runtime.settle()
        runtime.join_prune()
        self.assertTrue(self.token(g1))
        self.assertEqual([e['generation'] for e in self.events(runtime) if e['event'] == 'committed'], [g0, g1])
        self.assertEqual(runtime.loader.consumed, {'group0:0', 'group1:0'})
        self.assertEqual(state.select_recovery(self.owner)['generation'], g1)
        # Single writer slot: g0 released before g1 was bound.
        self.assertEqual([g for _, g in runtime.gates], [g0, g0, g1, g1])
        self.assertEqual([p for p, _ in runtime.gates], [True, False, True, False])

    def test_collectives_and_capture_only_on_main_thread(self):
        runtime = self.runtime()
        self.prepare(runtime, 0); self.save(runtime, 0)
        self.prepare(runtime, 1); runtime.join_snapshot(); self.save(runtime, 1)
        runtime.settle(); runtime.join_prune()
        threads = runtime.actor.checkpointer.threads + runtime.actor.checkpointer._async_queue.threads
        self.assertTrue(threads and all(thread is MAIN for thread in threads))

    def test_f2_shape_crash_promotes_predecessor(self):
        runtime = self.runtime(writer_seconds=0.0)
        g0 = self.prepare(runtime, 0); self.save(runtime, 0)
        g1 = self.prepare(runtime, 1)
        runtime.join_snapshot()                    # B1 inside the optimizer wrapper
        runtime.actor.checkpointer.wait_async_saves()  # fault hook drain, then SIGKILL
        self.assertFalse(self.token(g0))
        runtime.pending['hasher'].join(10)
        result = state.select_recovery(self.owner)
        self.assertEqual((result['generation'], result['promoted']), (g0, [g0]))
        self.assertTrue((self.gdir(g1) / 'abandoned.json').exists())

    def test_crash_before_snapshot_published_abandons(self):
        runtime = self.runtime(writer_seconds=0.0)
        release = threading.Event()
        original = training_adapter._Hasher.publish
        def slow(hasher):
            release.wait(20); original(hasher)
        with patch.object(training_adapter._Hasher, 'publish', slow):
            g0 = self.prepare(runtime, 0); self.save(runtime, 0)
            runtime.actor.checkpointer.wait_async_saves()
            result = state.select_recovery(self.owner)
            release.set()
        self.assertIsNone(result['generation'])
        self.assertIn('missing', json.loads((self.gdir(g0) / 'abandoned.json').read_text())['reason'])

    def test_crash_before_finalize_abandons(self):
        runtime = self.runtime(writer_seconds=0.0)
        g0 = self.prepare(runtime, 0); self.save(runtime, 0)
        runtime.join_snapshot()
        self.assertIsNone(state.select_recovery(self.owner)['generation'])
        self.assertTrue((self.gdir(g0) / 'abandoned.json').exists())

    def test_finalize_failure_writes_no_evidence(self):
        runtime = self.runtime()
        g0 = self.prepare(runtime, 0); self.save(runtime, 0)
        runtime.actor.checkpointer._async_queue.fail_finalize = True
        with self.assertRaises(RuntimeError):
            runtime.settle()
        self.assertFalse((self.gdir(g0) / 'receipts' / ('rank-' + state._hash(b'actor:0') + '.json')).exists())
        self.assertFalse(self.token(g0))
        self.assertEqual([p for p, _ in runtime.gates], [True])

    def test_rng_and_scheduler_captured_by_value(self):
        runtime = self.runtime()
        g0 = self.prepare(runtime, 0)
        release = threading.Event()
        original = training_adapter._Hasher.publish
        def slow(hasher):
            release.wait(20); original(hasher)
        with patch.object(training_adapter._Hasher, 'publish', slow):
            self.save(runtime, 0)
            captured = runtime.actor.checkpointer.states[-1]
            rng = captured['rng_state'].data[0]
            rng['torch_rng_state'].add_(1)
            rng['rng_tracker_states']['model-parallel-rng'].add_(1)
            captured['lr_scheduler']['last_epoch'] = 99
            captured['optimizer']['param_groups'][0]['lr'] = 5.0
            release.set()
            runtime.join_snapshot()
        published = json.loads((self.gdir(g0) / 'checkpoint' / 'native-state.json').read_text())
        expected = training_adapter._check_snapshot(training_adapter.normalized(fixture_state(0)))
        self.assertEqual(published, json.loads(json.dumps(expected)))

    def test_b1_recomputes_when_hasher_fails(self):
        runtime = self.runtime()
        g0 = self.prepare(runtime, 0)
        with patch.object(training_adapter._Hasher, 'publish', side_effect=RuntimeError('boom')):
            self.save(runtime, 0)
            runtime.join_snapshot()
        published = json.loads((self.gdir(g0) / 'checkpoint' / 'native-state.json').read_text())
        expected = training_adapter._check_snapshot(training_adapter.normalized(fixture_state(0)))
        self.assertEqual(published, json.loads(json.dumps(expected)))

    def test_hung_writer_does_not_deadlock(self):
        runtime = self.runtime(writer_seconds=0.0)
        g0 = self.prepare(runtime, 0)
        with patch.object(training_adapter, '_writer_exited', lambda pid: False):
            self.save(runtime, 0)
            done = threading.Event()
            worker = threading.Thread(target=lambda: (runtime.settle(), done.set()))
            worker.start(); worker.join(20)
        self.assertTrue(done.is_set() and self.token(g0))

    def test_background_prune_with_concurrent_prepare(self):
        runtime = self.runtime(writer_seconds=0.0)
        gids = []
        for number in range(6):
            gids.append(self.prepare(runtime, number))
            if number:
                runtime.join_snapshot()
            self.save(runtime, number)
        runtime.settle(); runtime.join_prune()
        self.assertTrue(all(self.token(g) for g in gids))
        pruned = [g for g in gids if (self.gdir(g) / 'pruned.json').exists()]
        self.assertEqual(pruned, gids[:-2])
        self.assertEqual(state.select_recovery(self.owner)['generation'], gids[-1])


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
