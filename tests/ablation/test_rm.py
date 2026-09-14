"""CPU contracts for the diagnostic RM; uses actual repository math scorer."""
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ablation_rm import Adapter, VARIANTS, load_source

LIMITS = dict(log_bytes=128 * 1024**2, db_bytes=128 * 1024**2,
              seen_ids=100000, rss_bytes=8 * 1024**3, events=200000)


class Client:
    def __init__(self):
        self.failed = False
        self.errors = []
    def event(self, kind, **fields):
        pass
    def fatal(self, code, **fields):
        self.failed = True
        self.errors.append(code)
    def check(self):
        if self.failed:
            raise RuntimeError('fatal latched')


def sample(i, group=0):
    return SimpleNamespace(group_index=group, index=i, rollout_id=0,
                           response='答案是 \\boxed{' + str(i % 2) + '}', label='1')


class RMTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, RTX_FAULT='none', RTX_FAULT_WINDOWS='', RTX_GROUP_RM='1',
                              RTX_SEAL='1', RTX_SEAL_AUTO_FIX='1', RTX_LOG_RESPONSE='0',
                              ABLATION_RUN_NONCE='cpu-contract-nonce')
        self.env.start()
        self.counter = 0
    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()
    def adapter(self, variant, client=None, **limits):
        self.counter += 1
        directory = self.root / str(self.counter)
        os.environ.update(RTX_RUN_DIR=str(directory), RTX_CAS_INDEX_DIR=str(directory / 'cas'))
        return Adapter(variant, client or Client(), {**LIMITS, **limits})
    def export(self, adapter):
        path = self.root / ('export-' + str(self.counter))
        manifest = adapter.finalize_export(path)
        for name, info in manifest['streams'].items():
            if 'sha256' in info:
                self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), info['sha256'])
        return path, manifest
    def records(self, path, name):
        return [json.loads(line) for line in (path / name).read_text().splitlines()]
    def canonical(self, records):
        return [{k: v for k, v in r.items() if k not in ('ts', 'response', 'label')} for r in records]

    def test_real_clean_reverse_and_unmodified_samples(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                adapter = self.adapter(variant)
                samples = [sample(i) for i in range(8)][::-1]
                before = copy.deepcopy([vars(s) for s in samples])
                result = asyncio.run(adapter.score(None, samples))
                self.assertEqual(result, [float(s.index % 2) for s in samples])
                self.assertEqual(before, [vars(s) for s in samples])
                path, manifest = self.export(adapter)
                self.assertEqual(manifest['cas_rows'], 0 if variant in ('O', 'LITE') else 8)
                records = self.records(path, 'rewards.jsonl')
                self.assertEqual(len(records), 8)
                if variant == 'PAYLOAD':
                    self.assertTrue(all('response' not in r and 'label' not in r for r in records))
                elif variant not in ('O', 'LITE'):
                    self.assertTrue(all('response' in r and 'label' in r for r in records))
                with self.assertRaises(RuntimeError):
                    asyncio.run(adapter.score(None, samples))

    def test_repeat_batch_duplicate_concurrent_group_state_matches_R(self):
        expected = None
        for variant in ('R', 'DBM', 'LOGM', 'BOTHM', 'PAYLOAD'):
            adapter = self.adapter(variant)
            async def exercise():
                group = [sample(i) for i in range(8)]
                first = await adapter.score(None, group + [group[0]])
                repeated = await adapter.score(None, group[::-1])
                concurrent = await asyncio.gather(*[adapter.score(None, [sample(i, g) for i in range(8)]) for g in (1, 2)])
                return first, repeated, concurrent
            values = asyncio.run(exercise())
            path, manifest = self.export(adapter)
            self.assertEqual(manifest['cas_rows'], 24)
            state = (values, {name: self.canonical(self.records(path, name))
                             for name in ('rewards.jsonl', 'seals.jsonl', 'cas_rejects.jsonl')})
            if expected is None:
                expected = state
            else:
                self.assertEqual(state, expected, variant)
            with sqlite3.connect(path / 'cas_index.sqlite3') as db:
                size = int(db.execute("SELECT value FROM cas_meta WHERE key='rewards_size'").fetchone()[0])
                self.assertEqual(size, (path / 'rewards.jsonl').stat().st_size)
                rows = [json.loads(r[0]) for r in db.execute('SELECT record_json FROM cas_claims ORDER BY logical_id')]
                if variant == 'PAYLOAD':
                    self.assertTrue(all('response' not in row and 'label' not in row for row in rows))

    def test_swallowed_reward_seal_reject_failures_latch(self):
        for stream in ('rewards', 'seals', 'cas_rejects'):
            adapter = self.adapter('LOGM')
            original = adapter.append_memory
            def fail(path, lines, lock=False):
                if Path(path).stem == stream:
                    raise MemoryError(stream)
                return original(path, lines, lock)
            adapter.append_memory = fail
            samples = [sample(i) for i in range(8)]
            if stream == 'cas_rejects':
                asyncio.run(adapter.score(None, samples))
            with self.assertRaises((RuntimeError, MemoryError)):
                asyncio.run(adapter.score(None, samples))
            self.assertTrue(adapter.failed)
            self.assertIn('append_' + stream, adapter.client.errors)
            adapter.buffers.clear()
            with self.assertRaises(RuntimeError):
                adapter.finalize_export(self.root / 'must-not-export')

    def test_capacity_and_connection_rebuild_fatal(self):
        for limit in ('log_bytes', 'db_bytes', 'seen_ids', 'rss_bytes', 'events'):
            adapter = self.adapter('BOTHM', **{limit: 1})
            with self.assertRaises((RuntimeError, MemoryError)):
                asyncio.run(adapter.score(None, [sample(i) for i in range(8)]))
            self.assertTrue(adapter.failed, limit)
        for variant in ('DBM', 'LOGM', 'BOTHM'):
            adapter = self.adapter(variant)
            asyncio.run(adapter.score(None, [sample(i) for i in range(8)]))
            adapter.rm._reset_cas_connection()
            with self.assertRaises(RuntimeError):
                asyncio.run(adapter.score(None, [sample(i, 1) for i in range(8)]))
            self.assertIn('connection_rebuild', adapter.client.errors)

    def test_commit_failure_owner_change_and_export_failure(self):
        adapter = self.adapter('BOTHM')
        asyncio.run(adapter.score(None, [sample(i) for i in range(8)]))
        conn = adapter.connection
        raw_connection = conn.conn
        class Broken:
            def __getattr__(self, key):
                return getattr(raw_connection, key)
            def commit(self):
                raise sqlite3.OperationalError('injected')
        conn.conn = Broken()
        with self.assertRaises(sqlite3.Error):
            asyncio.run(adapter.score(None, [sample(i, 1) for i in range(8)]))
        self.assertIn('sqlite_commit', adapter.client.errors)
        adapter = self.adapter('R')
        adapter.pid -= 1
        with self.assertRaises(RuntimeError):
            asyncio.run(adapter.score(None, [sample(0)]))
        self.assertIn('rm_owner_changed', adapter.client.errors)
        adapter = self.adapter('O')
        asyncio.run(adapter.score(None, [sample(0)]))
        with patch('ablation_rm.os.fsync', side_effect=OSError('injected export')):
            with self.assertRaises(OSError):
                adapter.finalize_export(self.root / 'bad-export')
        self.assertIn('export_failure', adapter.client.errors)
        self.assertFalse((self.root / 'bad-export' / 'export_manifest.json').exists())

    def test_oracle_swallowed_json_and_file_failure(self):
        adapter = self.adapter('O')
        with patch('ablation_rm.json.dumps', side_effect=MemoryError('encoding')):
            with self.assertRaises(RuntimeError):
                asyncio.run(adapter.score(None, [sample(0)]))
        self.assertTrue(adapter.failed)
        self.assertIn('json_encode', adapter.client.errors)
        adapter = self.adapter('O')
        import io
        class BrokenFile(io.StringIO):
            def write(self, data):
                raise MemoryError('write')
        with patch('ablation_rm.builtins.open', return_value=BrokenFile()):
            with self.assertRaises(RuntimeError):
                asyncio.run(adapter.score(None, [sample(0)]))
        self.assertIn('file_body_exception', adapter.client.errors)

    def test_only_requested_cancellation_is_nonfatal(self):
        for requested in (False, True):
            adapter = self.adapter('R')
            async def cancel_waiting_score():
                await adapter.rm._lock.acquire()
                task = asyncio.create_task(adapter.score(None, [sample(0)]))
                await asyncio.sleep(0)
                if requested:
                    adapter.begin_cancellation()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                adapter.rm._lock.release()
            asyncio.run(cancel_waiting_score())
            self.assertEqual(adapter.failed, not requested)

    def test_oracle_and_lite_sampled_write_and_close_metrics(self):
        for variant in ('O', 'LITE'):
            adapter = self.adapter(variant)
            asyncio.run(adapter.score(None, [sample(i) for i in range(8)]))
            self.assertEqual(adapter.metrics['write'][0], 8)
            self.assertEqual(adapter.metrics['close_including_flush'][0], 8)
            asyncio.run(adapter.score(None, [sample(i, 1) for i in range(8)]))
            self.assertEqual(adapter.metrics['write'][0], 8)
            self.assertEqual(adapter.metrics['close_including_flush'][0], 8)

    def test_actual_rm_identity_is_captured_before_later_sample_changes(self):
        from ablation_rm import RM_ATTEMPT_ID
        adapter = self.adapter('R')
        samples = [sample(i) for i in range(8)]
        expected = [f'{s.group_index}:{s.index}:{s.rollout_id}' for s in samples]
        token = RM_ATTEMPT_ID.set('actual-attempt')
        try:
            asyncio.run(adapter.score(None, samples))
        finally:
            RM_ATTEMPT_ID.reset(token)
        event = next(e for e in adapter.events if e['kind'] == 'reward_start')
        self.assertEqual(event['attempt_id'], 'actual-attempt')
        self.assertTrue(all(isinstance(identity, tuple) for identity in event['ids']))
        for s in samples:
            s.index += 10000
            s.rollout_id = 42
        path, _ = self.export(adapter)
        captured = next(e for e in self.records(path, 'events.jsonl') if e['kind'] == 'reward_start')
        self.assertEqual(captured['ids'], expected)

    def test_memory_log_counter_checked_against_exported_buffers(self):
        adapter = self.adapter('LOGM')
        asyncio.run(adapter.score(None, [sample(i) for i in range(8)]))
        self.assertEqual(adapter.memory_log_bytes, sum(map(len, adapter.buffers.values())))
        next(iter(adapter.buffers.values())).extend(b'x')
        with self.assertRaises(RuntimeError):
            adapter.finalize_export(self.root / 'corrupted')
        self.assertIn('memory_log_accounting', adapter.client.errors)

    def test_explicit_proxy_signatures_preserve_original_arguments(self):
        adapter = self.adapter('R')
        obj = {'text': '中文', 'items': [1, 2]}
        for sampled in (False, True):
            adapter.sample_group = sampled
            self.assertEqual(adapter.dumps(obj), json.dumps(obj))
            self.assertEqual(adapter.dumps(obj, ensure_ascii=False, separators=(',', ':')),
                             json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
        path = adapter.run_dir / 'signature.txt'
        with adapter.open(path, 'a') as stream:
            stream.write('ok')
        with adapter.open(path) as stream:
            self.assertEqual(stream.read(), 'ok')
        path.write_bytes(b'bad\xff')
        with adapter.open(path, errors='replace') as stream:
            self.assertEqual(stream.read(), 'bad\ufffd')
        fields = {'attempt_id': 17}
        adapter.event('signature', **fields)
        self.assertEqual(fields, {'attempt_id': 17})
        self.assertEqual(adapter.events[-1]['attempt_id'], 17)
        self.assertEqual(adapter.events[-1]['kind'], 'signature')

    def test_original_calls_remain_source_functions(self):
        import builtins
        original_import = builtins.__import__
        adapter = self.adapter('R')
        self.assertIs(builtins.__import__, original_import)
        self.assertIsNot(adapter.rm.__builtins__, vars(builtins))
        original = load_source('phase2_seal_rm')
        self.assertEqual(adapter.rm.rm_function.__code__, original.rm_function.__code__)
        self.assertEqual(adapter.rm._rm_batch_group.__code__, original._rm_batch_group.__code__)
        self.assertEqual(adapter.rm._cas_write_many.__code__, original._cas_write_many.__code__)

    @unittest.skipUnless(os.environ.get('ABLATION_FULL_PAYLOAD') == '1', 'explicit CPU real-payload gate')
    def test_all_576_pilot_payloads(self):
        import torch
        root = Path(__file__).resolve().parents[2]
        manifest = json.loads((root / 'runs/e7_restart_0.5B_20260911_pilot.json').read_text())
        payloads = []
        for run in manifest['runs']:
            for step in (0, 249, 499):
                raw = torch.load(root / run / f'rollout_debug/{step}.pt', map_location='cpu', weights_only=False)
                payloads.append(raw['samples'])
        self.assertEqual(sum(map(len, payloads)), 576)
        for variant in VARIANTS:
            for batch in payloads:
                adapter = self.adapter(variant)
                samples = [SimpleNamespace(**copy.deepcopy(s)) for s in batch]
                expected = [s['reward'] for s in batch]
                before = copy.deepcopy([vars(s) for s in samples])
                values = asyncio.run(adapter.score(None, samples))
                self.assertEqual(values, expected, variant)
                def unchanged(actual, original):
                    if isinstance(actual, torch.Tensor):
                        torch.testing.assert_close(actual, original, rtol=0, atol=0, equal_nan=True)
                    elif isinstance(actual, dict):
                        self.assertEqual(actual.keys(), original.keys())
                        for key in actual:
                            unchanged(actual[key], original[key])
                    elif isinstance(actual, (list, tuple)):
                        self.assertEqual(len(actual), len(original))
                        for a, b in zip(actual, original):
                            unchanged(a, b)
                    else:
                        self.assertEqual(actual, original)
                unchanged([vars(s) for s in samples], before)
                reverse = asyncio.run(adapter.score(None, samples[::-1]))
                self.assertEqual(reverse[::-1], expected, variant)
                self.export(adapter)


def benchmark(observation_only=False, live=False, heartbeat_interval=.05, randomized_phase=False):
    """Randomized A/B/C CPU gate: bare original, safety-only, fully observed.

    One timed cell scores 72 complete groups/576 samples, preserving first-call setup
    and fixed 1/32 whole-call detailed sampling. Fresh logical-ID namespace per source batch.
    Live control benchmarks use a fresh real supervisor and heartbeat client
    for each B/C cell; A has neither. Cleanup deliberately terminates the
    benchmark supervisor, never publishes training success.
    """
    import random
    import importlib.util
    import statistics
    import subprocess
    import sys
    import time
    import torch
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / 'runs/e7_restart_0.5B_20260911_pilot.json').read_text())
    input_paths = [root / run / f'rollout_debug/{step}.pt'
                   for run in manifest['runs'] for step in (0, 249, 499)]
    batches = [torch.load(path, map_location='cpu', weights_only=False)['samples'] for path in input_paths]
    assert sum(map(len, batches)) == 576
    image_id = os.environ.get('ABLATION_BENCH_IMAGE_ID')
    if live and randomized_phase and not (image_id and image_id.startswith('sha256:') and len(image_id) == 71):
        raise ValueError('production benchmark requires full ABLATION_BENCH_IMAGE_ID')
    rng = random.Random(20260911)
    phase_rng = random.Random(20260912)
    report = {'paired_order_seed': 20260911, 'scope': 'randomized_three_level' if live else 'observation_on_off' if observation_only else 'complete_vs_bare',
              'control_client': 'real supervisor' if live else 'in_process_fake',
              'heartbeat_seconds': heartbeat_interval if live else None,
              'randomized_heartbeat_phase': randomized_phase, 'phase_rng_seed': 20260912,
              'phase_distribution': 'uniform[0, heartbeat_seconds)' if randomized_phase else 'none',
              'samples_per_cell': 576, 'original_sample_entries': 576, 'input_batches': 18,
              'blocks_per_variant': 21, 'warmup_blocks': 3, 'timing_sample_period_rm_calls': 32,
              'rm_calls_per_cell': 72, 'samples_per_call': 8, 'variants': {}, 'image_id': image_id,
              'input_debug': [dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in input_paths],
              'logical_id_transform': 'group_index and index += input_batch_ordinal * 1000000; scoring payload and original group membership unchanged',
              'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [root / 'scripts/ablation_rm.py', root / 'scripts/day2_custom_rm.py',
                                          root / 'scripts/phase2_seal_rm.py', Path(__file__)]}}
    fixture = RMTest()
    fixture.setUp()
    if live:
        control_source = root / 'scripts/ablation_control.py'
        control_snapshot = fixture.root / 'ablation_control_snapshot.py'
        control_snapshot.write_bytes(control_source.read_bytes())
        control_snapshot.chmod(0o444)
        control_sha = hashlib.sha256(control_snapshot.read_bytes()).hexdigest()
        report['control_snapshot_sha256'] = control_sha
        report['control_source_path'] = str(control_source.relative_to(root))
        report['source_sha256'][str(control_source.relative_to(root))] = control_sha
        spec = importlib.util.spec_from_file_location('_benchmark_control_snapshot', control_snapshot)
        control_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = control_module
        spec.loader.exec_module(control_module)
        LiveClient = control_module.Client
        print('immutable control snapshot ' + control_sha, file=sys.stderr, flush=True)
    try:
        for variant in (VARIANTS if live or observation_only else ('O', 'R')):
            timings, all_blocks = [], []
            for repetition in range(21):
                pair = {}
                order = ['safety', 'observed'] if live or observation_only else ['bare', 'observed']
                if live and variant in ('O', 'R'):
                    order.append('bare')
                rng.shuffle(order)
                for mode in order:
                    process, client = None, None
                    if live and mode != 'bare':
                        socket_path = fixture.root / f'control-{variant}-{repetition}-{mode}.sock'
                        process = subprocess.Popen([sys.executable, str(control_snapshot),
                                                    'serve', '--socket', str(socket_path), '--nonce', 'cpu-contract-nonce',
                                                    '--output', str(socket_path) + '.jsonl'], stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.PIPE)
                        deadline = time.monotonic() + 10
                        while not socket_path.exists():
                            if process.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError('benchmark supervisor startup failed')
                            time.sleep(.01)
                        client = LiveClient(str(socket_path), 'cpu-contract-nonce', 'rm-worker', heartbeat_interval=heartbeat_interval)
                        if randomized_phase:
                            time.sleep(phase_rng.random() * heartbeat_interval)
                    try:
                        adapter = fixture.adapter(variant, client=client)
                        adapter.observations = mode != 'safety'
                        module = load_source('day2_custom_rm' if variant == 'O' else 'phase2_seal_rm')
                        module.GROUP_RM, module.SEAL = True, variant == 'R'
                        cells, expected = [], []
                        for ordinal, raw in enumerate(batches):
                            samples = [SimpleNamespace(**copy.deepcopy(s)) for s in raw]
                            for sample in samples:
                                sample.group_index += ordinal * 1000000
                                sample.index += ordinal * 1000000
                            groups = {}
                            for sample in samples:
                                groups.setdefault(sample.group_index, []).append(sample)
                            for group in groups.values():
                                assert len(group) == 8
                                cells.append(group)
                                expected.append([s.reward for s in group])
                        async def measure():
                            start = time.perf_counter_ns()
                            results = []
                            for samples in cells:
                                results.append(await (module.rm_function(None, samples) if mode == 'bare' else adapter.score(None, samples)))
                            return time.perf_counter_ns() - start, results
                        elapsed, result = asyncio.run(measure())
                        if mode == 'bare' and variant == 'R' and module._cas_conn is not None:
                            module._reset_cas_connection()
                        pair[mode] = elapsed
                        assert result == expected
                        if client:
                            client.check()
                            assert process.poll() is None
                    finally:
                        if client:
                            client.close_transport()
                        if process:
                            process.terminate()
                            process.communicate(timeout=5)
                all_blocks.append(dict(block=repetition, warmup=repetition < 3, order=order, timings_ns=pair))
                if repetition >= 3:
                    timings.append(pair)
            reference = 'safety' if live or observation_only else 'bare'
            ratio = statistics.median(p['observed'] / p[reference] - 1 for p in timings)
            row = {'all_blocks': all_blocks, 'pairs': timings, 'median_observation_fraction': ratio, 'observation_gate_5_percent': ratio <= .05}
            if 'bare' in timings[0]:
                total = statistics.median(p['observed'] / p['bare'] - 1 for p in timings)
                row.update(median_total_fraction=total, total_gate_5_percent=total <= .05)
            report['variants'][variant] = row
            print('completed benchmark variant ' + variant, file=sys.stderr, flush=True)
        report['passed'] = all(r['observation_gate_5_percent'] and r.get('total_gate_5_percent', True)
                               for r in report['variants'].values())
        print(json.dumps(report, indent=2))
    finally:
        fixture.tearDown()


if __name__ == '__main__':
    import sys
    if '--benchmark-live-production' in sys.argv:
        benchmark(live=True, heartbeat_interval=1.0, randomized_phase=True)
    elif '--benchmark-live' in sys.argv:
        benchmark(live=True)
    elif '--benchmark-observation' in sys.argv:
        benchmark(observation_only=True)
    elif '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
