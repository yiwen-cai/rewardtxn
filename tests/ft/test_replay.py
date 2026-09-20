"""CPU storage contracts, including actual process death; no GPU replay claims."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from scripts.ft import replay
from replay_fixture import Loader, opened

ROOT = Path(__file__).resolve().parents[2]


class DrawContracts(unittest.TestCase):
    def test_durable_tail_replayed_before_fresh_draw(self):
        with tempfile.TemporaryDirectory() as directory:
            with opened(directory) as loader:
                batches = [next(loader)]
                snapshot = loader.state_dict()
                batches.extend([next(loader), next(loader)])
            base = Loader()
            with opened(directory, base) as recovered:
                recovered.load_state_dict(snapshot)
                self.assertEqual([next(recovered) for _ in range(3)], batches)
                self.assertEqual(base.calls, 0)
                fresh = next(recovered)
                self.assertEqual([item['source_row_id'] for item in fresh], list(range(12, 16)))
                self.assertEqual([item['_r_draw']['occurrence'] for item in fresh], list(range(12, 16)))
                self.assertEqual(base.calls, 1)
                with self.assertRaises(replay.ReplayError): recovered.load_state_dict(snapshot)
            with opened(directory) as again:
                self.assertEqual(next(again), batches[0])  # delivered != committed

    def test_occurrence_not_prompt_and_snapshot_is_detached(self):
        with tempfile.TemporaryDirectory() as directory, opened(directory) as loader:
            batch = next(loader)
            self.assertEqual(len({i['_r_draw']['group_id'] for i in batch}), 4)
            state = loader.state_dict(); state['contract']['k'] = -1
            self.assertEqual(loader.state_dict()['contract']['k'], 8)
            batch[0]['messages'][0]['content'] = 'caller mutation'
        # Return values cannot alter immutable journal bytes.

    def test_real_process_crash_windows(self):
        points = ('after_genesis', 'after_next', 'blob_temp_written', 'blob_temp_synced',
                  'blob_published', 'draw_temp_written', 'draw_published', 'draw_durable',
                  'before_yield', 'after_delivery')
        for point in points:
            with self.subTest(point=point), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'r'
                process = subprocess.run([sys.executable, str(ROOT / 'tests/ft/replay_fixture.py'), str(root), point],
                                         cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                                         capture_output=True, text=True, timeout=10)
                self.assertEqual(process.returncode, -9, process.stderr)
                self.assertTrue((Path(directory) / 'cut.json').exists())
                published = (root / 'draws/000000000001.json').exists()
                base = Loader()
                with opened(root, base) as recovered:
                    result = next(recovered)
                    self.assertEqual([i['source_row_id'] for i in result], list(range(4)))
                    self.assertEqual([i['_r_draw']['occurrence'] for i in result], list(range(4)))
                    self.assertEqual(base.calls, 0 if published else 1)
                    self.assertEqual([i['source_row_id'] for i in next(recovered)], list(range(4, 8)))
                    if point == 'after_delivery':
                        self.assertEqual(json.loads((Path(directory) / 'first-item.json').read_text()), result[0])

    def test_exception_after_next_poisons_live_cursor(self):
        with tempfile.TemporaryDirectory() as directory:
            with opened(directory) as loader:
                def fail(phase, path=None):
                    if phase == 'after_next': raise OSError('disk fault')
                with patch.object(replay, '_cut', fail), self.assertRaises(OSError): next(loader)
                with self.assertRaisesRegex(replay.ReplayError, 'poisoned'): loader.state_dict()
                with self.assertRaisesRegex(replay.ReplayError, 'poisoned'): next(loader)
            with opened(directory) as recovered:
                self.assertEqual(next(recovered)[0]['source_row_id'], 0)

    def test_snapshot_waits_for_transaction_and_second_thread_rejected(self):
        with tempfile.TemporaryDirectory() as directory, opened(directory) as loader:
            entered, release, snapshot_done = threading.Event(), threading.Event(), threading.Event()
            snapshots, errors = [], []
            def gate(phase, path=None):
                if phase == 'after_next': entered.set(); self.assertTrue(release.wait(5))
            def take_next():
                try: next(loader)
                except BaseException as exc: errors.append(exc)
            def take_snapshot():
                try: snapshots.append(loader.state_dict())
                except BaseException as exc: errors.append(exc)
                finally: snapshot_done.set()
            with patch.object(replay, '_cut', gate):
                worker = threading.Thread(target=take_next); worker.start()
                self.assertTrue(entered.wait(5))
                observer = threading.Thread(target=take_snapshot); observer.start()
                self.assertFalse(snapshot_done.wait(0.05))
                release.set(); worker.join(5); observer.join(5)
            self.assertFalse(errors)
            self.assertFalse(worker.is_alive() or observer.is_alive())
            self.assertEqual(snapshots[0]['wal_sequence'], 1)
            self.assertEqual(snapshots[0]['occurrence_count'], 4)
            with self.assertRaisesRegex(replay.ReplayError, 'one iterator'): next(loader)

    def test_bad_snapshot_and_second_epoch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with opened(directory) as loader:
                next(loader); snapshot = loader.state_dict()
            for key, value in [('wal_sequence', 2), ('record_sha256', 'f'*64), ('occurrence_count', 99)]:
                with opened(directory) as recovered:
                    bad = copy.deepcopy(snapshot); bad[key] = value
                    with self.assertRaises(replay.ReplayError): recovered.load_state_dict(bad)
            with opened(directory) as recovered:
                recovered.sampler.set_epoch(0)
                with self.assertRaises(replay.ReplayError): recovered.sampler.set_epoch(1)
            with self.assertRaises(replay.ReplayError):
                replay.DrawLoader(Loader(), directory, 'other-run', 'a'*64, 'b'*64, k=8)

    def test_corrupt_final_record_gap_blob_and_symlink_rejected(self):
        for damage in ('tail', 'gap', 'blob', 'size', 'symlink'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with opened(root) as loader:
                    next(loader); next(loader)
                path = root / 'draws/000000000002.json'
                record = json.loads(path.read_text())
                if damage == 'tail': path.write_bytes(path.read_bytes()[:10])
                elif damage == 'gap': (root / 'draws/000000000001.json').unlink()
                elif damage == 'blob': (root / 'blobs' / record['batch']['sha256']).write_bytes(b'bad')
                elif damage == 'size':
                    record['batch']['size'] += 1; path.write_bytes(replay._encode(record))
                else:
                    raw = path.read_bytes(); path.unlink()
                    external = root / 'target'; external.write_bytes(raw); path.symlink_to(external)
                with self.assertRaises(replay.ReplayError): opened(root)

    def test_owner_exclusive_and_inherited_owner_rejected(self):
        with tempfile.TemporaryDirectory() as directory, opened(directory) as loader:
            with self.assertRaises(BlockingIOError): opened(directory)
            if not hasattr(os, 'fork'): self.skipTest('fork unavailable')
            pid = os.fork()
            if pid == 0:
                try:
                    try: next(loader)
                    except replay.ReplayError: os._exit(0)
                    os._exit(1)
                except BaseException: os._exit(2)
            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            self.assertEqual(next(loader)[0]['source_row_id'], 0)


class BlobContracts(unittest.TestCase):
    def test_idempotent_publish_and_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            blobs = replay.BlobStore(directory)
            ref = blobs.put(b'original')
            self.assertEqual(blobs.put(b'original'), ref)
            self.assertEqual(blobs.get(ref), b'original')
            (Path(directory) / ref['sha256']).write_bytes(b'conflicting')
            with self.assertRaises(replay.ReplayError): blobs.put(b'original')
            with self.assertRaises(replay.ReplayError): blobs.get(ref)
            with self.assertRaises(replay.ReplayError): blobs.get({'sha256': '../escape', 'size': 1})
            path = Path(directory) / 'record'
            replay._publish(path, b'one', 'test')
            replay._publish(path, b'one', 'test')
            with self.assertRaises(replay.ReplayError): replay._publish(path, b'two', 'test')
            self.assertEqual(path.read_bytes(), b'one')


if __name__ == '__main__': unittest.main()
