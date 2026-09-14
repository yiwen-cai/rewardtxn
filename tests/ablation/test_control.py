"""CPU protocol/failure injection; no GPU, Ray, or training needed."""
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from ablation_control import Client, ControlFailure, Supervisor


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.s = Supervisor('run', self.root / 'control', heartbeat_timeout=.15, exit_timeout=.2)
        data = self.root / 'export.json'
        data.write_text('{"count": 32}')
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps({'nonce': 'run', 'fatal': False,
            'files': [{'path': str(data), 'sha256': hashlib.sha256(data.read_bytes()).hexdigest()}]}))
        self.terminal = dict(tasks=0, requests=0, writers=0)
        self.register('launcher')

    def tearDown(self):
        self.s.journal.close()
        self.tmp.cleanup()

    def call(self, component, op, **fields):
        return self.s.handle(dict(nonce='run', component=component, op=op, **fields), component)

    def register(self, name, role=None):
        return self.call(name, 'register', role=role or name, identity={'host': 'cpu', 'pid': 123, 'start': '1'})

    def prepare(self, name):
        return self.call(name, 'prepare_close', seq=0, fatal=False, manifest=str(self.manifest), terminal=self.terminal)

    def offline(self, name, role=None):
        self.register(name, role)
        permit = self.prepare(name)
        self.s.disconnected(name, name)
        return self.call('launcher', 'confirm_exit', target=name, permit=permit,
                         evidence=dict(exited=True, exitcode=0, oom=False, joined=True, thread_alive=False, waited=True, **self.terminal))

    def assert_failed(self):
        self.assertIsNotNone(self.s.failed)
        self.assertIsNone(self.s.final)
        self.assertFalse((self.root / 'control' / 'shutdown.json').exists())
        self.assertFalse((self.root / 'control' / 'run_success.json').exists())
        with self.assertRaises(ControlFailure):
            self.call('launcher', 'finalize_run', required=[])

    def test_offline_worker_driver_then_long_evaluation(self):
        self.offline('worker')
        self.offline('driver')
        self.assertIsNotNone(self.s.evaluation_deadline)
        self.register('evaluator')
        # 3x heartbeat window represents evaluation > 10s after clean shutdown.
        for _ in range(10):
            time.sleep(.05)
            self.call('launcher', 'heartbeat')
            self.call('evaluator', 'heartbeat')
            self.s.tick()
        self.assertEqual(self.s.components['driver']['state'], 'OFFLINE_CONFIRMED')
        permit = self.prepare('evaluator')
        self.call('launcher', 'confirm_exit', target='evaluator', permit=permit,
                  evidence=dict(exited=True, exitcode=0, oom=False, waited=True, **self.terminal))
        decision = self.call('launcher', 'finalize_run', required=['worker', 'driver', 'evaluator'])
        self.call('launcher', 'shutdown', decision=decision, seq=0)
        self.assertTrue((self.root / 'control' / 'shutdown.json').exists())

    def test_early_disconnect_even_exit_zero(self):
        self.register('driver')
        self.s.disconnected('driver', 'driver')
        self.assert_failed()

    def test_prepared_ack_then_crash(self):
        self.register('driver')
        permit = self.prepare('driver')
        with self.assertRaises(ControlFailure):
            self.call('launcher', 'confirm_exit', target='driver', permit=permit,
                      evidence=dict(exited=True, exitcode=-9, oom=True, **self.terminal))
        self.assert_failed()

    def test_permitted_but_not_exited(self):
        self.register('worker')
        self.prepare('worker')
        self.s.components['worker']['exit_deadline'] = 0
        with self.assertRaises(ControlFailure):
            self.s.tick()
        self.assert_failed()

    def test_late_fatal_after_offline_and_decision(self):
        self.offline('driver')
        self.offline('evaluator')
        self.call('launcher', 'finalize_run', required=['driver', 'evaluator'])
        with self.assertRaises(ControlFailure):
            self.call('driver', 'fatal', seq=9999, code='LATE_CALLBACK')
        self.assert_failed()

    def test_fatal_after_supervisor_permit_revokes_certificate(self):
        self.offline('evaluator')
        decision = self.call('launcher', 'finalize_run', required=['evaluator'])
        self.call('launcher', 'shutdown', decision=decision, seq=0)
        with self.assertRaises(ControlFailure):
            self.call('evaluator', 'fatal', code='LATE_FATAL', seq=999)
        self.assert_failed()

    def test_sequence_gap(self):
        self.register('driver')
        with self.assertRaises(ControlFailure):
            self.call('driver', 'event', seq=2, kind='update', fields={})
        self.assert_failed()

    def test_event_after_ready(self):
        self.register('driver')
        self.prepare('driver')
        with self.assertRaises(ControlFailure):
            self.call('driver', 'event', seq=1, kind='update', fields={})
        self.assert_failed()

    def test_live_writer_rejects_export(self):
        self.register('driver')
        with self.assertRaises(ControlFailure):
            self.call('driver', 'prepare_close', seq=0, fatal=False, manifest=str(self.manifest),
                      terminal=dict(tasks=0, requests=0, writers=1))
        self.assert_failed()

    def test_bad_export_hash(self):
        self.register('driver')
        (self.root / 'export.json').write_text('corrupt')
        with self.assertRaises(ControlFailure):
            self.prepare('driver')
        self.assert_failed()

    def test_restart_cannot_reset_latch(self):
        self.register('rm-host', 'rm_host')
        with self.assertRaises(ControlFailure):
            self.register('rm-host', 'rm_host')
        self.assert_failed()

    def test_evaluator_lost(self):
        self.offline('driver')
        self.register('evaluator')
        self.s.components['evaluator']['heartbeat'] = 0
        with self.assertRaises(ControlFailure):
            self.s.tick()
        self.assert_failed()

    def test_training_offline_keeps_rm_host_monitored(self):
        self.register('rm-host', 'rm_host')
        self.offline('worker')
        self.s.components['rm-host']['heartbeat'] = 0
        with self.assertRaises(ControlFailure):
            self.s.tick()
        self.assert_failed()

    def test_evaluation_registration_deadline(self):
        self.offline('driver')
        self.s.evaluation_deadline = 0
        with self.assertRaises(ControlFailure):
            self.s.tick()
        self.assert_failed()

    def test_control_record_failure_is_irreversible(self):
        self.register('driver')
        with patch.object(self.s.journal, 'write', side_effect=OSError('disk full')):
            with self.assertRaises(ControlFailure):
                self.prepare('driver')
        self.assert_failed()

    def test_manifest_changed_after_export_ack(self):
        self.offline('driver')
        self.offline('evaluator')
        (self.root / 'export.json').write_text('changed after ACK')
        with self.assertRaises(ControlFailure):
            self.call('launcher', 'finalize_run', required=['driver', 'evaluator'])
        self.assert_failed()

    def test_self_witness_not_exit_evidence(self):
        permit = self.prepare('launcher')
        with self.assertRaises(ControlFailure):
            self.call('launcher', 'confirm_exit', target='launcher', permit=permit,
                      evidence=dict(exited=True, exitcode=0, oom=False, waited=True, **self.terminal))
        self.assert_failed()

    def ray_exit(self, **override):
        self.register('driver')
        self.register('rm-host', 'rm_host')
        permit = self.prepare('rm-host')
        evidence = dict(exited=True, oom=False, **self.terminal,
                        exit_kind='ray_intended_actor_exit', worker_id='worker-id', actor_id='actor-id',
                        is_alive=False, exit_type='INTENDED_USER_EXIT',
                        exit_detail='exit_actor() is called.', actor_state='DEAD',
                        num_restarts=0, registered_process_gone=True)
        evidence.update(override)
        return self.call('driver', 'confirm_exit', target='rm-host', permit=permit, evidence=evidence)

    def test_ray_actor_expected_exit_without_fake_waitstatus(self):
        self.assertEqual(self.ray_exit()['state'], 'OFFLINE_CONFIRMED')

    def test_ray_actor_kill_not_intended_exit(self):
        with self.assertRaises(ControlFailure):
            self.ray_exit(exit_detail='ray.kill is called.')
        self.assert_failed()

    def test_ray_actor_pid_still_exists(self):
        with self.assertRaises(ControlFailure):
            self.ray_exit(registered_process_gone=False)
        self.assert_failed()

    def test_export_timeout_with_healthy_heartbeat(self):
        self.register('worker')
        with patch('ablation_control.time.monotonic', return_value=100):
            self.call('worker', 'event', seq=1, kind='PHASE_BEGIN', fields=dict(phase='export', timeout=120))
        with patch('ablation_control.time.monotonic', return_value=221):
            self.call('worker', 'heartbeat')
            self.call('launcher', 'heartbeat')
            with self.assertRaises(ControlFailure):
                self.s.tick()
        self.assertEqual(self.s.failed['code'], 'PHASE_TIMEOUT')
        self.assert_failed()

    def test_cancel_timeout_with_healthy_heartbeat(self):
        self.assert_phase_timeout('cancel')

    def assert_phase_timeout(self, phase):
        self.register('worker')
        with patch('ablation_control.time.monotonic', return_value=100):
            self.call('worker', 'event', seq=1, kind='PHASE_BEGIN', fields=dict(phase=phase, timeout=30))
        with patch('ablation_control.time.monotonic', return_value=131):
            self.call('worker', 'heartbeat')
            self.call('launcher', 'heartbeat')
            with self.assertRaises(ControlFailure):
                self.s.tick()
        self.assertEqual(self.s.failed['code'], 'PHASE_TIMEOUT')
        self.assertEqual(self.s.failed['phase'], phase)
        self.assert_failed()

    def test_cancel_timeout_cannot_be_extended(self):
        self.assert_phase_extension_rejected('cancel')

    def assert_phase_extension_rejected(self, phase):
        self.register('worker')
        with self.assertRaises(ControlFailure):
            self.call('worker', 'event', seq=1, kind='PHASE_BEGIN', fields=dict(phase=phase, timeout=31))
        self.assert_failed()

    def test_late_export_end_cannot_erase_deadline(self):
        self.register('worker')
        with patch('ablation_control.time.monotonic', return_value=100):
            self.call('worker', 'event', seq=1, kind='PHASE_BEGIN', fields=dict(phase='export', timeout=120))
        with patch('ablation_control.time.monotonic', return_value=221):
            with self.assertRaises(ControlFailure):
                self.call('worker', 'event', seq=2, kind='PHASE_END', fields=dict(phase='export'))
        self.assert_failed()

    def test_export_timeout_cannot_be_extended_by_client(self):
        self.register('worker')
        with self.assertRaises(ControlFailure):
            self.call('worker', 'event', seq=1, kind='PHASE_BEGIN', fields=dict(phase='export', timeout=121))
        self.assert_failed()

    def test_finished_export_removes_only_phase_deadline(self):
        self.register('worker')
        with patch('ablation_control.time.monotonic', return_value=100):
            self.call('worker', 'event', seq=1, kind='PHASE_BEGIN', fields=dict(phase='export', timeout=120))
        with patch('ablation_control.time.monotonic', return_value=110):
            self.call('worker', 'event', seq=2, kind='PHASE_END', fields=dict(phase='export'))
        with patch('ablation_control.time.monotonic', return_value=230):
            self.call('worker', 'heartbeat')
            self.call('launcher', 'heartbeat')
            self.s.tick()
        self.assertIsNone(self.s.failed)

    def test_export_in_progress_cannot_close(self):
        self.register('worker')
        self.call('worker', 'event', seq=1, kind='PHASE_BEGIN', fields=dict(phase='export', timeout=120))
        with self.assertRaises(ControlFailure):
            self.call('worker', 'prepare_close', seq=1, fatal=False, manifest=str(self.manifest), terminal=self.terminal)
        self.assert_failed()


class ProcessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sock = str(self.root / 'control.sock')
        script = Path(__file__).resolve().parents[2] / 'scripts' / 'ablation_control.py'
        self.proc = subprocess.Popen([sys.executable, str(script), 'serve', '--socket', self.sock,
                                     '--nonce', 'run', '--output', str(self.root / 'records'),
                                     '--heartbeat-timeout', '.4'])
        deadline = time.monotonic() + 3
        while not Path(self.sock).exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.clients = []

    def client(self, name):
        c = Client(self.sock, 'run', name, heartbeat_interval=.05, ack_timeout=.2)
        self.clients.append(c)
        return c

    def tearDown(self):
        for c in self.clients:
            c.close_transport()
        if self.proc.poll() is None:
            self.proc.terminate()
        self.proc.wait(timeout=3)
        self.tmp.cleanup()

    def test_swallowed_memoryerror_still_fatal(self):
        launcher, rm = self.client('launcher'), self.client('rm-host')
        try:
            try:
                raise MemoryError('injected write failure')
            except MemoryError:
                rm.fatal('SEAL_APPEND_CAPACITY')
                raise
        except MemoryError:
            pass  # Original reward function can swallow this.
        with self.assertRaises(ControlFailure):
            rm.check()
        with self.assertRaises(ControlFailure):
            launcher.check()
        self.assertEqual(self.proc.wait(timeout=3), 1)
        self.assertFalse((self.root / 'records' / 'run_success.json').exists())

    def test_supervisor_death_during_evaluation(self):
        launcher, evaluator = self.client('launcher'), self.client('evaluator')
        self.proc.kill()
        self.proc.wait(timeout=3)
        with self.assertRaises(ControlFailure):
            launcher.check()
        with self.assertRaises(ControlFailure):
            evaluator.check()

    def test_early_process_transport_disconnect(self):
        self.client('launcher')
        c = self.client('driver')
        c.close_transport()
        self.assertEqual(self.proc.wait(timeout=3), 1)

    def test_nonblocking_bounded_oversized_event(self):
        self.client('launcher')
        c = self.client('driver')
        with self.assertRaises(ControlFailure):
            c.event('oversized', data='x' * 70000)
        self.assertIsNotNone(c.latch)
        # Local transport failure must also stop liveness, not heartbeat forever.
        self.assertEqual(self.proc.wait(timeout=3), 1)

    def test_ack_timeout_fail_closed(self):
        c = self.client('launcher')
        os.kill(self.proc.pid, signal.SIGSTOP)
        try:
            started = time.monotonic()
            with self.assertRaises(ControlFailure):
                c.check()
            self.assertLess(time.monotonic() - started, 1)
            self.assertIsNotNone(c.latch)
        finally:
            os.kill(self.proc.pid, signal.SIGCONT)
        self.assertEqual(self.proc.wait(timeout=3), 1)

    def test_actual_registered_child_crash(self):
        self.client('launcher')
        scriptdir = str(Path(__file__).resolve().parents[2] / 'scripts')
        code = ("import sys,os;sys.path.insert(0,sys.argv[1]);"
                "from ablation_control import Client;"
                "c=Client(sys.argv[2],'run','driver');os._exit(7)")
        child = subprocess.run([sys.executable, '-c', code, scriptdir, self.sock], timeout=3)
        self.assertEqual(child.returncode, 7)
        self.assertEqual(self.proc.wait(timeout=3), 1)

    def test_complete_socket_handshake(self):
        launcher, driver = self.client('launcher'), self.client('driver')
        data = self.root / 'artifact.json'
        data.write_text('{}')
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'nonce': 'run', 'fatal': False, 'files': {
            'artifact.json': hashlib.sha256(data.read_bytes()).hexdigest()}}))
        terminal = dict(tasks=0, requests=0, writers=0)
        evidence = dict(exited=True, exitcode=0, oom=False, waited=True, **terminal)
        permit = driver.prepare_close(manifest, terminal)
        driver.close_transport()
        launcher.confirm_exit('driver', permit, evidence)
        evaluator = self.client('evaluator')
        time.sleep(.85)  # > 2 heartbeat windows; offline driver stays exempt.
        permit = evaluator.prepare_close(manifest, terminal)
        evaluator.close_transport()
        launcher.confirm_exit('evaluator', permit, evidence)
        decision = launcher.finalize_run(['driver', 'evaluator'])
        shutdown = launcher.shutdown_supervisor(decision)
        self.assertEqual(self.proc.wait(timeout=3), 0)
        self.assertEqual(json.loads((self.root / 'records' / 'shutdown.json').read_text()), shutdown)

    def test_local_check_performs_no_request(self):
        c = self.client('launcher')
        # Stop only the mock's RPC entry, leaving the actual heartbeat thread alive.
        with patch.object(c, '_request', side_effect=AssertionError('hot path RPC')):
            for _ in range(100):
                c.check_local()

    def test_local_check_dead_heartbeat_latches(self):
        c = self.client('launcher')
        c.stop.set()
        c.thread.join(timeout=1)
        with self.assertRaises(ControlFailure):
            c.check_local()
        self.assertEqual(c.latch, 'LOCAL_HEARTBEAT_STOPPED')
        self.assertEqual(self.proc.wait(timeout=3), 1)
        with self.assertRaises(ControlFailure):
            c.check_local()

    def test_local_check_stale_ack_latches(self):
        c = self.client('launcher')
        # Keep real heartbeat alive, inject only its most recent ACK timestamp.
        c.last_successful_ack = time.monotonic() - 11
        with self.assertRaises(ControlFailure):
            c.check_local()
        self.assertEqual(c.latch, 'LOCAL_ACK_STALE')
        self.assertEqual(self.proc.wait(timeout=3), 1)

    def test_heartbeat_failure_is_visible_locally_without_rpc(self):
        c = self.client('launcher')
        self.proc.kill()
        self.proc.wait(timeout=3)
        deadline = time.monotonic() + 2
        while c.latch is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertIsNotNone(c.latch)
        with patch.object(c, '_request', side_effect=AssertionError('hot path RPC')):
            with self.assertRaises(ControlFailure):
                c.check_local()


if __name__ == '__main__':
    unittest.main()
