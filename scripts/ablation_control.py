"""Diagnostic-only fail-closed control plane. Run the server outside Ray/container.

serve --socket PATH --nonce NONCE --output DIR [--heartbeat-timeout 10]
All components use ABLATION_CONTROL_SOCKET and ABLATION_RUN_NONCE. The launcher
must independently wait for the server process and verify shutdown.json before
publishing success or starting another run. Transport ACKs are not run success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import threading
import time
import uuid

MAX_FRAME = 65536
PHASE_BUDGETS = {'cancel': 30.0, 'export': 120.0}
LOCAL_ACK_MAX_AGE = 10.0


class ControlFailure(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity():
    # PID alone does not identify a process across reuse/restart.
    return {"host": socket.gethostname(), "pid": os.getpid(),
            "start": Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]}


def _zero_terminal(value):
    return isinstance(value, dict) and all(value.get(k) == 0 for k in ('tasks', 'requests', 'writers'))


class Supervisor:
    def __init__(self, nonce, output, heartbeat_timeout=10, exit_timeout=10,
                 evaluation_timeout=60, expected=()):
        self.nonce, self.output = nonce, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.heartbeat_timeout, self.exit_timeout = heartbeat_timeout, exit_timeout
        self.evaluation_timeout = evaluation_timeout
        self.expected = set(expected)
        self.lock = threading.RLock()
        self.components = {}
        self.failed = None
        self.final = None
        self.shutdown = None
        self.evaluation_deadline = None
        self.journal = (self.output / 'control.jsonl').open('x')

    def record(self, kind, durable=False, **fields):
        try:
            self.journal.write(json.dumps(dict(time=time.time(), nonce=self.nonce, kind=kind, **fields)) + '\n')
            self.journal.flush()
            if durable:
                os.fsync(self.journal.fileno())
        except BaseException as exc:
            self.failed = self.failed or {'code': 'CONTROL_RECORD_FAILURE', 'detail': repr(exc)}
            raise ControlFailure('control journal unavailable') from exc

    def fail(self, code, **fields):
        if self.failed is None:
            self.failed = dict(code=code, **fields)
            self.final = None
            for c in self.components.values():
                c['state'] = 'FAILED'
            # Revoke even a provisional decision if fatal arrives during shutdown.
            (self.output / 'shutdown.json').unlink(missing_ok=True)
            self.record('FATAL', durable=True, **self.failed)
        raise ControlFailure(str(self.failed))

    def snapshot(self):
        return {'nonce': self.nonce, 'fatal': self.failed,
                'components': {k: dict(v) for k, v in self.components.items()},
                'final': self.final, 'shutdown': self.shutdown}

    def disconnected(self, component, connection):
        with self.lock:
            c = self.components.get(component)
            if c and c['connection'] == connection and c['state'] not in ('EXIT_PERMITTED', 'OFFLINE_CONFIRMED', 'FAILED') and not self.shutdown:
                try:
                    self.fail('EARLY_DISCONNECT', component=component)
                except ControlFailure:
                    pass

    def tick(self):
        with self.lock:
            if self.failed or self.shutdown:
                return
            now = time.monotonic()
            for name, c in self.components.items():
                if c.get('phase') and now > c['phase']['deadline']:
                    self.fail('PHASE_TIMEOUT', component=name, phase=c['phase']['name'])
                if c['state'] in ('ACTIVE', 'QUIESCING', 'READY_TO_CLOSE') and now - c['heartbeat'] > self.heartbeat_timeout:
                    self.fail('HEARTBEAT_TIMEOUT', component=name)
                if c['state'] == 'EXIT_PERMITTED' and now > c['exit_deadline']:
                    self.fail('EXIT_TIMEOUT', component=name)
            if self.evaluation_deadline is not None and now > self.evaluation_deadline:
                self.fail('EVALUATION_NOT_STARTED')

    def _manifest(self, path):
        p = Path(path)
        obj = json.loads(p.read_text())
        if obj.get('nonce', obj.get('run_nonce')) != self.nonce or obj.get('fatal') is not False:
            self.fail('BAD_MANIFEST_IDENTITY')
        files = obj.get('files')
        if not isinstance(files, (list, dict)) or not files:
            self.fail('MISSING_MANIFEST_FILES')
        entries = [{'path': k, 'sha256': v if isinstance(v, str) else v['sha256']} for k, v in files.items()] if isinstance(files, dict) else files
        for item in entries:
            f = Path(item['path'])
            if not f.is_absolute():
                f = p.parent / f
            if digest(f) != item['sha256']:
                self.fail('EXPORT_HASH_MISMATCH', path=str(f))
        return digest(p)

    def handle(self, msg, connection):
        with self.lock:
            if msg.get('nonce') != self.nonce:
                self.fail('NONCE_MISMATCH')
            op, name = msg.get('op'), msg.get('component')
            # FATAL is processed even after final sequence, offline, or a decision.
            if op == 'fatal':
                self.fail(msg.get('code', 'CLIENT_FATAL'), component=name, detail=msg.get('fields'), seq=msg.get('seq'))
            if self.failed:
                raise ControlFailure(str(self.failed))
            if self.shutdown:
                self.fail('MESSAGE_AFTER_SHUTDOWN_PERMIT', component=name)
            if op == 'register':
                if name in self.components:
                    self.fail('COMPONENT_RESTART_OR_DUPLICATE', component=name)
                role = msg['role']
                if role == 'rm_host' and any(c['role'] == 'rm_host' for c in self.components.values()):
                    self.fail('MULTIPLE_RM_HOSTS')
                self.components[name] = dict(state='ACTIVE', role=role, identity=msg['identity'], seq=0,
                                             heartbeat=time.monotonic(), connection=connection)
                self.record('REGISTERED', durable=True, component=name, role=role, identity=msg['identity'])
                if role in ('evaluator', 'auditor'):
                    self.evaluation_deadline = None
                return {'state': 'ACTIVE'}
            c = self.components.get(name)
            if not c or c['connection'] != connection:
                self.fail('UNKNOWN_CONNECTION', component=name)
            if op in ('heartbeat', 'check'):
                c['heartbeat'] = time.monotonic()
                return self.snapshot()
            if op == 'event':
                if c['state'] not in ('ACTIVE', 'QUIESCING'):
                    self.fail('EVENT_AFTER_FINAL_SEQUENCE', component=name)
                if msg['seq'] != c['seq'] + 1:
                    self.fail('SEQUENCE_GAP', component=name)
                c['seq'] = msg['seq']
                if msg['kind'] == 'QUIESCING':
                    c['state'] = 'QUIESCING'
                elif msg['kind'] == 'PHASE_BEGIN':
                    fields = msg.get('fields', {})
                    phase = fields.get('phase')
                    if phase not in PHASE_BUDGETS or fields.get('timeout') != PHASE_BUDGETS[phase] or c.get('phase') or phase in c.get('completed_phases', []):
                        self.fail('INVALID_PHASE_BEGIN', component=name)
                    c['phase'] = {'name': phase, 'deadline': time.monotonic() + PHASE_BUDGETS[phase]}
                elif msg['kind'] == 'PHASE_END':
                    phase = msg.get('fields', {}).get('phase')
                    if not c.get('phase') or c['phase']['name'] != phase:
                        self.fail('INVALID_PHASE_END', component=name)
                    if time.monotonic() > c['phase']['deadline']:
                        self.fail('PHASE_TIMEOUT', component=name, phase=phase)
                    c.setdefault('completed_phases', []).append(phase)
                    c['phase'] = None
                self.record('EVENT', component=name, seq=c['seq'], event=msg['kind'], fields=msg.get('fields'))
                return {'ack': c['seq']}
            if op == 'prepare_close':
                if c.get('phase'):
                    self.fail('UNFINISHED_PHASE_AT_CLOSE', component=name)
                if c['state'] not in ('ACTIVE', 'QUIESCING') or msg['seq'] != c['seq'] or msg.get('fatal') is not False:
                    self.fail('INVALID_FINAL_BARRIER', component=name)
                if not _zero_terminal(msg['terminal']):
                    self.fail('LIVE_TASKS_AT_CLOSE', component=name)
                for dep in msg['terminal'].get('dependencies', []):
                    if self.components.get(dep, {}).get('state') != 'OFFLINE_CONFIRMED':
                        self.fail('DEPENDENCY_NOT_OFFLINE', component=dep)
                c['state'] = 'READY_TO_CLOSE'
                self.record('READY_TO_CLOSE', durable=True, component=name, seq=c['seq'])
                h = self._manifest(msg['manifest'])
                permit = dict(nonce=self.nonce, component=name, identity=c['identity'], seq=c['seq'],
                              manifest_sha256=h, terminal=msg['terminal'], token=uuid.uuid4().hex, fatal=False)
                self.record('CLOSE_PERMIT', durable=True, permit=permit)
                c.update(state='EXIT_PERMITTED', permit=permit, manifest=str(msg['manifest']), exit_deadline=time.monotonic() + self.exit_timeout)
                return permit
            if op == 'confirm_exit':
                if c['role'] not in ('launcher', 'rm_host', 'driver') or c['state'] not in ('ACTIVE', 'QUIESCING'):
                    self.fail('EXIT_REPORTER_NOT_ACTIVE')
                target = self.components.get(msg['target'])
                evidence = msg['evidence']
                allowed = {'launcher': {'driver', 'rm_host', 'worker', 'evaluator', 'auditor'},
                           'driver': {'rm_host'}, 'rm_host': {'worker'}}
                if msg['target'] == name or not target or target['role'] not in allowed[c['role']]:
                    self.fail('INVALID_EXIT_WITNESS')
                if not target or target['state'] != 'EXIT_PERMITTED' or msg['permit'] != target['permit']:
                    self.fail('EXIT_WITHOUT_MATCHING_PERMIT')
                valid = (evidence.get('exited') is True and evidence.get('oom') is False
                         and _zero_terminal(evidence))
                if target['role'] == 'worker':
                    valid = valid and evidence.get('joined') is True and evidence.get('thread_alive') is False
                elif target['role'] == 'rm_host':
                    # Raylet owns the process: a driver cannot obtain waitpid status.
                    # Require Ray's intended user exit plus disappearance of the exact
                    # registered PID/start identity, never synthesize exitcode=0.
                    valid = valid and all((
                        evidence.get('exit_kind') == 'ray_intended_actor_exit',
                        bool(evidence.get('worker_id')), bool(evidence.get('actor_id')),
                        evidence.get('is_alive') is False,
                        evidence.get('exit_type') == 'INTENDED_USER_EXIT',
                        'exit_actor() is called.' in evidence.get('exit_detail', ''),
                        evidence.get('actor_state') == 'DEAD',
                        evidence.get('num_restarts') == 0,
                        evidence.get('registered_process_gone') is True,
                    ))
                else:
                    valid = valid and evidence.get('waited') is True and evidence.get('exitcode') == 0
                if not valid or time.monotonic() > target['exit_deadline']:
                    self.fail('ABNORMAL_EXIT', component=msg['target'], evidence=evidence)
                target['state'] = 'OFFLINE_CONFIRMED'
                self.record('OFFLINE_CONFIRMED', durable=True, component=msg['target'], reporter=name, evidence=evidence)
                train = [v for v in self.components.values() if v['role'] in ('driver', 'worker', 'rm_host')]
                if train and all(v['state'] == 'OFFLINE_CONFIRMED' for v in train) and not any(v['role'] in ('evaluator', 'auditor') for v in self.components.values()):
                    self.evaluation_deadline = time.monotonic() + self.evaluation_timeout
                return {'state': 'OFFLINE_CONFIRMED'}
            if op == 'finalize_run':
                if c['role'] != 'launcher':
                    self.fail('FINAL_REPORTER_NOT_LAUNCHER')
                required = set(msg['required']) | self.expected | (set(self.components) - {name})
                if not required or any(self.components.get(n, {}).get('state') != 'OFFLINE_CONFIRMED' for n in required):
                    self.fail('INCOMPLETE_COMPONENTS')
                if not any(self.components[n]['role'] in ('evaluator', 'auditor') for n in required):
                    self.fail('MISSING_EVALUATION')
                barriers = {}
                for n in sorted(required):
                    entry = self.components[n]
                    if self._manifest(entry['manifest']) != entry['permit']['manifest_sha256']:
                        self.fail('MANIFEST_CHANGED_AFTER_CLOSE', component=n)
                    barriers[n] = entry['permit']
                self.final = dict(nonce=self.nonce, token=uuid.uuid4().hex, components=sorted(required),
                                  barriers=barriers, fatal=False)
                self.record('FINAL_DECISION', durable=True, decision=self.final)
                return self.final
            if op == 'shutdown':
                if c['role'] != 'launcher' or not self.final or msg['decision'] != self.final or msg['seq'] != c['seq']:
                    self.fail('INVALID_SHUTDOWN_BARRIER')
                self.shutdown = dict(nonce=self.nonce, decision=self.final, token=uuid.uuid4().hex, launcher_seq=c['seq'])
                self.record('SUPERVISOR_CLOSE_PERMIT', durable=True, permit=self.shutdown)
                tmp = self.output / 'shutdown.tmp'
                with tmp.open('x') as f:
                    json.dump(self.shutdown, f)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.output / 'shutdown.json')
                fd = os.open(self.output, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return self.shutdown
            self.fail('UNKNOWN_OPERATION', operation=op)


class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        connection, component = uuid.uuid4().hex, None
        try:
            while True:
                line = self.rfile.readline(MAX_FRAME + 1)
                if not line:
                    return
                if len(line) > MAX_FRAME or not line.endswith(b'\n'):
                    self.server.control.fail('FRAME_OVERFLOW')
                msg = json.loads(line)
                component = msg.get('component')
                try:
                    result = self.server.control.handle(msg, connection)
                    response = {'ok': True, 'result': result}
                except Exception as exc:
                    if self.server.control.failed is None:
                        try:
                            self.server.control.fail('CONTROL_PROTOCOL_FAILURE', detail=repr(exc))
                        except ControlFailure:
                            pass
                    response = {'ok': False, 'error': str(exc)}
                self.wfile.write(json.dumps(response).encode() + b'\n')
                self.wfile.flush()
                if msg.get('op') == 'shutdown' and response['ok']:
                    self.server.shutdown_ack.set()
                    return
        except Exception as exc:
            with self.server.control.lock:
                try:
                    self.server.control.fail('CONTROL_TRANSPORT_FAILURE', detail=repr(exc))
                except ControlFailure:
                    pass
        finally:
            self.server.control.disconnected(component, connection)


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    request_queue_size = 16

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(32)
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            with self.control.lock:
                try:
                    self.control.fail('CONTROL_CONNECTION_CAPACITY')
                except ControlFailure:
                    pass
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


def serve(path, nonce, output, heartbeat_timeout=10, exit_timeout=10, evaluation_timeout=60, expected=()):
    ctl = Supervisor(nonce, output, heartbeat_timeout, exit_timeout, evaluation_timeout, expected)
    with _Server(path, _Handler) as server:
        server.control, server.shutdown_ack = ctl, threading.Event()
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        thread.start()
        try:
            while not server.shutdown_ack.wait(.05):
                ctl.tick()
                if ctl.failed:
                    return 1
            return 1 if ctl.failed else 0
        except ControlFailure:
            return 1
        finally:
            server.shutdown()
            ctl.journal.close()


class Client:
    _instances = {}
    _instances_lock = threading.Lock()

    @classmethod
    def from_env(cls, component):
        key = (os.getpid(), os.environ['ABLATION_CONTROL_SOCKET'], os.environ['ABLATION_RUN_NONCE'], component)
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(key[1], key[2], component)
            return cls._instances[key]

    def __init__(self, path, nonce, component, role=None, heartbeat_interval=1, ack_timeout=2):
        self.path, self.nonce, self.component = path, nonce, component
        self.role = role or {'rm-host': 'rm_host', 'rm-worker': 'worker'}.get(component, component)
        self.seq, self.latch, self.permit = 0, None, None
        self.lock, self.stop = threading.Lock(), threading.Event()
        self._event_lock = threading.Lock()
        self.ack_timeout = ack_timeout
        self.last_successful_ack = 0.0
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(ack_timeout)
        self.sock.connect(path)
        self.reader = self.sock.makefile('rb')
        self._request('register', role=self.role, identity=identity())
        self.thread = threading.Thread(target=self._heartbeat, args=(heartbeat_interval,), daemon=True, name='ablation-control')
        self.thread.start()

    def _request(self, op, **fields):
        if not self.lock.acquire(timeout=self.ack_timeout):
            self.latch = self.latch or 'CONTROL_LOCK_TIMEOUT'
            self.sock.close()
            raise ControlFailure(self.latch)
        try:
            payload = json.dumps(dict(op=op, nonce=self.nonce, component=self.component, **fields)).encode() + b'\n'
            if len(payload) > MAX_FRAME:
                raise ControlFailure('CONTROL_FRAME_OVERFLOW')
            self.sock.sendall(payload)
            response = self.reader.readline(MAX_FRAME + 1)
            if not response or len(response) > MAX_FRAME:
                raise ControlFailure('CONTROL_ACK_MISSING_OR_OVERFLOW')
            response = json.loads(response)
            if not response['ok']:
                raise ControlFailure(response['error'])
            self.last_successful_ack = time.monotonic()
            return response['result']
        except Exception as exc:
            self.latch = self.latch or str(exc)
            self.stop.set()
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            raise ControlFailure(self.latch) from exc
        finally:
            self.lock.release()

    def _heartbeat(self, interval):
        while not self.stop.wait(interval):
            try:
                self._request('heartbeat')
            except ControlFailure:
                self.stop.set()
                return

    def check(self):
        if self.latch:
            raise ControlFailure(self.latch)
        return self._request('check')

    def check_local(self):
        """Hot-path health gate with no RPC on success; never clears a failure.

        The independent heartbeat owns remote failure discovery. A stale ACK or
        dead heartbeat is itself fatal, even if the last received state was OK.
        Failure reporting is bounded by the same transport deadline as fatal().
        """
        if self.latch:
            raise ControlFailure(self.latch)
        code = None
        if self.stop.is_set() or not self.thread.is_alive():
            code = 'LOCAL_HEARTBEAT_STOPPED'
        elif time.monotonic() - self.last_successful_ack > LOCAL_ACK_MAX_AGE:
            code = 'LOCAL_ACK_STALE'
        if code:
            self.fatal(code)
            raise ControlFailure(self.latch)

    def event(self, kind, **fields):
        if self.latch:
            raise ControlFailure(self.latch)
        # Business event producers are serialized independently of heartbeat.
        # Caller must quiesce all producers before prepare_close.
        with self._event_lock:
            self.seq += 1
            return self._request('event', seq=self.seq, kind=kind, fields=fields)

    def fatal(self, code, **fields):
        self.latch = self.latch or code
        try:
            self._request('fatal', code=code, fields=fields, seq=self.seq + 1)
        except ControlFailure:
            # An already closed permitted connection must not hide a late fatal.
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as emergency:
                    emergency.settimeout(self.ack_timeout)
                    emergency.connect(self.path)
                    payload = dict(op='fatal', nonce=self.nonce, component=self.component,
                                   code=code, fields={}, seq=self.seq + 1)
                    emergency.sendall(json.dumps(payload).encode() + b'\n')
                    emergency.recv(MAX_FRAME)
            except OSError:
                pass  # Local latch remains; launcher independently monitors server.

    def prepare_close(self, manifest, terminal):
        with self._event_lock:
            self.check()
            self.permit = self._request('prepare_close', seq=self.seq, fatal=False,
                                        manifest=str(manifest), terminal=terminal)
            return self.permit

    def confirm_exit(self, component, permit, evidence):
        self.check()
        return self._request('confirm_exit', target=component, permit=permit, evidence=evidence)

    def finalize_run(self, required_components):
        self.check()
        return self._request('finalize_run', required=list(required_components))

    def shutdown_supervisor(self, decision):
        self.check()
        self.stop.set()
        self.thread.join(self.ack_timeout)
        return self._request('shutdown', decision=decision, seq=self.seq)

    def close_transport(self):
        self.stop.set()
        self.thread.join(self.ack_timeout)
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.reader.close()
        self.sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['serve'])
    parser.add_argument('--socket', required=True)
    parser.add_argument('--nonce', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--heartbeat-timeout', type=float, default=10)
    parser.add_argument('--exit-timeout', type=float, default=10)
    parser.add_argument('--evaluation-timeout', type=float, default=60)
    parser.add_argument('--expected', default='')
    args = parser.parse_args()
    return serve(args.socket, args.nonce, args.output, args.heartbeat_timeout, args.exit_timeout,
                 args.evaluation_timeout, filter(None, args.expected.split(',')))


if __name__ == '__main__':
    raise SystemExit(main())
