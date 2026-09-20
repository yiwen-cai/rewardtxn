"""Isolated CPU-only controller-loss probe; never run kill paths on the host."""
import argparse
import array
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ft import run, state
from scripts.ft.descendants import exited, snapshot, validate_pidfd
from scripts.ft.faults import OwnedProcess, barrier

MODES = ('before-send-crash', 'after-wait-crash', 'before-send-control', 'after-wait-control')


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    with Path(path).open('xb') as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def contained():
    """Require the live outer P1 PID-1 container controller and host launch proof."""
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('pidfd capability required; no kill fallback')
    launch = json.loads(Path('/output/launch.json').read_text())
    frozen = json.loads(Path('/output/frozen.json').read_text())
    parent = snapshot(1)
    if (os.getpid() == 1 or parent['pid_namespace'] == launch['host_pid_namespace']
            or parent['pid_namespace'] != snapshot(os.getpid())['pid_namespace']
            or frozen['run_nonce'] != launch['nonce']
            or b'scripts.ft.namespace_run' not in Path('/proc/1/cmdline').read_bytes()
            or '--network=none' not in launch['argv']
            or any(arg == '--gpus' for arg in launch['argv'])):
        raise RuntimeError('private CPU P1 namespace required; no host kill fallback')
    return {'guardian': snapshot(os.getpid()), 'namespace_controller': parent,
            'host_pid_namespace': launch['host_pid_namespace'], 'nonce': launch['nonce']}


def packet(channel, value, fd=None):
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', [fd]))]
    channel.sendmsg([encoded(value)], ancillary)


def recv(channel):
    data, ancillary, flags, _ = channel.recvmsg(65536, socket.CMSG_SPACE(4), socket.MSG_CMSG_CLOEXEC)
    fds = []
    for level, kind, value in ancillary:
        if (level, kind) != (socket.SOL_SOCKET, socket.SCM_RIGHTS):
            raise RuntimeError('unexpected probe descriptor')
        values = array.array('i'); values.frombytes(value)
        fds.extend(values)
    if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC) or not data:
        for fd in fds:
            os.close(fd)
        raise RuntimeError('probe channel closed/truncated')
    return json.loads(data), fds


def target(directory, bad_ready=None):
    contained()
    spec = importlib.util.spec_from_file_location('cpu_state_fixture', ROOT / 'tests/ft/test_state.py')
    fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)
    with fixture.new_owner(directory / 'state') as owner:
        intent, data = fixture.fixture(owner)
        generation = state.prepare_generation(owner, intent, data)
        path = owner.root / 'generations' / generation / 'intent.json'
        save(directory / 'target-boundary.json', {'identity': snapshot(os.getpid()), 'generation': generation,
            'intent_file': str(path.relative_to(directory)), 'intent_sha256': digest(path.read_bytes()),
            'epoch': owner.epoch, 'boundary': 'real_state_prepare_returned', 'event': 'primary-fault'})
        if bad_ready:
            from scripts.ft import faults
            original_send = faults.send
            def invalid_ready(channel, message):
                if message.get('kind') == 'ready':
                    message = dict(message)
                    message['event_nonce' if bad_ready == 'nonce' else 'pid'] = 'wrong-nonce' if bad_ready == 'nonce' else 1
                return original_send(channel, message)
            with patch.object(faults, 'send', invalid_ready):
                barrier('primary-fault', {'boundary': 'real_state_prepare_returned', 'k': 1})
        else:
            barrier('primary-fault', {'boundary': 'real_state_prepare_returned', 'k': 1})


def controller(directory, mode, channel_fd):
    contained()
    channel = socket.socket(fileno=channel_fd); channel.settimeout(10)
    original_add = run.Journal.add
    original_register = OwnedProcess.register
    original_send = OwnedProcess.send

    def registered(process, pidfd=None):
        owned = original_register(process, pidfd)
        packet(channel, {'kind': 'registered', 'identity': owned.original}, owned.pidfd)
        answer, fds = recv(channel)
        if fds or answer != {'continue': 'registered'}:
            raise RuntimeError('registration witness handshake failed')
        return owned

    def sent(owned, sig):
        result = original_send(owned, sig)
        if result and sig == signal.SIGKILL:
            packet(channel, {'kind': 'signal_returned', 'identity': owned.original, 'signal': int(sig)})
        return result

    def added(journal, kind, **fields):
        if kind != 'observed':
            original_add(journal, kind, **fields)
        point = 'before-send' if kind == 'fired' else 'after-wait' if kind == 'observed' else None
        if point == mode.rsplit('-', 1)[0]:
            packet(channel, {'kind': 'cut', 'point': point, 'fields': fields,
                'journal_sha256': digest(journal.path.read_bytes()), 'journal_size': journal.path.stat().st_size})
            answer, fds = recv(channel)
            if fds or answer != {'continue': 'cut'}:
                raise RuntimeError('cut witness handshake failed')
        if kind == 'observed':
            original_add(journal, kind, **fields)

    frozen = run.preflight(directory / 'inner-config.json', directory / 'inner-schedule.json')
    with patch.object(OwnedProcess, 'register', side_effect=registered), patch.object(OwnedProcess, 'send', sent), patch.object(run.Journal, 'add', added):
        result = run.execute(frozen, directory / 'run')
    save(directory / 'controller-result.json', result)
    channel.close()


def resume(directory, config, destination):
    contained()
    try:
        frozen = run.preflight(config, directory / 'inner-schedule.json')
        result = run.execute(frozen, directory / 'run', resume=True)
        save(destination, {'identity': snapshot(os.getpid()), 'result': result})
    except (OSError, ValueError, RuntimeError) as exc:
        save(destination, {'identity': snapshot(os.getpid()), 'error': type(exc).__name__, 'message': str(exc)})
        return 2
    return 0


def launch_resume(directory, config, destination, expected):
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--resume', str(directory),
                                '--config', str(config), '--result', str(destination)], stdout=subprocess.DEVNULL)
    try:
        if process.wait(timeout=10) != expected:
            raise RuntimeError('resume process exit differs')
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


def one_probe(root, mode):
    containment = contained()
    directory = root / mode; directory.mkdir()
    save(directory / 'containment.json', containment)
    config = {'attempts': [{'target': [sys.executable, str(Path(__file__).resolve()), '--target', str(directory)]}],
              'env': {'PYTHONPATH': str(ROOT)}, 'handshake_timeout': 5, 'run_timeout': 15, 'cleanup_timeout': 2}
    schedule = [{'event_id': 'primary-fault', 'attempt': 0, 'target': 'target', 'waiters': ['target'],
                 'evidence': {'boundary': 'real_state_prepare_returned', 'k': 1}}]
    save(directory / 'inner-config.json', config); save(directory / 'inner-schedule.json', schedule)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    parent.settimeout(10)
    with (directory / 'inner-controller.log').open('xb') as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--controller', str(directory),
            '--mode', mode, '--channel', str(child.fileno())], pass_fds=(child.fileno(),), stdout=log, stderr=subprocess.STDOUT)
        child.close()
        controller_fd = os.pidfd_open(process.pid)
        original = snapshot(process.pid)
        target_fd = None
        observations = []
        try:
            while True:
                message, fds = recv(parent)
                if message['kind'] == 'registered':
                    if target_fd is not None or len(fds) != 1:
                        raise RuntimeError('duplicate/missing target pidfd')
                    target_fd = fds[0]
                    ident = message['identity']
                    validate_pidfd(target_fd, ident['pid'])
                    actual = snapshot(ident['pid'])
                    if actual['ppid'] != process.pid or actual['pid_namespace'] != original['pid_namespace'] or any(actual[k] != v for k, v in ident.items()):
                        raise RuntimeError('target not live controller-owned child')
                    observations.append(dict(message, independently_live=True, actual=actual))
                    packet(parent, {'continue': 'registered'})
                    continue
                if fds:
                    for fd in fds:
                        os.close(fd)
                    raise RuntimeError('unexpected target descriptor')
                observations.append(message)
                if message['kind'] == 'signal_returned':
                    continue
                if message['kind'] != 'cut' or target_fd is None:
                    raise RuntimeError('unexpected cut protocol')
                raw = (directory / 'run/events.jsonl').read_bytes()
                if digest(raw) != message['journal_sha256'] or len(raw) != message['journal_size']:
                    raise RuntimeError('journal cut witness mismatch')
                records = [json.loads(line) for line in raw.splitlines()]
                if sum(r['kind'] == 'fired' for r in records) != 1 or any(r['kind'] == 'observed' for r in records):
                    raise RuntimeError('not the real fired-before-observed window')
                proof = json.loads((directory / 'target-boundary.json').read_text())
                if any(proof['identity'][k] != v for k, v in observations[0]['identity'].items()) or digest((directory / proof['intent_file']).read_bytes()) != proof['intent_sha256']:
                    raise RuntimeError('real prepare boundary not bound')
                dead = exited(target_fd)
                after = mode.startswith('after-wait')
                if dead != after or (after and (message['fields']['exit_code'] != -signal.SIGKILL or not any(x['kind'] == 'signal_returned' for x in observations))):
                    raise RuntimeError('independent pidfd target state disagrees with exact cut')
                save(directory / 'cut-witness.json', {'controller': original, 'target_pidfd_readable': dead,
                    'observations': observations, 'journal_sha256': digest(raw), 'journal_size': len(raw),
                    'records': records, 'target_intent_sha256': proof['intent_sha256'],
                    'scope': 'guardian independently observes target liveness; target wait status belongs to original controller'})
                if mode.endswith('crash'):
                    if snapshot(process.pid) != original:
                        raise RuntimeError('controller identity changed')
                    signal.pidfd_send_signal(controller_fd, signal.SIGKILL)
                    expected = -signal.SIGKILL
                else:
                    packet(parent, {'continue': 'cut'})
                    expected = 0
                code = process.wait(timeout=10)
                if code != expected:
                    raise RuntimeError('controller wait did not confirm outcome')
                # No orphan kill by PID: namespace PID1 owns eventual containment cleanup.
                if not exited(target_fd, 7):
                    raise RuntimeError('target did not exit after controller loss; namespace cleanup required')
                save(directory / 'process-exits.json', {'controller': original, 'controller_wait': code,
                    'target_pidfd_readable_after': True, 'guardian_still_alive': snapshot(os.getpid())})
                break
        finally:
            if process.poll() is None:
                signal.pidfd_send_signal(controller_fd, signal.SIGKILL)
                process.wait(timeout=2)
            os.close(controller_fd)
            if target_fd is not None:
                os.close(target_fd)
            parent.close()
    before = (directory / 'run/events.jsonl').read_bytes()
    save(directory / 'journal-before-resume.json', {'sha256': digest(before), 'records': [json.loads(x) for x in before.splitlines()]})
    launch_resume(directory, directory / 'inner-config.json', directory / 'resume-result.json', 0 if mode.endswith('crash') else 2)
    after = (directory / 'run/events.jsonl').read_bytes()
    save(directory / 'journal-after-resume.json', {'sha256': digest(after), 'records': [json.loads(x) for x in after.splitlines()]})
    if not after.startswith(before):
        raise RuntimeError('resume rewrote journal')
    added = [json.loads(line) for line in after[len(before):].splitlines()]
    if mode.endswith('crash'):
        if len(added) != 1 or added[0].get('classification') != 'technical_invalid' or added[0].get('oracle_status') != 'not_evaluated':
            raise RuntimeError('resume injected/restarted instead of refusing uncertain attempt')
    elif added:
        raise RuntimeError('completed control was modified')
    # Real new-process negative checks on same original immutable journal.
    launch_resume(directory, directory / 'inner-config.json', directory / 'repeat-result.json', 2)
    changed = dict(config, run_timeout=14)
    save(directory / 'changed-config.json', changed)
    launch_resume(directory, directory / 'changed-config.json', directory / 'changed-result.json', 2)
    if (directory / 'run/events.jsonl').read_bytes() != after:
        raise RuntimeError('negative resume modified original journal')
    # Bad tail test only on a copied run; never mutate the original evidence.
    bad = directory / 'bad-tail'; bad.mkdir()
    shutil.copytree(directory / 'run', bad / 'run')
    shutil.copyfile(directory / 'inner-schedule.json', bad / 'inner-schedule.json')
    with (bad / 'run/events.jsonl').open('ab') as stream:
        stream.write(b'{broken'); stream.flush(); os.fsync(stream.fileno())
    corrupt = (bad / 'run/events.jsonl').read_bytes()
    launch_resume(bad, directory / 'inner-config.json', bad / 'rejection.json', 2)
    if (bad / 'run/events.jsonl').read_bytes() != corrupt:
        raise RuntimeError('bad-tail resume mutated evidence')
    save(directory / 'result.json', {'controller_protocol': 'pass', 'mode': mode, 'pending_count': 184,
        'training': 'not_evaluated', 'oracle': 'not_evaluated', 'new_attempts_after_resume': 0})


def bad_ready_probe(root, kind):
    contained()
    directory = root / ('bad-' + kind); directory.mkdir()
    config = {'attempts': [{'target': [sys.executable, str(Path(__file__).resolve()), '--target', str(directory), '--bad-ready', kind]}],
              'env': {'PYTHONPATH': str(ROOT)}, 'handshake_timeout': 3, 'run_timeout': 10, 'cleanup_timeout': 2}
    schedule = [{'event_id': 'primary-fault', 'attempt': 0, 'target': 'target', 'waiters': ['target'],
                 'evidence': {'boundary': 'real_state_prepare_returned', 'k': 1}}]
    save(directory / 'inner-config.json', config); save(directory / 'inner-schedule.json', schedule)
    with (directory / 'inner-controller.log').open('xb') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--plain-controller', str(directory)], stdout=log, stderr=subprocess.STDOUT)
        fd = os.pidfd_open(child.pid)
        try:
            if child.wait(timeout=15) != 0:
                raise RuntimeError('negative controller process failed')
        finally:
            if child.poll() is None:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
                child.wait(timeout=2)
            os.close(fd)
    records = [json.loads(line) for line in (directory / 'run/events.jsonl').read_text().splitlines()]
    result = json.loads((directory / 'result.json').read_text())
    if result.get('classification') != 'technical_invalid' or 'ready identity/nonce mismatch' not in result.get('reason', '') or any(r['kind'] in ('fired', 'observed') for r in records):
        raise RuntimeError('bad nonce/identity was not rejected before injection')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-config', type=Path)
    p.add_argument('--guardian', type=Path)
    p.add_argument('--controller', type=Path)
    p.add_argument('--target', type=Path)
    p.add_argument('--plain-controller', type=Path)
    p.add_argument('--bad-ready', choices=('nonce', 'pid'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--channel', type=int)
    p.add_argument('--config', type=Path)
    p.add_argument('--result', type=Path)
    args = p.parse_args()
    if args.make_config:
        save(args.make_config, {'argv': ['/opt/.venv/bin/python', '/workspace/tests/ft/p2_controller_probe.py', '--guardian', '/output/probes'],
            'env': {'PYTHONPATH': '/workspace'}, 'schedule': [], 'timeouts': {'run': 160, 'handshake': 5, 'lease': 10}})
    elif args.guardian:
        contained(); args.guardian.mkdir()
        save(args.guardian / 'source-sha256.json', {str(path.relative_to(ROOT)): digest(path.read_bytes())
            for path in (Path(__file__).resolve(), ROOT / 'scripts/ft/run.py', ROOT / 'scripts/ft/faults.py',
                         ROOT / 'scripts/ft/state.py', ROOT / 'tests/ft/test_state.py')})
        def deadline(signum, frame):
            raise RuntimeError('probe guardian absolute per-case deadline')
        signal.signal(signal.SIGALRM, deadline)
        for mode in MODES:
            signal.alarm(25)
            try:
                one_probe(args.guardian, mode)
            finally:
                signal.alarm(0)
        for kind in ('nonce', 'pid'):
            signal.alarm(15)
            try:
                bad_ready_probe(args.guardian, kind)
            finally:
                signal.alarm(0)
        save(args.guardian / 'summary.json', {'cases': list(MODES), 'controller_protocol': 'pass',
            'training': 'not_evaluated', 'pending_count': 184})
    elif args.controller:
        controller(args.controller, args.mode, args.channel)
    elif args.target:
        target(args.target, args.bad_ready)
    elif args.plain_controller:
        contained()
        directory = args.plain_controller
        frozen = run.preflight(directory / 'inner-config.json', directory / 'inner-schedule.json')
        save(directory / 'result.json', run.execute(frozen, directory / 'run'))
    elif args.resume:
        raise SystemExit(resume(args.resume, args.config, args.result))
    else:
        p.error('choose one action')


if __name__ == '__main__':
    main()
