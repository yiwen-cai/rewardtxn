"""Same-namespace pending-save recovery under the common job subreaper.

This trusts the local launcher and its receipts. It is not a storage fence or
proof for processes outside that job's descendant tree.
"""
import os
from pathlib import Path
import time

from . import state


def capture_job():
    child = state.process_identity(os.getpid())
    pid = os.getppid()
    while pid > 1:
        ident = state.process_identity(pid)
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        args = [v.decode() for v in argv if v]
        if len(args) >= 3 and args[1:3] == ['-m', 'scripts.ft.job_lifecycle']:
            if os.stat(f'/proc/{pid}/ns/pid').st_ino != os.stat('/proc/self/ns/pid').st_ino:
                raise RuntimeError('job supervisor is in a different PID namespace')
            log = args[args.index('--log') + 1]
            command = args[args.index('--command') + 1]
            return {'supervisor': ident, 'root': child, 'log_path': log,
                    'command': command, 'captured_ns': time.monotonic_ns()}
        child = ident
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        pid = int(fields[1])
    return None


def pending_gate(owner, generation):
    return {'pending': True, 'pid_namespace': os.stat('/proc/self/ns/pid').st_ino,
            'generation': generation, 'epoch': owner.epoch, 'owner_nonce': owner.nonce,
            'owner': state.process_identity(os.getpid()), 'job': capture_job()}


def verify_cleanup(gate, control):
    """Return a bound completed-job receipt; never clear or promote a candidate."""
    if gate['pid_namespace'] != os.stat('/proc/self/ns/pid').st_ino:
        raise RuntimeError('changed PID namespace; no safe writer takeover')
    job = gate.get('job')
    if (not gate['pending'] or job is None or control is None
            or (gate['epoch'], gate['owner_nonce']) != (control['epoch'], control['owner_nonce'])
            or gate['owner'] != control['processes'][0]):
        raise RuntimeError('pending writer lacks bound owner/job evidence')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if any(p['boot_id'] != boot or not state._exited(p)
           for p in (gate['owner'], job['supervisor'], job['root'])):
        raise RuntimeError('old job still alive or changed boot')
    matches = []
    for path in (Path(job['log_path']).parent / 'job-lifecycle').glob('*.json'):
        receipt = state._read(path)
        if (receipt['supervisor_pid'] == job['supervisor']['pid']
                and receipt['root_pid'] == job['root']['pid']
                and receipt['command'] == job['command']
                and receipt['log_path'] == job['log_path']
                and receipt['started_ns'] < job['captured_ns'] < receipt['root_exited_ns']
                <= receipt['finished_ns'] < time.monotonic_ns()):
            matches.append((path, receipt))
    if (len(matches) != 1 or matches[0][1]['cleanup'].get('empty') is not True
            or 'error' in matches[0][1]['cleanup']):
        raise RuntimeError('no unique completed cleanup receipt for pending writer job')
    path, receipt = matches[0]
    return {'gate': gate, 'receipt': str(path), 'receipt_sha256': state._hash(path.read_bytes()),
            'cleanup_finished_ns': receipt['finished_ns']}
