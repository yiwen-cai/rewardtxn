"""Proposed common local-job cleanup; no retries or training-state access.

Not installed by default. A dedicated subreaper owns one command and its
descendants, so orphaned log writers cannot outlive job completion.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def write_record(path, record):
    with path.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def drain_children():
    """Reap/signal only actual children of this dedicated subreaper."""
    started = time.monotonic()
    exits, signals = [], []
    while True:
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return {'empty': True, 'exits': exits, 'signals': signals}
            if pid == 0:
                break
            exits.append({'pid': pid, 'exit_code': os.waitstatus_to_exitcode(status)})
        elapsed = time.monotonic() - started
        if elapsed > 10:
            raise RuntimeError('owned descendants did not exit within cleanup bound')
        children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
        for value in children:
            pid = int(value)
            try:
                fd = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            try:
                # An unreaped child PID cannot be recycled. Never signal by name.
                stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                if int(stat[1]) != os.getpid():
                    raise RuntimeError('pidfd target no longer belongs to this job')
                sig = signal.SIGTERM if elapsed < 1 else signal.SIGKILL
                signal.pidfd_send_signal(fd, sig)
                signals.append({'pid': pid, 'start_time': int(stat[19]), 'signal': sig.value})
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                os.close(fd)
        time.sleep(0.02)


def supervise(command, log_path):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'cannot establish child subreaper')
    log_path = Path(log_path)
    directory = log_path.parent / 'job-lifecycle'
    directory.mkdir(exist_ok=True)
    record = {'schema': 1, 'supervisor_pid': os.getpid(), 'command': command,
              'started_ns': time.monotonic_ns(), 'log_path': str(log_path)}
    receipt = directory / (uuid.uuid4().hex + '.json')
    interrupted = []
    def stop(signum, frame):
        if not interrupted:
            interrupted.append((signum, time.monotonic()))
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with log_path.open('ab', buffering=0) as log:
        child = subprocess.Popen(command, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        record['root_pid'] = child.pid
        while child.poll() is None:
            if interrupted:
                sig = interrupted[0][0] if time.monotonic() - interrupted[0][1] < 1 else signal.SIGKILL
                child.send_signal(sig)
            time.sleep(0.02)
        code = child.wait()
    record['root_exit_code'] = code
    record['root_exited_ns'] = time.monotonic_ns()
    try:
        record['cleanup'] = drain_children()
    except BaseException as exc:
        record['cleanup'] = {'empty': False, 'error': repr(exc)}
        record['finished_ns'] = time.monotonic_ns()
        write_record(receipt, record)
        # Native local_main retries every completed job, regardless of exit
        # code. Keep this job non-completed if descendants remain uncertain;
        # the experiment's outer timeout must tear down the private namespace.
        while True:
            time.sleep(1)
    record['external_signals'] = [s for s, _ in interrupted]
    record['finished_ns'] = time.monotonic_ns()
    write_record(receipt, record)
    return code if code >= 0 else 128 - code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--command', required=True)
    args = parser.parse_args()
    raise SystemExit(supervise(args.command, args.log))
