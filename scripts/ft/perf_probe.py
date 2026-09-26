"""Opt-in R timing probe (RTX_PERF_PROBE=<path>). Off by default; no effect on
state, receipts or verdicts. Aggregates wall and per-thread CPU time per span."""
import atexit
import contextlib
import json
import os
import threading
import time

_PATH = os.environ.get('RTX_PERF_PROBE')
_LOCK = threading.Lock()
_STATS = {}


def enabled():
    return bool(_PATH)


def add(name, wall, cpu=0.0):
    with _LOCK:
        entry = _STATS.setdefault(name, [0, 0.0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += wall
        entry[2] += cpu
        entry[3] = max(entry[3], wall)


@contextlib.contextmanager
def span(name):
    if not _PATH:
        yield
        return
    wall, cpu = time.perf_counter(), time.thread_time()
    try:
        yield
    finally:
        add(name, time.perf_counter() - wall, time.thread_time() - cpu)


def dump():
    if not _PATH:
        return
    with _LOCK:
        data = {name: {'count': c, 'wall_s': w, 'thread_cpu_s': u, 'max_wall_s': m}
                for name, (c, w, u, m) in sorted(_STATS.items())}
    path = f'{_PATH}.{os.getpid()}.json'
    with open(path + '.tmp', 'w') as stream:
        json.dump(data, stream, indent=1, sort_keys=True)
    os.replace(path + '.tmp', path)


if _PATH:
    atexit.register(dump)
