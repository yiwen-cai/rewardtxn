"""Diagnostic-only clean group RM adapters; never modifies production modules.

Call install on the RM event-loop thread, then finalize_export on that same
thread after all callbacks have terminated. Resource budgets must be frozen in
ABLATION_RM_LIMITS (JSON). A successful export is prepared, never run success.
"""
from __future__ import annotations
import asyncio
import builtins
import contextvars
import fcntl
import hashlib
import importlib.util
import json
import os
import resource
from pathlib import Path
import socket
import sqlite3
import threading
import time
from types import SimpleNamespace

VARIANTS = ('O', 'R', 'DBM', 'LOGM', 'BOTHM', 'PAYLOAD', 'LITE')
_STATE = None
RM_ATTEMPT_ID = contextvars.ContextVar('ablation_rm_attempt_id', default=None)


def load_source(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('_ablation_' + name, path)
    module = importlib.util.module_from_spec(spec)
    module.__dict__['__builtins__'] = dict(vars(builtins))
    spec.loader.exec_module(module)
    return module


class Connection:
    __slots__ = ("owner", "conn")
    def __init__(self, owner, conn):
        self.owner, self.conn = owner, conn
    def execute(self, sql, *args):
        if self.owner.closed:
            self.owner.fatal('sqlite_after_seal')
            raise RuntimeError('SQLite sealed')
        if not self.owner.sample_group:
            try:
                return self.conn.execute(sql, *args)
            except BaseException as exc:
                self.owner.fatal('sqlite_execute', exception=type(exc).__name__)
                raise
        segment = 'sqlite_begin' if sql.startswith('BEGIN') else ('sqlite_insert' if sql.startswith('INSERT') else 'sqlite_execute')
        return self.owner.guard(segment, self.conn.execute, sql, *args)
    def commit(self):
        if self.owner.sample_group:
            return self.owner.guard('sqlite_commit', self.conn.commit)
        try:
            return self.conn.commit()
        except BaseException as exc:
            self.owner.fatal('sqlite_commit', exception=type(exc).__name__)
            raise
    def rollback(self):
        return self.owner.guard('sqlite_rollback', self.conn.rollback)
    def close(self):
        self.owner.event('connection_close', generation=self.owner.generation)
        return self.owner.guard('sqlite_close', self.conn.close)


class File:
    __slots__ = ("owner", "stream")
    def __init__(self, owner, stream):
        self.owner, self.stream = owner, stream
    def __enter__(self):
        return self if self.owner.sample_io else self.stream
    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.owner.fatal('file_body_exception', exception=exc_type.__name__)
        try:
            return self.stream.__exit__(exc_type, exc_value, traceback)
        except BaseException as exc:
            self.owner.fatal('file_close', exception=type(exc).__name__)
            raise
    def __iter__(self):
        return iter(self.stream)
    def fileno(self):
        return self.stream.fileno()
    def write(self, data):
        if self.owner.closed:
            self.owner.fatal('file_after_seal')
            raise RuntimeError('JSONL sealed')
        if self.owner.sample_io:
            start = time.monotonic_ns()
        try:
            return self.stream.write(data)
        except BaseException as exc:
            self.owner.fatal('file_write', exception=type(exc).__name__)
            raise
        finally:
            if self.owner.sample_io:
                self.owner.record_io('write', start)
    def flush(self):
        if self.owner.sample_io:
            start = time.monotonic_ns()
        try:
            return self.stream.flush()
        except BaseException as exc:
            self.owner.fatal('file_flush', exception=type(exc).__name__)
            raise
        finally:
            if self.owner.sample_io:
                self.owner.record_io('flush', start)
    def __getattr__(self, key):
        return getattr(self.stream, key)


class OracleSampleFile(File):
    __slots__ = ()
    """Only selected O/LITE calls use this close/implicit-flush timing."""
    def __exit__(self, exc_type, exc_value, traceback):
        start = time.monotonic_ns()
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.owner.record_io('close_including_flush', start)


class Adapter:
    def __init__(self, variant, client, limits):
        if variant not in VARIANTS:
            raise ValueError(variant)
        required = ('log_bytes', 'db_bytes', 'seen_ids', 'rss_bytes', 'events')
        if any(not isinstance(limits.get(k), int) or limits[k] <= 0 for k in required):
            raise ValueError('all five positive frozen RM limits are required')
        self.variant, self.client, self.limits = variant, client, limits
        self.control_check = getattr(client, 'check_local', client.check)
        self.pid, self.thread = os.getpid(), threading.get_ident()
        self.identity = dict(pid=self.pid, thread_id=self.thread, host=socket.gethostname(),
                             process_start=Path('/proc/self/stat').read_text().split()[21])
        self.closed, self.failed, self.generation = False, False, 0
        self.buffers, self.events, self.metrics = {}, [], {}
        self.memory_log_bytes = 0
        self.connection = None
        self.event_lock = threading.Lock()
        self.manifest_path = None
        self.sample_io = False
        self.observations = True
        self.sample_group = False
        self.score_calls = 0
        self.cancelling = False
        self.verifiers = []
        self.memlog = variant in ('LOGM', 'BOTHM')
        self.memdb = variant in ('DBM', 'BOTHM')
        self.oracle = load_source('day2_custom_rm')
        self.rm = load_source('phase2_seal_rm')
        for mod in (self.oracle, self.rm):
            if mod.FAULT != 'none':
                raise ValueError('diagnostic adapter requires clean fault=none')
            mod.open = self.open
            # Oracle encodes inside its with-open body: File.__exit__ catches
            # encoding failures before the original _log swallows them.
            mod.json = json if mod is self.oracle else SimpleNamespace(dumps=self.dumps, loads=json.loads)
            for verifier in ('_v1_reward', '_v2_reward'):
                original = getattr(mod, verifier)
                self.verifiers.append((mod, verifier, original, self.wrap_verifier(original, verifier)))
        self.rm.SEAL, self.rm.GROUP_RM = True, True
        self.rm.AUTO_FIX = True
        if self.rm._WINDOWS:
            raise ValueError('fault windows must be empty')
        self.run_dir = Path(self.rm.RUN_DIR).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in ('rewards.jsonl', 'seals.jsonl', 'cas_rejects.jsonl'):
            if (self.run_dir / name).exists():
                raise ValueError('new diagnostic run must have no reward/Seal streams')
        cas_path = Path(self.rm._cas_index_path())
        if cas_path.exists():
            raise ValueError('new diagnostic run must have no CAS database')
        self.rm.sqlite3 = SimpleNamespace(connect=self.connect, Error=sqlite3.Error)
        self.rm.os = SimpleNamespace(environ=dict(os.environ), getpid=os.getpid, makedirs=os.makedirs,
                                     path=SimpleNamespace(join=os.path.join, dirname=os.path.dirname,
                                                          getsize=self.getsize, exists=self.exists))
        append = self.rm._append_json_lines
        def append_guard(path, lines, lock=False):
            code = 'append_' + os.path.basename(path).removesuffix('.jsonl')
            self.sample_io = self.sample_group
            try:
                return self.guard(code, self.append_memory if self.memlog else append, path, lines, lock=lock)
            finally:
                self.sample_io = False
        self.rm._append_json_lines = append_guard
        self.fcntl_proxy = SimpleNamespace(flock=self.flock, LOCK_EX=fcntl.LOCK_EX, LOCK_UN=fcntl.LOCK_UN)
        original_import = builtins.__import__
        def local_import(name, *args, **kw):
            if name == 'fcntl':
                return self.fcntl_proxy
            return original_import(name, *args, **kw)
        self.rm.__builtins__['__import__'] = local_import
        if variant == 'PAYLOAD':
            self.rm._make_log_record = self.payload_record
        self.event('rm_installed', variant=variant, identity=self.identity, limits=limits)

    def wrap_verifier(self, fn, name):
        def verify(*args):
            return self.guard(name, fn, *args)
        return verify

    def flock(self, stream, operation):
        if self.sample_io:
            start = time.monotonic_ns()
        try:
            return fcntl.flock(stream, operation)
        except BaseException as exc:
            self.fatal('file_lock', exception=type(exc).__name__)
            raise
        finally:
            if self.sample_io:
                self.record_io('lock_wait' if operation == fcntl.LOCK_EX else 'unlock', start)

    def record_io(self, code, start):
        metric = self.metrics.setdefault(code, [0, 0, 0])
        metric[0] += 1
        metric[1] += time.monotonic_ns() - start
        metric[2] += 1

    def fatal(self, code, **fields):
        self.failed = True
        self.client.fatal(code, **fields)

    def event(self, kind, **fields):
        if not self.observations:
            return
        try:
            with self.event_lock:
                if len(self.events) >= self.limits['events']:
                    raise MemoryError('RM event capacity')
                if self.closed:
                    raise RuntimeError('observation after seal')
                fields['sequence'] = len(self.events) + 1
                fields['kind'] = kind
                fields['monotonic_ns'] = time.monotonic_ns()
                self.events.append(fields)
        except BaseException:
            self.fatal('observation_failure', kind=kind)
            raise

    observe = event

    def guard(self, code, fn, *args, **kw):
        # Fatal interception is unconditional; detailed timings cover only one
        # complete RM call in each fixed block of 32 (all arms same schedule).
        if not self.sample_group:
            try:
                if self.closed:
                    raise RuntimeError('RM state closed for writes')
                return fn(*args, **kw)
            except BaseException as exc:
                self.fatal(code, exception=type(exc).__name__)
                raise
        start = time.monotonic_ns()
        try:
            if self.closed:
                raise RuntimeError('RM state closed for writes')
            return fn(*args, **kw)
        except BaseException as exc:
            self.fatal(code, exception=type(exc).__name__)
            raise
        finally:
            metric = self.metrics.setdefault(code, [0, 0, 0])
            metric[0] += 1
            metric[1] += time.monotonic_ns() - start
            metric[2] += 1

    def dumps(self, obj, *, ensure_ascii=True, separators=None):
        if self.sample_group:
            return self.guard('json_encode', json.dumps, obj, ensure_ascii=ensure_ascii, separators=separators)
        try:
            return json.dumps(obj, ensure_ascii=ensure_ascii, separators=separators)
        except BaseException as exc:
            self.fatal('json_encode', exception=type(exc).__name__)
            raise

    def begin_cancellation(self):
        self.check()
        self.cancelling = True
        self.event('rm_cancelling')

    def check(self):
        if os.getpid() != self.pid or threading.get_ident() != self.thread:
            self.fatal('rm_owner_changed')
            raise RuntimeError('RM process/thread ownership changed')
        if self.closed:
            self.fatal('write_after_seal')
            raise RuntimeError('RM state sealed')
        if self.failed:
            raise RuntimeError('RM failed')
        self.control_check()

    def capacity(self, report=False):
        if self.failed:
            raise RuntimeError('RM failed')
        pages = 0
        if self.connection is not None and self.memdb:
            c = self.connection.conn
            pages = c.execute('PRAGMA page_count').fetchone()[0] * c.execute('PRAGMA page_size').fetchone()[0]
        # Linux peak RSS retains transient overshoots and costs no procfs I/O.
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        seen = len(self.rm._seen_logical)
        log_bytes = self.memory_log_bytes
        limits = self.limits
        if (log_bytes > limits['log_bytes'] or pages > limits['db_bytes'] or
                seen > limits['seen_ids'] or rss > limits['rss_bytes']):
            counts = dict(log_bytes=log_bytes, db_bytes=pages, seen_ids=seen, rss_bytes=rss)
            for key, value in counts.items():
                if value > limits[key]:
                    self.fatal('capacity_' + key, actual=value, maximum=limits[key])
                    raise MemoryError(key)
        if report:
            if log_bytes != sum(len(b) for b in self.buffers.values()):
                self.fatal('memory_log_accounting')
                raise RuntimeError('memory log byte counter differs from sealed buffers')
            return dict(log_bytes=log_bytes, db_bytes=pages, seen_ids=seen, rss_bytes=rss)

    def open(self, file, mode='r', *, errors=None):
        try:
            if self.closed:
                raise RuntimeError('JSONL sealed')
            stream = builtins.open(file, mode) if errors is None else builtins.open(file, mode, errors=errors)
            return File(self, stream)
        except BaseException as exc:
            self.fatal('file_open', exception=type(exc).__name__)
            raise

    def open_sampled_oracle(self, file, mode='r', *, errors=None):
        wrapper = self.open(file, mode, errors=errors)
        return OracleSampleFile(self, wrapper.stream)

    def connect(self, path, **kw):
        self.check()
        if self.generation and (self.memdb or self.memlog):
            self.fatal('connection_rebuild')
            raise RuntimeError('memory variant connection cannot be rebuilt')
        self.generation += 1
        conn = self.guard('sqlite_connect', sqlite3.connect, ':memory:' if self.memdb else path, **kw)
        self.connection = Connection(self, conn)
        self.event('connection_open', generation=self.generation, memory=self.memdb)
        return self.connection

    def exists(self, path):
        if self.memlog and str(path) in self.buffers:
            return True
        return os.path.exists(path)

    def getsize(self, path):
        if self.memlog and str(path) in self.buffers:
            return len(self.buffers[str(path)])
        return os.path.getsize(path)

    def append_memory(self, path, lines, lock=False):
        if not lines:
            return
        raw = ''.join(lines).encode('utf-8')
        if self.memory_log_bytes + len(raw) > self.limits['log_bytes']:
            raise MemoryError('log_bytes')
        self.buffers.setdefault(str(path), bytearray()).extend(raw)
        self.memory_log_bytes += len(raw)

    def payload_record(self, sample, reward, verifier, injected, autofix=False):
        return dict(group_index=sample.group_index, index=sample.index, rollout_id=sample.rollout_id,
                    reward=reward, verifier=verifier, injected=injected, fault=self.rm.FAULT,
                    seal_enabled=self.rm.SEAL, autofix=autofix, ts=time.time())

    async def lite(self, samples):
        result = [0.0] * len(samples)
        groups = {}
        for i, s in enumerate(samples):
            groups.setdefault(s.group_index if s.group_index is not None else -1, []).append(i)
        async with self.rm._lock:
            for indices in groups.values():
                for i in sorted(indices, key=lambda i: samples[i].index):
                    s = samples[i]
                    r1 = float(self.rm._v1_reward(s.response, s.label or ''))
                    float(self.rm._v2_reward(s.response, s.label or ''))
                    self.oracle._log(s, r1, 'v1', False)
                    result[i] = r1
        return result

    async def score(self, args, samples):
        self.check()
        if not isinstance(samples, list):
            self.fatal('non_group_call')
            raise ValueError('diagnostic requires group RM list')
        if self.cancelling:
            self.event('reward_cancelled', attempt_id=RM_ATTEMPT_ID.get(), entered=False)
            raise asyncio.CancelledError()
        self.score_calls += 1
        sampled = self.observations and self.score_calls % 32 == 1
        if sampled != self.sample_group:
            self.sample_group = sampled
            self.oracle.json = SimpleNamespace(dumps=self.dumps, loads=json.loads) if sampled else json
            self.oracle.open = self.open_sampled_oracle if sampled else self.open
            if self.variant in ('O', 'LITE'):
                self.sample_io = sampled
            for mod, name, original, timed in self.verifiers:
                setattr(mod, name, timed if sampled else original)
        self.event('reward_start', attempt_id=RM_ATTEMPT_ID.get(), ids=[(s.group_index, s.index, s.rollout_id) for s in samples])
        try:
            if self.variant == 'O':
                result = await self.oracle.rm_function(args, samples)
            elif self.variant == 'LITE':
                result = await self.lite(samples)
            else:
                result = await self.rm.rm_function(args, samples)
            self.capacity()
            self.event('reward_return', attempt_id=RM_ATTEMPT_ID.get(), count=len(result))
            self.check()
            return result
        except asyncio.CancelledError:
            if self.cancelling:
                self.event('reward_cancelled', attempt_id=RM_ATTEMPT_ID.get(), entered=True)
            else:
                self.fatal('unexpected_reward_cancellation', attempt_id=RM_ATTEMPT_ID.get())
            raise
        except BaseException as exc:
            self.fatal('reward_call', exception=type(exc).__name__)
            raise

    def finalize_export(self, output_dir):
        self.check()
        counts = self.capacity(report=True)
        self.event('rm_sealed', counters=counts)
        with self.event_lock:
            self.closed = True
        out = Path(output_dir).resolve()
        files = {}
        try:
            out.mkdir(parents=True, exist_ok=False)
            for name in ('rewards.jsonl', 'seals.jsonl', 'cas_rejects.jsonl'):
                src = self.run_dir / name
                data = bytes(self.buffers[str(src)]) if str(src) in self.buffers else (src.read_bytes() if src.exists() else b'')
                if name != 'rewards.jsonl' and self.variant in ('O', 'LITE'):
                    files[name] = {'status': 'not_applicable'}
                    continue
                (out / name).write_bytes(data)
                files[name] = dict(bytes=len(data), records=len(data.splitlines()), sha256=hashlib.sha256(data).hexdigest())
            rows = 0
            if self.connection:
                with sqlite3.connect(out / 'cas_index.sqlite3') as target:
                    self.connection.conn.backup(target)
                    rows = target.execute('SELECT COUNT(*) FROM cas_claims').fetchone()[0]
                raw = (out / 'cas_index.sqlite3').read_bytes()
                files['cas_index.sqlite3'] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            else:
                files['cas_index.sqlite3'] = {'status': 'not_applicable'}
            def event_record(event):
                if event['kind'] == 'reward_start':
                    return {**event, 'ids': [f'{g}:{i}:{r}' for g, i, r in event['ids']]}
                return event
            raw = ''.join(json.dumps(event_record(e)) + '\n' for e in self.events).encode()
            (out / 'events.jsonl').write_bytes(raw)
            files['events.jsonl'] = dict(bytes=len(raw), records=len(self.events), sha256=hashlib.sha256(raw).hexdigest())
            manifest = dict(status='prepared', variant=self.variant, nonce=os.environ.get('ABLATION_RUN_NONCE'),
                            identity=self.identity, generation=self.generation, fatal=self.failed,
                            files=[dict(path=str(out / name), **info) for name, info in files.items() if 'sha256' in info],
                            streams=files, cas_rows=rows, counters=counts, metrics=self.metrics,
                            metric_schema=['sampled_calls', 'sampled_duration_ns', 'timed_calls'], timing_sample_period_rm_calls=32, rss_measurement='linux_peak_rss_bytes',
                            config_sha256=hashlib.sha256(json.dumps(dict(variant=self.variant, limits=self.limits), sort_keys=True).encode()).hexdigest(),
                            sealed=True, source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                             for n in ('ablation_rm.py', 'day2_custom_rm.py', 'phase2_seal_rm.py')})
            for path in out.iterdir():
                with path.open('rb') as f:
                    os.fsync(f.fileno())
            tmp = out / 'export_manifest.json.tmp'
            with tmp.open('w') as f:
                json.dump(manifest, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, out / 'export_manifest.json')
            fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            self.manifest_path = str(out / 'export_manifest.json')
            self.client.event('rm_export_prepared', manifest_sha256=hashlib.sha256(Path(self.manifest_path).read_bytes()).hexdigest(), event_count=len(self.events))
            self.client.check()
            return manifest
        except BaseException as exc:
            self.fatal('export_failure', exception=type(exc).__name__)
            raise


def install(variant=None, client=None, limits=None):
    global _STATE
    if _STATE is not None:
        raise RuntimeError('RM already installed')
    if client is None:
        from ablation_control import Client
        client = Client.from_env('rm-worker')
    _STATE = Adapter(variant or os.environ['ABLATION_VARIANT'], client,
                     limits if limits is not None else json.loads(os.environ['ABLATION_RM_LIMITS']))
    return _STATE


async def rm_function(args, samples):
    state = _STATE or install()
    return await state.score(args, samples)


def finalize_export(output_dir):
    if _STATE is None:
        raise RuntimeError('RM not installed')
    return _STATE.finalize_export(output_dir)


def get_adapter():
    return _STATE
