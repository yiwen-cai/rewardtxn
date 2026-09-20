"""R-owned CPU draw journal. No consumed authority, workflow replay or GPU fencing.

Only trusted local StatefulDataLoader states are unpickled. A digest does not
make an untrusted pickle safe. One owner, epoch 0, num_workers=0, one iterator.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import stat
import threading
import uuid


class ReplayError(RuntimeError):
    pass


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode()


def _decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReplayError('duplicate JSON key')
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ReplayError('nonfinite JSON')))
    except (ValueError, UnicodeError) as exc:
        raise ReplayError('invalid journal JSON') from exc


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _cut(phase, path=None):
    """Private deterministic crash seam for CPU tests; no configured runtime action."""


def _directory(path):
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ReplayError('symlink directory forbidden')
    if not path.exists():
        _directory(path.parent)
        path.mkdir(exist_ok=True)
        _sync(path.parent)
    if not path.is_dir():
        raise ReplayError('expected directory')


def _sync(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ReplayError('nonregular artifact')
            return stream.read()
    except OSError as exc:
        raise ReplayError(f'cannot read artifact {path.name}') from exc


def _publish(path, raw, kind):
    temporary = path.parent / ('.tmp-' + uuid.uuid4().hex)
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            _cut(kind + '_temp_written', temporary)
            os.fsync(stream.fileno())
        _cut(kind + '_temp_synced', temporary)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read(path) != raw:
                raise ReplayError('immutable artifact conflict')
        _cut(kind + '_published', path)
        _sync(path.parent)
        _cut(kind + '_durable', path)
    finally:
        temporary.unlink(missing_ok=True)


class BlobStore:
    def __init__(self, root):
        self.root = Path(root).absolute()
        _directory(self.root)

    def put(self, raw):
        if not isinstance(raw, bytes):
            raise ReplayError('blob accepts bytes only')
        reference = {'sha256': _sha(raw), 'size': len(raw)}
        _publish(self.root / reference['sha256'], raw, 'blob')
        return reference

    def get(self, reference):
        if (not isinstance(reference, dict) or set(reference) != {'sha256', 'size'}
                or not isinstance(reference['sha256'], str)
                or not re.fullmatch('[0-9a-f]{64}', reference['sha256'])
                or type(reference['size']) is not int or reference['size'] < 0):
            raise ReplayError('invalid blob reference')
        raw = _read(self.root / reference['sha256'])
        if len(raw) != reference['size'] or _sha(raw) != reference['sha256']:
            raise ReplayError('blob content mismatch')
        return raw


class _Sampler:
    def __init__(self, loader):
        self.loader = loader

    def set_epoch(self, epoch):
        with self.loader._lock:
            self.loader._check()
            if type(epoch) is not int or epoch != 0:
                raise ReplayError('only epoch 0 supported')
            # Base epoch was set before initializing/restoring its iterator.
            # cycle_dataloader's subsequent set_epoch(0) must not reset it.


class DrawLoader:
    def __init__(self, base, root, run_nonce, source_sha256, loader_config_sha256, *, k):
        self._lock = threading.RLock()
        self._pid = os.getpid()
        self._fd = -1
        self._closed, self._poisoned = False, False
        self._thread = None
        self._delivered = False
        self._replay_index = 0
        self._base = base
        self.root = Path(root).absolute()
        if (not isinstance(run_nonce, str) or not run_nonce or type(k) is not int or k < 1
                or any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v)
                       for v in (source_sha256, loader_config_sha256))):
            raise ReplayError('run/source/config/K contract required')
        if (getattr(base, 'num_workers', None) != 0 or type(base.batch_size) is not int
                or base.batch_size < 1 or getattr(base.sampler, 'num_replicas', None) != 1
                or getattr(base.sampler, 'rank', None) != 0):
            raise ReplayError('requires local batch loader, num_workers0 and single rank')
        self.batch_size = base.batch_size
        self._length = len(base)
        self.sampler = _Sampler(self)
        self._contract = {'schema': 1, 'run_nonce': run_nonce, 'source_sha256': source_sha256,
                          'loader_config_sha256': loader_config_sha256, 'k': k,
                          'loader': {'class': type(base).__module__ + '.' + type(base).__qualname__,
                                     'sampler': type(base.sampler).__module__ + '.' + type(base.sampler).__qualname__,
                                     'batch_size': base.batch_size, 'length': self._length,
                                     'drop_last': base.drop_last, 'shuffle': base.sampler.shuffle,
                                     'seed': base.sampler.seed, 'num_workers': 0, 'rank': 0, 'world_size': 1}}
        _directory(self.root)
        try:
            self._fd = os.open(self.root / 'owner.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            if not stat.S_ISREG(os.fstat(self._fd).st_mode):
                raise ReplayError('nonregular owner lock')
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.blobs = BlobStore(self.root / 'blobs')
            _directory(self.root / 'draws')
            base.sampler.set_epoch(0)
            genesis_path = self.root / 'genesis.json'
            if genesis_path.exists() or genesis_path.is_symlink():
                raw = _read(genesis_path)
                self._genesis = _decode(raw)
                if self._genesis.get('contract') != self._contract or _encode(self._genesis) != raw:
                    raise ReplayError('genesis contract mismatch')
            else:
                if any(not p.name.startswith('.tmp-') for p in (self.root / 'draws').iterdir()):
                    raise ReplayError('draws without genesis')
                self._iterator = iter(base)
                initial = self.blobs.put(pickle.dumps(base.state_dict(), protocol=5))
                self._genesis = {'contract': self._contract, 'initial_state': initial}
                _publish(genesis_path, _encode(self._genesis), 'genesis')
            self._records = []
            self._scan()
            self._restore_base()
            self._pending_count = len(self._records)
            os.register_at_fork(after_in_child=self.close)
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self._closed or self._fd < 0 or self._pid != os.getpid():
            raise ReplayError('closed or inherited owner')
        if self._poisoned:
            raise ReplayError('poisoned iterator; close and reopen')

    def _batch(self, raw):
        value = _decode(raw)
        if (not isinstance(value, list) or not value or len(value) > self.batch_size
                or any(not isinstance(item, dict) or '_r_draw' in item
                       or not isinstance(item.get('messages'), list) or not item['messages'] for item in value)):
            raise ReplayError('requires nonempty list of message dictionaries without _r_draw')
        if _encode(value) != raw:
            raise ReplayError('noncanonical batch')
        return value

    def _slots(self, batch, occurrence, sequence):
        slots = []
        for offset, item in enumerate(batch):
            identity = [self._contract['run_nonce'], self._contract['source_sha256'], 0, occurrence + offset]
            slots.append({'sequence': sequence, 'occurrence': occurrence + offset, 'epoch': 0,
                          'group_id': _sha(_encode(identity)), 'k': self._contract['k'],
                          'prompt_sha256': _sha(_encode(item['messages']))})
        return slots

    def _scan(self):
        # Verify the entire durable prefix and all bytes before unpickling state.
        self.blobs.get(self._genesis['initial_state'])
        previous, occurrence = _sha(_encode(self._genesis)), 0
        paths = sorted(p for p in (self.root / 'draws').iterdir() if not p.name.startswith('.tmp-'))
        for sequence, path in enumerate(paths, 1):
            if path.name != f'{sequence:012d}.json':
                raise ReplayError('draw sequence gap or unexpected artifact')
            raw = _read(path); record = _decode(raw)
            if (set(record) != {'sequence', 'previous', 'batch', 'after_state', 'slots'}
                    or _encode(record) != raw or record['sequence'] != sequence or record['previous'] != previous):
                raise ReplayError('draw chain mismatch')
            batch = self._batch(self.blobs.get(record['batch']))
            self.blobs.get(record['after_state'])
            if record['slots'] != self._slots(batch, occurrence, sequence):
                raise ReplayError('draw identity coverage mismatch')
            self._records.append(record)
            previous, occurrence = _sha(raw), occurrence + len(batch)

    def _restore_base(self):
        state_ref = self._records[-1]['after_state'] if self._records else self._genesis['initial_state']
        # Trusted-local-state boundary: never call this on external checkpoint bytes.
        self._base.load_state_dict(pickle.loads(self.blobs.get(state_ref)))
        self._iterator = iter(self._base)

    def __len__(self):
        return self._length

    def __iter__(self):
        with self._lock:
            self._check()
            current = threading.get_ident()
            if self._thread is not None and self._thread != current:
                raise ReplayError('one iterator thread only')
            self._thread = current
            return self

    def _return(self, record):
        batch = self._batch(self.blobs.get(record['batch']))
        for item, slot in zip(batch, record['slots']):
            item['_r_draw'] = copy.deepcopy(slot)
        return batch

    def __next__(self):
        with self._lock:
            self.__iter__()
            if self._replay_index < self._pending_count:
                record = self._records[self._replay_index]
                batch = self._return(record)
                self._replay_index += 1
                self._delivered = True
                return batch
            try:
                batch = next(self._iterator)
            except StopIteration:
                raise
            except BaseException:
                self._poisoned = True
                raise
            try:
                _cut('after_next')
                raw = _encode(batch)
                batch = self._batch(raw)
                after = pickle.dumps(self._base.state_dict(), protocol=5)
                _cut('after_state_serialized')
                sequence = len(self._records) + 1
                occurrence = sum(len(r['slots']) for r in self._records)
                record = {'sequence': sequence,
                          'previous': _sha(_encode(self._records[-1] if self._records else self._genesis)),
                          'batch': self.blobs.put(raw), 'after_state': self.blobs.put(after),
                          'slots': self._slots(batch, occurrence, sequence)}
                _publish(self.root / 'draws' / f'{sequence:012d}.json', _encode(record), 'draw')
                self._records.append(record)
                result = self._return(record)
                _cut('before_yield')
                self._delivered = True
                return result
            except BaseException:
                self._poisoned = True
                raise

    def _descriptor(self, count):
        anchor = self._records[count - 1] if count else self._genesis
        return {'contract': copy.deepcopy(self._contract), 'wal_sequence': count,
                'record_sha256': _sha(_encode(anchor)),
                'after_state': copy.deepcopy(anchor['after_state'] if count else anchor['initial_state']),
                'occurrence_count': sum(len(r['slots']) for r in self._records[:count])}

    def state_dict(self):
        with self._lock:
            self._check()
            return self._descriptor(len(self._records))

    def load_state_dict(self, descriptor):
        with self._lock:
            self._check()
            if self._delivered or self._thread is not None:
                raise ReplayError('restore only before first iteration in a new proxy')
            count = descriptor.get('wal_sequence') if isinstance(descriptor, dict) else None
            if type(count) is not int or not 0 <= count <= len(self._records) or descriptor != self._descriptor(count):
                raise ReplayError('snapshot does not anchor durable prefix')
            # Do not rewind to N: verified WAL suffix through M remains pending.
            self._restore_base()
            self._replay_index = 0

    def close(self):
        if self._pid != os.getpid():
            # A fork child may inherit a lock held by a vanished parent thread.
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            self._closed = True
            return
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
