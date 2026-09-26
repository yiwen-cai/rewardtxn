"""A+R single-rank Megatron integration; explicit engineering scope.

Uses native asynchronous DCP then waits for finalization before publishing the
retained consumption token. Pending saves require a bound common-job cleanup
receipt before same-namespace takeover; arbitrary orphan recovery is unsupported.
"""
import concurrent.futures
import dataclasses
import hashlib
import os
from pathlib import Path
import threading

import numpy as np
import torch

from . import perf_probe
from . import state
from . import writer_recovery
from .batch_identity import strip_identity
from .replay import DrawLoader, _encode, _decode, _read, _publish
from .rlvr_replay import digest
from .training_replay import RetainedLoader, TrainingRLVR, TrainingBridge

COMPONENTS = ('model', 'optimizer_master', 'optimizer_moments', 'optimizer_step', 'scheduler',
              'rng_python', 'rng_numpy', 'rng_torch_cpu', 'rng_device', 'rng_tracker', 'policy')


def normalized(value):
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
    if isinstance(value, LocalNonpersistentObject):
        return {'type': 'LocalNonpersistentObject', 'persistence': 'not_saved'}
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    if isinstance(value, np.ndarray):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [normalized(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if type(value).__module__.startswith('megatron.core.dist_checkpointing') and hasattr(value, 'data'):
        return {'type': type(value).__name__, 'key': value.key, 'data': normalized(value.data)}
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f'unsupported native state payload {type(value)}')


def native_snapshot(engine):
    result = normalized(engine.checkpointer.generate_state_dict(with_optimizer=True, with_rng=True))
    if set(result) != {'model', 'optimizer', 'lr_scheduler', 'rng_state'}:
        raise RuntimeError('unsupported native full-state schema')
    optimizer = result['optimizer']
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    if not {'param', 'exp_avg', 'exp_avg_sq', 'step'} <= keys(optimizer):
        raise RuntimeError('missing native optimizer master/moments/step')
    rng = result['rng_state']['data']
    if len(rng) != 1 or not {'random_rng_state', 'np_rng_state', 'torch_rng_state',
                             'cuda_rng_state', 'rng_tracker_states'} <= set(rng[0]):
        raise RuntimeError('missing native RNG components')
    return result


def _normalized_leaf_r(value):
    """``normalized`` for one leaf, hashing the tensor buffer in place instead
    of via an extra ``tobytes()`` copy. Output is identical."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy()).hexdigest()}
    if isinstance(value, np.ndarray) and value.flags['C_CONTIGUOUS']:
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(value.reshape(-1).view(np.uint8)).hexdigest()}
    return normalized(value)


def _normalized_async(value, pool):
    """Mirror of ``normalized`` whose tensor hashes run in ``pool``."""
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject
    if isinstance(value, (LocalNonpersistentObject, torch.Tensor, np.ndarray)):
        return pool.submit(_normalized_leaf_r, value)
    if isinstance(value, dict):
        return {str(k): _normalized_async(v, pool) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return [_normalized_async(v, pool) for v in value]
    if type(value).__module__.startswith('megatron.core.dist_checkpointing') and hasattr(value, 'data'):
        return {'type': type(value).__name__, 'key': value.key, 'data': _normalized_async(value.data, pool)}
    return normalized(value)


def _resolve(value):
    if isinstance(value, concurrent.futures.Future):
        return value.result()
    if isinstance(value, dict):
        return {k: _resolve(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v) for v in value]
    return value


SNAPSHOT_THREADS = 8


def native_snapshot_r(engine):
    """R-only copy of ``native_snapshot`` with per-tensor hashing in parallel.
    Output is byte-identical; the shared observer keeps the serial original."""
    return snapshot_of(engine.checkpointer.generate_state_dict(with_optimizer=True, with_rng=True))


def snapshot_of(state_dict):
    """Signature of an already captured native state dict (see native_snapshot_r)."""
    with perf_probe.span('r.native_snapshot'):
        with concurrent.futures.ThreadPoolExecutor(SNAPSHOT_THREADS) as pool:
            result = _resolve(_normalized_async(state_dict, pool))
    return _check_snapshot(result)


def _writer_exited(pid):
    """Read-only /proc check (never waitpid: finalize owns reaping)."""
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return True
    return raw[raw.rfind(')') + 2:].split()[0] in ('Z', 'X')


HASH_WAIT_SECONDS = 900


class _Hasher(threading.Thread):
    """Speculative off-path work for one pending generation: once the forked
    DCP writer has exited (or the barrier says finalize succeeded), hash the
    bulk shards and the captured state. No collectives, no AsyncCallsQueue
    access; any failure only means the barrier recomputes synchronously."""

    def __init__(self, checkpoint, writer_pid, state_dict):
        super().__init__(name='r-commit-hasher', daemon=True)
        self.checkpoint, self.writer_pid, self.state_dict = checkpoint, writer_pid, state_dict
        self.finalized = threading.Event()
        self.prehashed = self.signature = self.error = None

    def run(self):
        try:
            self.signature = snapshot_of(self.state_dict)
        except BaseException as exc:  # recomputed at the barrier
            self.error = exc
        try:
            while not self.finalized.is_set():
                if self.writer_pid is not None and _writer_exited(self.writer_pid):
                    break
                self.finalized.wait(0.05)
            self.prehashed = state.prehash(self.checkpoint, only=prunable)
        except BaseException as exc:
            self.error = self.error or exc
            self.prehashed = None


def _check_snapshot(result):
    if set(result) != {'model', 'optimizer', 'lr_scheduler', 'rng_state'}:
        raise RuntimeError('unsupported native full-state schema')
    optimizer = result['optimizer']
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(v) for v in value))
        return set()
    if not {'param', 'exp_avg', 'exp_avg_sq', 'step'} <= keys(optimizer):
        raise RuntimeError('missing native optimizer master/moments/step')
    rng = result['rng_state']['data']
    if len(rng) != 1 or not {'random_rng_state', 'np_rng_state', 'torch_rng_state',
                             'cuda_rng_state', 'rng_tracker_states'} <= set(rng[0]):
        raise RuntimeError('missing native RNG components')
    return result


def prunable(name):
    """Only bulk DCP shards are dropped; small metadata/policy files stay."""
    return name.startswith('native/') and name.endswith('.distcp')


PRUNE_KEEP = 2


class TrainingStop(Exception):
    """Engineering probe reached its explicit committed-step boundary."""


class Runtime:
    def __init__(self, config, root, *, stop_after=None):
        self.config, self.root = config, Path(root)
        self.stop_after = stop_after
        self.actor = self.owner = self.draw = self.bridge = self.workflow = None
        self.generation = self.update = None
        self.optimizer_stats = None
        self.scheduler_done = False
        self.finalized_ids = []
        self.scheduled_ids = []
        self.closed = False
        self.pending = None
        self.pin_first_commit = False
        self.config_hash = digest(dataclasses.asdict(config))
        self.root.mkdir(parents=True, exist_ok=True)

    def event(self, name, **fields):
        import time
        record = {'event': name, 'pid': os.getpid(), 'monotonic_ns': time.monotonic_ns(), **fields}
        with (self.root / 'events.jsonl').open('ab') as stream:
            stream.write(_encode(record) + b'\n'); stream.flush(); os.fsync(stream.fileno())

    def make_loader(self, base):
        ledger = self.root / 'state'
        control = state._read(ledger / 'control.json') if (ledger / 'control.json').exists() else None
        gate = self.root / 'writer.json'
        cleanup = None
        if gate.exists():
            prior = state._read(gate)
            if prior['pid_namespace'] != os.stat('/proc/self/ns/pid').st_ino:
                raise RuntimeError('changed PID namespace; no safe backend takeover')
            if prior['pending']:
                cleanup = writer_recovery.verify_cleanup(prior, control)
        from .reward_return import verifier_fingerprint
        verifier = verifier_fingerprint()
        self.owner = state.acquire_owner(ledger, -1 if control is None else control['epoch'],
            None if control is None else control['processes'], run_nonce=self.config_hash,
            config_sha256=self.config_hash, verifier_version=verifier)
        import time
        mark, started = len(state.CONTENT_HASHED), time.monotonic()
        self.recovery = state.select_recovery(self.owner)
        self.event('recovery_selected', generation=self.recovery['generation'],
                   seconds=time.monotonic() - started,
                   content_hashed=[Path(p).parent.name for p in state.CONTENT_HASHED[mark:]])
        self.pin_first_commit = self.recovery['generation'] is not None
        if self.pin_first_commit:
            state.pin_generation(self.owner, self.recovery['generation'], 'recovery_loaded')
        if cleanup is not None:
            self.event('pending_writer_abandoned', **cleanup,
                       recovered_generation=self.recovery['generation'])
            self.writer_gate(False)
        source = hashlib.sha256(Path(self.config.train_dataset.path).read_bytes()).hexdigest()
        self.draw = DrawLoader(base, self.root / 'draw', self.config_hash, source,
            digest(dataclasses.asdict(self.config.train_dataset)), k=self.config.gconfig.n_samples)
        self.loader = RetainedLoader(self.draw, self.owner, lambda item: self.bridge.authorize(item))
        return self.loader

    def attach(self, actor):
        from areal.reward.gsm8k import gsm8k_reward_fn
        if (not torch.distributed.is_initialized() or torch.distributed.get_world_size() != 1
                or actor.config.backend != 'megatron:d1p1t1' or not actor.config.megatron.async_save
                or not actor.config.megatron.use_checkpoint_opt_param_scheduler
                or actor.config.ppo_n_minibatches != 1):
            raise RuntimeError('requires single native actor, async DCP, scheduler restore and one PPO minibatch')
        self.actor = actor
        assets = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(Path(self.config.tokenizer_path).iterdir())
                  if p.is_file() and (p.suffix == '.json' or p.name in ('merges.txt', 'vocab.txt', 'tokenizer.model'))}
        self.workflow = TrainingRLVR(owner=self.owner, root=self.root / 'artifacts', attempts={},
            tokenizer_sha256=digest(assets), reward_fn=gsm8k_reward_fn,
            gconfig=self.config.gconfig, tokenizer=self.config.tokenizer_path)
        self.workflow.current_version = actor.get_version
        self.workflow.max_lag = self.config.rollout.max_head_offpolicyness
        self.bridge = TrainingBridge(self.workflow, self.root / 'identity', policy_version=0,
            version=actor.get_version, max_lag=self.workflow.max_lag)
        queue = actor.checkpointer._async_queue
        schedule, finalize = queue.schedule_async_request, queue.maybe_finalize_async_calls
        def scheduled(request):
            result = schedule(request)
            self.scheduled_ids.append(result)
            self.event('async_scheduled', call_id=result, generation=self._io_generation())
            return result
        def finalized(*args, **kwargs):
            result = finalize(*args, **kwargs)
            self.finalized_ids.extend(result)
            for call_id in result:
                self.event('async_finalized', call_id=call_id, generation=self._io_generation())
            return result
        queue.schedule_async_request, queue.maybe_finalize_async_calls = scheduled, finalized
        if getattr(queue, 'persistent', False):
            raise RuntimeError('deferred commit requires a non-persistent (per-save fork) DCP writer')

    def _io_generation(self):
        return self.pending['generation'] if self.pending is not None else self.generation

    def begin(self, trajectories):
        self.settle()
        if self.generation is not None or self.update is not None:
            raise RuntimeError('previous update not committed')
        self.update = self.bridge.begin_update(trajectories, dataclasses.asdict(self.config.actor))
        self.optimizer_stats, self.scheduler_done = None, False

    def prepare(self, data):
        _, clean = self.bridge.prepare_train_batch(data)
        rows = self.update['rows']
        groups = {}
        for row in rows:
            record, _ = self.bridge.resolve_verified(row['identity'])
            draw = row['draw']; group = draw['group_id']
            prompt = {k: v for k, v in record['data'].items() if k != '_r_draw'}
            if group not in groups:
                groups[group] = {'logical_group_id': group, 'k': draw['k'], 'prompt': prompt,
                    'prompt_sha256': state._hash(state._bytes(prompt)), 'samples': []}
            groups[group]['samples'].append({'sample_index': int(row['sample'].rsplit(':', 1)[1]),
                                            'sample': row['sample'], 'receipt': record['receipt']})
        with state._locked(self.owner) as control:
            # Token-derived head: control['head'] can be stale after a crash
            # between token and control writes.
            parent, _ = state._head(self.owner, control)
        intent = {'parent': parent, 'ack_capability': 'none', 'config_sha256': self.config_hash,
            'expected_ranks': ['actor:0'], 'data_snapshot_id': self.update['physical_invocation_id'],
            'components': {c: {'actor:0': ['native/', 'recover/', 'policy.json', 'native-state.json']} for c in COMPONENTS},
            'updates': [{'logical_update_id': self.update['logical_update_id'],
                'physical_update_id': self.update['physical_invocation_id'],
                'train_input_sha256': digest(self.update['input_tensors']), 'groups': list(groups.values())}]}
        self.generation = state.prepare_generation(self.owner, intent,
            self.loader.snapshot([r['sample'] for r in rows]))
        self.event('update_prepared', generation=self.generation, samples=[r['sample'] for r in rows])
        return clean

    def writer_gate(self, pending, generation=None):
        generation = self.generation if generation is None else generation
        value = writer_recovery.pending_gate(self.owner, generation) if pending else {
            'pending': False, 'pid_namespace': os.stat('/proc/self/ns/pid').st_ino,
            'generation': generation}
        state._write(self.root / 'writer.json', value, immutable=False)

    def save(self, handler, original_dump, engine, step_info, *args, **kwargs):
        if self.generation is None or self.optimizer_stats is None or not self.scheduler_done:
            raise RuntimeError('save lacks successful optimizer and scheduler')
        if self.pending is not None:
            raise RuntimeError('previous generation not settled')
        checkpoint = self.owner.root / 'generations' / self.generation / 'checkpoint'
        checkpoint.mkdir()
        # Bind the entire job before any async writer can fork. Recovery needs
        # its completed subreaper receipt, including unregistered fork gaps.
        self.writer_gate(True)
        self.io_checkpoint = checkpoint
        before = len(self.scheduled_ids)
        original_dump(handler, engine, step_info, *args, **kwargs)
        scheduled = self.scheduled_ids[before:]
        if len(scheduled) != 1:
            raise RuntimeError('missing actual async request identity')
        writer_pid = None
        try:
            writer_pid = self.actor.checkpointer._async_queue.async_calls[-1].async_caller.process.pid
        except (AttributeError, IndexError):
            pass  # hasher then waits for the barrier's finalize
        # Everything the commit needs is captured by value on the main thread,
        # at the same point the synchronous version captured it (no optimizer,
        # scheduler or RNG use happens between dump and this capture).
        state_dict = self.actor.checkpointer.generate_state_dict(with_optimizer=True, with_rng=True)
        hasher = _Hasher(checkpoint, writer_pid, state_dict)
        self.pending = {'generation': self.generation, 'checkpoint': checkpoint, 'scheduled': scheduled,
            'snapshot_id': self.update['physical_invocation_id'], 'hasher': hasher,
            'policy': {'version': self.actor.get_version(), 'step_info': dataclasses.asdict(step_info),
                       'optimizer': self.optimizer_stats},
            'global_step': step_info.global_step}
        hasher.start()
        self.generation = self.update = None
        if self.stop_after is not None and step_info.global_step + 1 == self.stop_after:
            self.settle()
            raise TrainingStop()

    def settle(self):
        """Commit barrier for the pending generation, on the main thread: at the
        next begin, before train() returns, or at close. Finalize (a collective)
        stays here; the hasher only did speculative, verifiable hashing."""
        pending = self.pending
        if pending is None:
            return
        self.pending = None
        hasher = pending['hasher']
        try:
            with perf_probe.span('r.settle_finalize'):
                self.actor.checkpointer.wait_async_saves()
            if not set(pending['scheduled']) <= set(self.finalized_ids):
                raise RuntimeError('missing actual async request/finalize identity')
        finally:
            hasher.finalized.set()
        with perf_probe.span('r.settle_join'):
            hasher.join(HASH_WAIT_SECONDS)
        if hasher.is_alive():
            raise RuntimeError('commit hasher did not finish')
        signature = hasher.signature
        if signature is None:
            signature = snapshot_of(hasher.state_dict)
        generation, checkpoint = pending['generation'], pending['checkpoint']
        with perf_probe.span('r.settle_commit'):
            _publish(checkpoint / 'native-state.json', _encode(signature), 'native_state')
            _publish(checkpoint / 'policy.json', _encode(pending['policy']), 'policy')
            snapshot_id = pending['snapshot_id']
            state.record_evidence(self.owner, generation, {'kind': 'optimizer', 'snapshot_id': snapshot_id,
                'physical_updates': [snapshot_id], 'successful': True, 'scheduler_applied': True})
            state.record_evidence(self.owner, generation, {'kind': 'finalize', 'snapshot_id': snapshot_id,
                'rank': 'actor:0', 'writer_closed': True, 'async_call_ids': pending['scheduled']})
            # Writers are finished; checkpoint inventory is fsynced and hashed by
            # state.commit_generation before the token becomes consumption authority.
            self.writer_gate(False, generation)
            token = state.commit_generation(self.owner, generation, prehashed=hasher.prehashed)
            with state._locked(self.owner) as control:
                _, chain = state._head(self.owner, control)
                self.loader.consumed = set(chain[generation][1]['data']['consumed'])
        self.event('committed', generation=generation, token=token, global_step=pending['global_step'])
        if self.pin_first_commit:
            state.pin_generation(self.owner, generation, 'first_commit_after_recovery')
            self.pin_first_commit = False
        with perf_probe.span('r.prune'):
            removed = state.prune_generations(self.owner, prunable, keep=PRUNE_KEEP)
        if removed:
            self.event('pruned', generation=generation, removed_bytes=removed)

    def abandon_pending(self):
        """Without a live engine the pending generation stays uncommitted;
        recovery resolves it. Only the hasher thread is joined."""
        pending, self.pending = self.pending, None
        if pending is not None:
            pending['hasher'].finalized.set()
            pending['hasher'].join(HASH_WAIT_SECONDS)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.abandon_pending()
        if self.bridge is not None:
            self.bridge.close()
        if self.workflow is not None:
            self.workflow.close()
        if self.draw is not None:
            self.draw.close()
        if self.owner is not None:
            self.owner.close()


def install(runtime):
    """Install only in the dedicated A+R trainer process before construction."""
    import areal.trainer.rl_trainer as trainer_module
    from areal.engine.megatron_engine import MegatronPPOActor
    from areal.api import SaveLoadMeta
    from areal.utils.recover import RecoverHandler
    trainer_cls = trainer_module.PPOTrainer
    create = trainer_cls._create_dataloader
    def create_loader(self, dataset, dataset_config, rank, world_size):
        base = create(self, dataset, dataset_config, rank, world_size)
        if dataset_config is self.config.train_dataset:
            return runtime.make_loader(base)
        return base
    trainer_cls._create_dataloader = create_loader

    class RRecoverHandler(RecoverHandler):
        def recover_info_path(self, *args):
            return str(runtime.io_checkpoint / 'recover')

        def _save_checkpoint(self, engine, name='default', **kwargs):
            if engine is not runtime.actor or name != 'default':
                raise RuntimeError('only actor checkpoint supported')
            native = runtime.io_checkpoint / 'native'
            native.mkdir()
            engine.save(SaveLoadMeta(path=str(native), weight_format='dcp', with_optim=True))

        def _load_checkpoint(self, engine, name='default', **kwargs):
            if engine is not runtime.actor or name != 'default':
                raise RuntimeError('only actor checkpoint supported')
            engine.load(SaveLoadMeta(path=str(runtime.io_checkpoint / 'native'), weight_format='dcp', with_optim=True))
            expected = _decode(_read(runtime.io_checkpoint / 'native-state.json'))
            if native_snapshot_r(engine) != expected:
                raise RuntimeError('native loaded model/optimizer/scheduler/RNG differs from retained snapshot')
            runtime.event('native_state_loaded', generation=runtime.recovery['generation'], exact_match=True)

        def load(self, engine, *args, **kwargs):
            runtime.attach(engine)
            if runtime.recovery['generation'] is None:
                return None
            runtime.io_checkpoint = Path(runtime.recovery['checkpoint'])
            result = super().load(engine, *args, **kwargs)
            if result is None:
                raise RuntimeError('retained generation did not actually load RecoverInfo')
            runtime.event('recover_info_loaded', step_info=dataclasses.asdict(result.last_step_info))
            return result

        def dump(self, engine, step_info, *args, **kwargs):
            return runtime.save(self, RecoverHandler.dump, engine, step_info, *args, **kwargs)

    trainer_module.RecoverHandler = RRecoverHandler
    original_trainer_train = trainer_cls.train
    def train(self, *args, **kwargs):
        # Deferred commit reads captured live tensors until the barrier; any
        # post-save offload/colocation swap would race with that.
        if self._should_offload_actor or self._is_v1_awex_colocate(self.config):
            raise RuntimeError('deferred commit requires no actor offload or colocation')
        try:
            result = original_trainer_train(self, *args, **kwargs)
        except BaseException:
            try:
                runtime.settle()
            except BaseException as exc:
                runtime.event('pending_commit_failed', error=type(exc).__name__)
                runtime.abandon_pending()
            raise
        runtime.settle()
        return result
    trainer_cls.train = train
    original_ppo = MegatronPPOActor.ppo_update
    def ppo(self, trajectories, *args, **kwargs):
        runtime.begin(trajectories)
        return original_ppo(self, trajectories, *args, **kwargs)
    MegatronPPOActor.ppo_update = ppo
    original_train = MegatronPPOActor.train_batch
    def train_batch(self, data, *args, **kwargs):
        return original_train(self, runtime.prepare(data), *args, **kwargs)
    MegatronPPOActor.train_batch = train_batch
    original_optimizer = MegatronPPOActor.optimizer_step
    def optimizer(self):
        if runtime.optimizer_stats is not None:
            raise RuntimeError('more than one optimizer call in frozen update')
        result = original_optimizer(self)
        if result.get('update_successful') != 1.0:
            raise RuntimeError('native optimizer update unsuccessful')
        runtime.optimizer_stats = result
        runtime.event('optimizer_applied', generation=runtime.generation, stats=result)
        return result
    MegatronPPOActor.optimizer_step = optimizer
    original_scheduler = MegatronPPOActor.lr_scheduler_step
    def scheduler(self):
        if runtime.optimizer_stats is None or runtime.scheduler_done:
            raise RuntimeError('scheduler out of update order')
        result = original_scheduler(self)
        runtime.scheduler_done = True
        runtime.event('scheduler_applied', generation=runtime.generation)
        return result
    MegatronPPOActor.lr_scheduler_step = scheduler
    original_forward = MegatronPPOActor.forward
    def forward(self, input_, *args, **kwargs):
        return original_forward(self, strip_identity(input_), *args, **kwargs)
    MegatronPPOActor.forward = forward
