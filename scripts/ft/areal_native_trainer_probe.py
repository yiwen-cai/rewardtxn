"""One post-update trainer kill; all recovery remains owned by native SPMD."""
import dataclasses
import functools
import inspect
import json
import os
from pathlib import Path
import sys
import time

EVENT = 'trainer-post-update-0'
EVIDENCE = {'successful_ordinal': 2, 'saved_global_step': 0, 'phase': 'post_optimizer_pre_save'}
METADATA = {'step_info.json', 'saver_info.json', 'evaluator_info.json',
            'stats_logger_info.json', 'checkpoint_info.json', 'dataloader_info.pkl'}


def verify_checkpoint(checkpoint, metadata, expected_step, manifest):
    checkpoint, metadata = Path(checkpoint), Path(metadata)
    if not all((metadata / name).is_file() for name in METADATA):
        raise RuntimeError('incomplete RecoverInfo')
    if json.loads((metadata / 'step_info.json').read_text())['global_step'] != expected_step:
        raise RuntimeError('RecoverInfo step mismatch')
    files = manifest(checkpoint)
    if not any(f['path'].endswith('.metadata') for f in files) or not any(
            f['path'].endswith('.distcp') and f['size'] > 0 for f in files):
        raise RuntimeError('incomplete DCP checkpoint')
    return {'checkpoint_path': str(checkpoint), 'checkpoint': files,
            'metadata_path': str(metadata), 'metadata': manifest(metadata)}


class Probe:
    def __init__(self, client, observe):
        self.client, self.observe = client, observe
        self.ordinal, self.saved = 0, None

    def updated(self, stats, update_id):
        if stats['update_successful'] != 1:
            self.observe('unsuccessful_update', stats=stats)
            return
        self.ordinal += 1
        self.observe('successful_update', ordinal=self.ordinal, stats=stats, update_id=update_id)
        if self.client.injection['status'] == 'pending' and self.ordinal == 2:
            if self.saved is None:
                raise RuntimeError('second successful update lacks complete step0 save')
            self.observe('ready_witness', predecessor=self.saved, evidence=EVIDENCE)
            self.client.ready(EVENT, EVIDENCE)
            self.client.wait_release(EVENT)
            raise RuntimeError('kill target unexpectedly survived release')


def install(probe, trainer_class, actor_class, manifest):
    save, step = trainer_class._save_recover_checkpoint, actor_class.optimizer_step
    if list(inspect.signature(save).parameters) != ['self', 'epoch', 'epoch_step', 'global_step']:
        raise RuntimeError('unexpected trainer save signature')
    if list(inspect.signature(step).parameters) != ['self']:
        raise RuntimeError('unexpected optimizer signature')

    @functools.wraps(save)
    def saved(self, epoch, epoch_step, global_step):
        result = save(self, epoch, epoch_step, global_step)
        from areal.utils.saver import Saver
        from areal.utils.recover import RecoverHandler
        cfg = self.recover_handler.config
        if (self.actor.config.megatron.async_save or cfg.no_save_optim or cfg.no_load_optim
                or cfg.mode != 'auto' or self.recover_handler.last_step_info.global_step != global_step):
            raise RuntimeError('synchronous complete recovery save contract failed')
        args = (cfg.experiment_name, cfg.trial_name, cfg.fileroot)
        witness = verify_checkpoint(Saver.get_recover_checkpoint_path(*args),
                                    RecoverHandler.recover_info_path(*args), global_step, manifest)
        probe.observe('complete_save', global_step=global_step, **witness)
        if global_step == 0:
            probe.saved = witness
        return result

    @functools.wraps(step)
    def updated(self):
        result = step(self)
        probe.updated(result, getattr(self, '_pilot_update', None))
        return result

    trainer_class._save_recover_checkpoint, actor_class.optimizer_step = saved, updated


def install_recover_load(probe, handler_class, manifest):
    original = handler_class.load

    @functools.wraps(original)
    def loaded(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result is None:
            probe.observe('recover_info_absent')
        else:
            cfg = self.config
            path = Path(self.recover_info_path(cfg.experiment_name, cfg.trial_name, cfg.fileroot))
            step = dataclasses.asdict(result.last_step_info)
            if not all((path / name).is_file() for name in METADATA):
                raise RuntimeError('loaded RecoverInfo metadata evidence incomplete')
            if json.loads((path / 'step_info.json').read_text()) != step:
                raise RuntimeError('loaded RecoverInfo disagrees with metadata file')
            probe.observe('recover_info_loaded', last_step_info=step,
                          metadata_path=str(path), metadata=manifest(path))
        return result

    handler_class.load = loaded


def main(args):
    from areal import PPOTrainer
    from areal.engine.megatron_engine import MegatronPPOActor
    from areal.utils.recover import RecoverHandler
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from scripts.ft import areal_pilot, areal_pilot_hooks as hooks
    from scripts.ft.descendants import Client, snapshot
    config, _ = load_expr_config(args, GRPOConfig)
    if (os.environ.get('RANK') != '0' or os.environ.get('WORLD_SIZE') != '1'
            or config.total_train_steps != 3 or config.recover.retries != 1
            or config.recover.freq_steps != 1 or config.actor.megatron.async_save
            or config.allocation_mode != 'sglang:d3p1t1+megatron:d1p1t1'):
        raise RuntimeError('probe supports only frozen one-actor three-step configuration')
    os.environ['AREAL_PILOT_EVENTS'] = str(Path(config.cluster.fileroot) / 'pilot_events')
    hooks.install_hooks()
    client = Client('trainer', event_id=EVENT)
    path = Path('/output') / f'trainer-witness-{os.getpid()}-{client.incarnation}.jsonl'
    def observe(event, **fields):
        hooks.writer().flush()
        with path.open('a') as stream:
            stream.write(json.dumps(dict(event=event, pid=os.getpid(), time_ns=time.time_ns(),
                                         incarnation=client.incarnation, identity=snapshot(os.getpid()), **fields), default=str) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    ancestry = []
    pid = os.getpid()
    while pid > 1 and len(ancestry) < 32:
        identity = snapshot(pid)
        ancestry.append({'identity': identity, 'argv': Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')})
        pid = identity['ppid']
    observe('registered', assignment=client.injection, ancestry=ancestry, pilot_incarnation=hooks.writer().incarnation,
            effective_env={k: os.environ[k] for k in ('PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'RANK', 'WORLD_SIZE', 'LD_LIBRARY_PATH', 'HOME') if k in os.environ})
    probe = Probe(client, observe)
    install(probe, PPOTrainer, MegatronPPOActor, hooks.file_manifest)
    install_recover_load(probe, RecoverHandler, hooks.file_manifest)
    try:
        areal_pilot.main(args)
    finally:
        client.close()


if __name__ == '__main__':
    main(sys.argv[1:])
