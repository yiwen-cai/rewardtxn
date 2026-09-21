"""Pause a real DCP writer between tensor writes, then kill its trainer once."""
import functools
import json
import os
from pathlib import Path
import sys
import threading
import time

EVENT = 'rewardtxn-checkpoint-midwrite-once'
EVIDENCE = {'successful_ordinal': 2, 'saved_global_step': 0,
            'phase': 'checkpoint_midwrite'}


def install_writer_probe(runtime, enabled):
    import torch
    from megatron.core.dist_checkpointing.strategies import filesystem_async as fs
    from scripts.ft.descendants import snapshot
    local = threading.local()
    original_bucket = fs.FileSystemWriterAsync.write_preloaded_data
    original_item = fs._write_item

    @functools.wraps(original_bucket)
    def bucket(transform_list, local_proc_idx, write_bucket, *args, **kwargs):
        local.total = len(write_bucket[2][1])
        local.written = 0
        return original_bucket(transform_list, local_proc_idx, write_bucket, *args, **kwargs)

    @functools.wraps(original_item)
    def item(*args, **kwargs):
        result = original_item(*args, **kwargs)
        if not enabled or runtime.ordinal != 2 or not any(isinstance(v, torch.Tensor) for v in args):
            return result
        local.written += 1
        if local.written != 1 or local.total <= 1:
            return result
        stream = next(v for v in args if hasattr(v, 'fileno') and hasattr(v, 'write'))
        if str(stream.name) != runtime.probe_target:
            return result
        # Flush the Python stream solely so the witness can inspect the actual
        # bytes already written. Native remaining writes/fsync/finalize stay pending.
        stream.flush()
        info = os.fstat(stream.fileno())
        witness = {'writer': snapshot(os.getpid()), 'fd': stream.fileno(),
            'path': str(stream.name), 'size': info.st_size, 'device': info.st_dev,
            'inode': info.st_ino, 'remaining_tensors': local.total - local.written,
            'generation': runtime.generation, 'monotonic_ns': time.monotonic_ns()}
        path = runtime.root / 'midwrite-witness.json'
        temporary = path.with_suffix('.tmp')
        with temporary.open('x') as f:
            json.dump(witness, f); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        while True:
            time.sleep(1)

    fs.FileSystemWriterAsync.write_preloaded_data = staticmethod(bucket)
    fs._write_item = item


def main(args):
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from scripts.ft.areal_pilot import load_pilot_dataset
    from scripts.ft.descendants import Client, snapshot, lineage
    from scripts.ft.training_adapter import Runtime, install
    config, _ = load_expr_config(args, GRPOConfig)
    if (config.gconfig.n_samples != 8 or config.train_dataset.batch_size != 4
            or config.train_dataset.num_workers != 0 or config.recover.freq_steps != 1
            or config.recover.no_save_optim or config.recover.no_load_optim
            or config.recover.mode not in ('on', 'auto') or config.recover.retries != 1
            or config.total_train_epochs != 1 or config.total_train_steps != 3):
        raise RuntimeError('unsupported single-fault engineering configuration')
    client = Client('trainer', event_id=EVENT)

    class MidwriteRuntime(Runtime):
        ordinal = 0

        def attach(self, actor):
            super().attach(actor)
            queue = actor.checkpointer._async_queue
            original = queue.schedule_async_request
            def schedule(request):
                if self.ordinal == 2 and client.injection['status'] == 'pending':
                    buckets = request.async_fn_args[1]
                    if queue.persistent or len(buckets[-1][2][1]) <= 1:
                        raise RuntimeError('probe requires temporal writer and incomplete last tensor bucket')
                    self.probe_target = str(buckets[-1][0])
                return original(request)
            queue.schedule_async_request = schedule

        def event(self, name, **fields):
            super().event(name, **fields)
            if name == 'optimizer_applied':
                self.ordinal += 1
            if name != 'async_scheduled' or self.ordinal != 2 or client.injection['status'] != 'pending':
                return
            path = self.root / 'midwrite-witness.json'
            deadline = time.monotonic() + 60
            while not path.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('real writer did not reach incomplete tensor bucket')
                time.sleep(.02)
            witness = json.loads(path.read_text())
            queue = self.actor.checkpointer._async_queue
            active = [c for c in queue.async_calls if c.idx == fields['call_id']]
            assert len(active) == 1 and active[0].async_caller.process.pid == witness['writer']['pid']
            assert snapshot(witness['writer']['pid']) == witness['writer']
            ancestry = lineage(witness['writer']['pid'], snapshot(os.getpid()))
            fd = Path(f"/proc/{witness['writer']['pid']}/fd/{witness['fd']}")
            info = fd.stat()
            assert (info.st_dev, info.st_ino, info.st_size) == (witness['device'], witness['inode'], witness['size'])
            assert witness['size'] > 0 and witness['remaining_tensors'] > 0
            checkpoint = self.io_checkpoint / 'native'
            assert Path(witness['path']).is_relative_to(checkpoint)
            assert witness['generation'] == self.generation and not (checkpoint / '.metadata').exists()
            control = json.loads((self.owner.root / 'control.json').read_text())
            gate = json.loads((self.root / 'writer.json').read_text())
            head = control['head']
            assert gate['pending'] and gate['generation'] == self.generation and self.owner.epoch == 0
            policy = json.loads((self.owner.root / 'generations' / head['generation'] / 'checkpoint/policy.json').read_text())
            assert policy['step_info']['global_step'] == 0
            super().event('fault_ready', generation=self.generation, retained=head,
                incarnation=client.incarnation, evidence=EVIDENCE, writer=witness,
                writer_ancestry=ancestry, gate=gate)
            client.ready(EVENT, EVIDENCE)
            client.wait_release(EVENT)
            raise RuntimeError('SIGKILL target unexpectedly survived')

    runtime = MidwriteRuntime(config, Path(config.cluster.fileroot).parent / 'rewardtxn')
    install(runtime)
    install_writer_probe(runtime, client.injection['status'] == 'pending')
    runtime.event('trainer_registered', incarnation=client.incarnation,
        identity=snapshot(os.getpid()), assignment=client.injection, argv=sys.argv, parent_pid=os.getppid())
    try:
        with PPOTrainer(config, train_dataset=load_pilot_dataset(config.train_dataset.path), valid_dataset=None) as trainer:
            trainer.train(workflow=runtime.bridge)
        runtime.event('training_returned')
    finally:
        runtime.close()
        client.close()


if __name__ == '__main__':
    main(sys.argv[1:])
