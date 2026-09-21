"""Dedicated A+R engineering entry point; same native PPOTrainer loop."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main(args):
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from scripts.ft.areal_pilot import load_pilot_dataset
    from scripts.ft.training_adapter import Runtime, TrainingStop, install

    config, _ = load_expr_config(args, GRPOConfig)
    if (config.gconfig.n_samples != 8 or config.train_dataset.batch_size != 4
            or config.train_dataset.num_workers != 0 or config.recover.freq_steps != 1
            or config.recover.no_save_optim or config.recover.no_load_optim
            or config.recover.mode not in ('on', 'auto') or config.total_train_epochs != 1):
        raise RuntimeError('unsupported engineering training contract')
    root = Path(config.cluster.fileroot).parent
    phase = json.loads((root / 'phase.json').read_text())
    runtime = Runtime(config, root / 'rewardtxn', stop_after=phase.get('stop_after'))
    install(runtime)
    try:
        dataset = load_pilot_dataset(config.train_dataset.path)
        with PPOTrainer(config, train_dataset=dataset, valid_dataset=None) as trainer:
            try:
                trainer.train(workflow=runtime.bridge)
            except TrainingStop:
                runtime.event('engineering_stop', stop_after=runtime.stop_after)
        runtime.event('training_returned')
    finally:
        runtime.close()


def supervise_trainer(args):
    """Observe a real OS exit, independently of the native launcher's status.

    local.py treats COMPLETED as JobException even when its shell exited zero.
    This parent never performs recovery or turns a nonzero trainer exit into 0.
    """
    config = Path(args[args.index('--config') + 1])
    root = config.parent
    phase = json.loads((root / 'phase.json').read_text())['phase']
    if phase not in ('first', 'resume'):
        raise ValueError('unknown engineering phase')
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--trainer-worker', *args])
    code = child.wait()
    receipt = {'supervisor_pid': os.getpid(), 'trainer_pid': child.pid, 'exit_code': code}
    with (root / (phase + '-trainer-exit.json')).open('x') as stream:
        json.dump(receipt, stream); stream.flush(); os.fsync(stream.fileno())
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--trainer-worker']:
        main(sys.argv[2:])
    else:
        supervise_trainer(sys.argv[1:])
