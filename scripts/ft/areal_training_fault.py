"""One real post-optimizer trainer fault; native local_main owns all retries."""
import json
import os
from pathlib import Path
import sys

EVENT = 'rewardtxn-post-optimizer-once'
EVIDENCE = {'successful_ordinal': 2, 'saved_global_step': 0,
            'phase': 'post_optimizer_pre_save'}


def main(args):
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from scripts.ft.areal_pilot import load_pilot_dataset
    from scripts.ft.descendants import Client, snapshot
    from scripts.ft.training_adapter import Runtime, install

    config, _ = load_expr_config(args, GRPOConfig)
    if (config.gconfig.n_samples != 8 or config.train_dataset.batch_size != 4
            or config.train_dataset.num_workers != 0 or config.recover.freq_steps != 1
            or config.recover.no_save_optim or config.recover.no_load_optim
            or config.recover.mode not in ('on', 'auto') or config.recover.retries != 1
            or config.total_train_epochs != 1 or config.total_train_steps != 3):
        raise RuntimeError('unsupported single-fault engineering configuration')
    root = Path(config.cluster.fileroot).parent
    client = Client('trainer', event_id=EVENT)

    class FaultRuntime(Runtime):
        ordinal = 0

        def event(self, name, **fields):
            super().event(name, **fields)
            if name != 'optimizer_applied':
                return
            self.ordinal += 1
            if client.injection['status'] != 'pending' or self.ordinal != 2:
                return
            control = json.loads((self.owner.root / 'control.json').read_text())
            gate = json.loads((self.root / 'writer.json').read_text())
            head = control['head']
            if head is None or gate['pending'] or self.owner.epoch != 0:
                raise RuntimeError('fault lacks prior committed generation or closed writer')
            policy = json.loads((self.owner.root / 'generations' / head['generation'] /
                                 'checkpoint/policy.json').read_text())
            if policy['step_info']['global_step'] != 0:
                raise RuntimeError('fault predecessor is not step 0')
            super().event('fault_ready', generation=self.generation, retained=head,
                          incarnation=client.incarnation, evidence=EVIDENCE)
            client.ready(EVENT, EVIDENCE)
            client.wait_release(EVENT)
            raise RuntimeError('SIGKILL target unexpectedly survived')

    runtime = FaultRuntime(config, root / 'rewardtxn')
    install(runtime)
    runtime.event('trainer_registered', incarnation=client.incarnation,
                  identity=snapshot(os.getpid()), assignment=client.injection,
                  argv=sys.argv, parent_pid=os.getppid())
    try:
        dataset = load_pilot_dataset(config.train_dataset.path)
        with PPOTrainer(config, train_dataset=dataset, valid_dataset=None) as trainer:
            trainer.train(workflow=runtime.bridge)
        runtime.event('training_returned')
    finally:
        runtime.close()
        client.close()


if __name__ == '__main__':
    main(sys.argv[1:])
