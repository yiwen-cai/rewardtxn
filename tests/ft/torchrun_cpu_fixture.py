"""Real torchrun CPU env/one-shot transport fixture, not native SPMD recovery."""
import json
import os
from pathlib import Path
import shutil
import sys

EVENT = 'torchrun-cpu-once'
EVIDENCE = {'boundary': 'cpu_torchrun_entry'}
ENV_KEYS = ('FT_RUN_NONCE', 'FT_CONTROL_SOCKET', 'FT_HANDSHAKE_TIMEOUT', 'PATH',
            'PYTHONPATH', 'USER', 'LOGNAME', 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')


def write(path, value):
    with path.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def main(mode):
    root = Path(os.environ['FT_CONTROL_SOCKET']).parent
    if mode == 'launch':
        from areal.infra.utils.launcher import BASE_ENVIRONS
        from areal.infra.utils.proc import build_target_cmd
        argv = ['torchrun', '--rdzv-backend=static', '--master-addr=127.0.0.1',
                '--master-port=29517', '--node-rank=0', '--nnodes=1', '--nproc-per-node=1',
                '--max-restarts=1', '--monitor-interval=0.1', __file__, 'worker']
        command = build_target_cmd(argv, env_vars=BASE_ENVIRONS, use_stdbuf=True)
        # Same inherited-env shell/stdbuf/tee construction as LocalLauncher.
        command += ' 2>&1 | tee -a /output/torchrun.log'
        write(root / 'torchrun-launch.json', {'pid': os.getpid(), 'argv': argv, 'command': command,
              'env': {k: os.environ[k] for k in ENV_KEYS}, 'base_env': BASE_ENVIRONS,
              'torchrun': shutil.which('torchrun'), 'python3': shutil.which('python3')})
        os.execv('/bin/sh', ['/bin/sh', '-c', command])
    elif mode == 'worker':
        from scripts.ft.descendants import Client, snapshot
        from scripts.ft.faults import ControlError
        client = Client('trainer', event_id=EVENT)
        try:
            identity = snapshot(os.getpid())
            info = {'identity': identity, 'incarnation': client.incarnation, 'assignment': client.injection,
                    'env': {k: os.environ[k] for k in ENV_KEYS},
                    'rank': os.environ['RANK'], 'world_size': os.environ['WORLD_SIZE'],
                    'restart_count': os.environ['TORCHELASTIC_RESTART_COUNT'],
                    'base_env': {k: os.environ[k] for k in ('TOKENIZERS_PARALLELISM', 'PYTORCH_KERNEL_CACHE_PATH', 'TRITON_CACHE_DIR', 'VLLM_CACHE_ROOT', 'CUDA_DEVICE_MAX_CONNECTIONS')},
                    'torchrun_parent': {'identity': snapshot(os.getppid()),
                        'argv': Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode().rstrip('\0').split('\0')}}
            write(root / f'torchrun-worker-{client.incarnation}.json', info)
            if client.injection['status'] == 'pending':
                client.ready(EVENT, EVIDENCE)
                client.wait_release(EVENT)
                raise RuntimeError('kill target unexpectedly released')
            try:
                client.ready(EVENT, EVIDENCE)
            except ControlError:
                write(root / 'torchrun-replacement.json', {'incarnation': client.incarnation,
                      'pid': os.getpid(), 'already_fired_rejected_before_send': True})
            else:
                raise RuntimeError('replacement reissued consumed injection')
        finally:
            client.close()
    else:
        raise ValueError(mode)


if __name__ == '__main__':
    main(sys.argv[1])
