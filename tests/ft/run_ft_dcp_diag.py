"""One fresh, nonformal A/F2 run with optional async DCP D2H diagnostics."""
import argparse
from pathlib import Path
import re
import shutil

from run_training_fault import REPO, main, write
from run_ft_minimal import DiskMonitor


def run_diagnostic(name, seed, devices, presync=False, cuda_launch_blocking=False,
                   blocking_copy=False, host_register=False, pointwise_autotune_off=False):
    if not re.fullmatch(r'[a-z0-9-]+', name) or len(devices) != 4 or len(set(devices)) != 4:
        raise ValueError('fresh name and four distinct GPU UUIDs required')
    base = REPO / 'docs/experiments/rewardtxn-ft-20260916/minimal_evidence'
    rid = f'{name}-a'
    if (base / rid).exists():
        raise FileExistsError(base / rid)
    if shutil.disk_usage(base).free < 64 * 1024**3:
        raise RuntimeError('diagnostic run requires at least 64 GiB free')
    monitor = DiskMonitor(base)
    try:
        with monitor:
            main(rid, ft1={'minimal': True, 'arm': 'A', 'seed': seed,
                           'devices': devices, 'scenario': 'F2', 'steps': 10,
                           'f2_ordinal': 2, 'formal_sample': False,
                           'dcp_diag': True, 'dcp_diag_presync': presync,
                           'dcp_diag_blocking_copy': blocking_copy,
                           'pinned_host_register': host_register,
                           'pointwise_autotune_off': pointwise_autotune_off,
                           'cuda_launch_blocking': cuda_launch_blocking,
                           'recovery_observation_seconds': 900})
    finally:
        if (base / rid).exists():
            write(base / rid / 'disk-peak.json', monitor.result())
    return base / rid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--devices', nargs=4, required=True)
    parser.add_argument('--presync', action='store_true')
    parser.add_argument('--cuda-launch-blocking', action='store_true')
    parser.add_argument('--blocking-copy', action='store_true')
    parser.add_argument('--host-register', action='store_true')
    parser.add_argument('--pointwise-autotune-off', action='store_true')
    args = parser.parse_args()
    print(run_diagnostic(args.name, args.seed, args.devices, args.presync,
                         args.cuda_launch_blocking, args.blocking_copy, args.host_register,
                         args.pointwise_autotune_off))
