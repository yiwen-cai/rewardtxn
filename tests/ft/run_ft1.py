"""FT1 no-fault pair only. Fault cells require separate verified mappings."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_fault import REPO, main as run, write
from check_ft1_smoke import verify


def smoke_config(config, seed):
    if seed not in (401, 409):
        raise ValueError('unfrozen FT1 pilot seed')
    replacements = {
        'total_train_steps: 3': 'total_train_steps: 10',
        'seed: 211': f'seed: {seed}',
        'retries: 1  # one native restart after one trainer kill': 'retries: 0',
    }
    for old, new in replacements.items():
        if config.count(old) != 1:
            raise ValueError(f'base configuration changed: {old}')
        config = config.replace(old, new)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(401, 409), required=True)
    parser.add_argument('--devices', nargs=4, required=True, help='same four UUIDs for both arms')
    args = parser.parse_args()
    if len(set(args.devices)) != 4 or any(not d.startswith('GPU-') for d in args.devices):
        parser.error('four distinct GPU UUIDs required')
    # R retains ten full generations (~70 GB); leave headroom for both arms.
    if shutil.disk_usage(REPO).free < 120 * 1024**3:
        raise RuntimeError('FT1 pair requires at least 120 GiB free disk')
    base = REPO/'docs/experiments/rewardtxn-ft-20260916/p3_evidence'
    order = ('A', 'R') if args.seed == 401 else ('R', 'A')
    names = [f'ft1-smoke-s{args.seed}-{arm.lower()}-r1' for arm in order]
    if any((base/name).exists() for name in names):
        raise RuntimeError('pair evidence already exists; preserve it and review before any rerun')
    for arm, name in zip(order, names):
        case = {'arm': arm, 'seed': args.seed, 'devices': args.devices,
                'scenario': 'no_fault', 'steps': 10, 'formal_sample': False}
        run(name, ft1=case)
        result = verify(base/name)
        write(base/name/'smoke-acceptance.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
