"""Serial, fresh-directory paired pilot for the minimal F2 or no-fault cell.

Formal seeds, order and retention must be frozen separately before formal use.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_fault import REPO, main as run, write
from check_ft_minimal_source import verify as verify_source
from minimal_storage import retain_or_clear


def minimal_config(config, scenario, seed):
    if scenario not in ('F2', 'F4T', 'F1', 'no_fault') or type(seed) is not int or seed < 0:
        raise ValueError('minimal experiment requires a fixed seed and F2/F4T/F1/no_fault')
    replacements = {'total_train_steps: 3': 'total_train_steps: 10', 'seed: 211': f'seed: {seed}'}
    if scenario == 'no_fault':
        replacements['retries: 1  # one native restart after one trainer kill'] = 'retries: 0'
    for old, new in replacements.items():
        if config.count(old) != 1:
            raise ValueError('base configuration drift: ' + old)
        config = config.replace(old, new)
    return config


class DiskMonitor:
    def __init__(self, path):
        self.path = path
        self.before = shutil.disk_usage(path).free
        self.minimum = self.before
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stopped.wait(0.5):
            self.minimum = min(self.minimum, shutil.disk_usage(self.path).free)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()
        self.minimum = min(self.minimum, shutil.disk_usage(self.path).free)

    def result(self):
        return {'free_before': self.before, 'free_minimum': self.minimum,
                'peak_incremental_bytes': max(0, self.before - self.minimum),
                'sample_interval_seconds': 0.5,
                'scope': 'training through independent input/chain/final-state load acceptance'}


def run_pair(name, scenario, seed, order, devices):
    if not re.fullmatch('[a-z0-9-]+', name) or scenario not in ('F2', 'no_fault'):
        raise ValueError('invalid pilot pair')
    if tuple(order) not in (('A', 'R'), ('R', 'A')):
        raise ValueError('explicit balanced arm order required')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not value.startswith('GPU-') for value in devices):
        raise ValueError('four distinct GPU UUIDs required')
    base = REPO / 'docs/experiments/rewardtxn-ft-20260916/minimal_evidence'
    base.mkdir(exist_ok=True)
    if shutil.disk_usage(base).free < 64 * 1024**3:
        raise RuntimeError('pilot requires at least 64 GiB free before its first run')
    pair = base / f'{name}-pair.json'
    names = [f'{name}-{arm.lower()}' for arm in order]
    if pair.exists() or any((base / rid).exists() for rid in names):
        raise FileExistsError('pilot evidence exists; rerun needs a new pair name')
    record = {'name': name, 'scenario': scenario, 'seed': seed, 'order': list(order),
              'devices': devices, 'formal_sample': False, 'status': 'running', 'runs': []}
    write(pair, record)
    for arm, rid in zip(order, names):
        root = base / rid
        if shutil.disk_usage(base).free < 64 * 1024**3:
            record['status'] = 'stopped_for_review'
            record['error'] = 'less than 64 GiB free before next pilot run'
            write(pair, record)
            raise RuntimeError(record['error'])
        case = {'minimal': True, 'arm': arm, 'seed': seed, 'devices': devices,
                'scenario': scenario, 'steps': 10, 'f2_ordinal': 2,
                'pointwise_autotune_off': True,
                'formal_sample': False, 'recovery_observation_seconds': 900}
        monitor = DiskMonitor(base)
        try:
            with monitor:
                run(rid, ft1=case)
            write(root / 'disk-peak.json', monitor.result())
            acceptance = json.loads((root / 'acceptance-status.json').read_text())
            if acceptance['result'] != 'functional_verification_written':
                raise RuntimeError('acceptance failed: ' + acceptance['result'])
            source = verify_source(root)
            write(root / 'source-verification.json', source)
            if scenario == 'F2' and arm == 'R' and not source['all_32_reused']:
                raise RuntimeError('R pilot did not retain all 32 original inputs')
            storage = retain_or_clear(root)
            record['runs'].append({'arm': arm, 'evidence': str(root), 'source': source,
                                   'disk_peak': monitor.result(), 'storage': storage['status']})
            write(pair, record)
        except Exception as exc:
            if root.exists() and not (root / 'disk-peak.json').exists():
                write(root / 'disk-peak.json', monitor.result())
            record['status'] = 'stopped_for_review'
            record['error'] = repr(exc)
            record['failed_arm'] = arm
            write(pair, record)
            raise
    record['status'] = 'pilot_pair_verified'
    write(pair, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--scenario', choices=('F2', 'no_fault'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--order', nargs=2, choices=('A', 'R'), required=True)
    parser.add_argument('--devices', nargs=4, required=True)
    args = parser.parse_args()
    print(json.dumps(run_pair(args.name, args.scenario, args.seed, args.order, args.devices), indent=2))
