"""Run the frozen minimal FT formal pairs, one accepted run at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_fault import REPO, main as run, write
from run_ft_minimal import DiskMonitor
from check_ft_minimal_source import verify as verify_source
from minimal_storage import retain_or_clear
from scripts.ft.native_gpu import idle_snapshot


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_freeze(freeze, base):
    assert freeze['pilot_peak_bytes'] + 20 * 1024**3 == freeze['minimum_free_bytes']
    assert len(freeze['devices']) == len(set(freeze['devices'])) == 4
    if freeze.get('kind') in ('perf_gate', 'pilot'):
        # Engineering gates/pilots, never formal samples.
        assert freeze['formal_sample'] is False and freeze['retain_full_pair_index'] is None
        assert {p['scenario'] for p in freeze['pairs']} <= {'F2', 'F4T', 'F1', 'no_fault'}
        assert len({p['seed'] for p in freeze['pairs']}) == len(freeze['pairs']) <= 4
    else:
        assert freeze['formal_sample'] is True
        assert len(freeze['pairs']) == 13
        assert [p['scenario'] for p in freeze['pairs']] == ['F2'] * 10 + ['no_fault'] * 3
        assert len({p['seed'] for p in freeze['pairs']}) == 13
        assert 0 <= freeze['retain_full_pair_index'] < 10
    source_path = base / freeze['source_map_file']
    assert sha256(source_path) == freeze['source_map_sha256']
    for name, expected in json.loads(source_path.read_text()).items():
        assert sha256(REPO / name) == expected, f'source drift: {name}'
    for name, expected in freeze['extra_sha256'].items():
        assert sha256(REPO / name) == expected, f'input drift: {name}'


def run_pair(freeze_path, freeze, index):
    base = freeze_path.parent / 'minimal_evidence'
    item = freeze['pairs'][index]
    name = item['name']
    pair_path = base / f'{name}-pair.json'
    if pair_path.exists():
        old = json.loads(pair_path.read_text())
        if old.get('status') == 'formal_pair_verified' and old.get('freeze_sha256') == sha256(freeze_path):
            return old
        raise FileExistsError(f'existing incomplete or mismatched pair: {pair_path}')
    record = {'name': name, 'scenario': item['scenario'], 'seed': item['seed'],
              'order': item['order'], 'devices': freeze['devices'], 'formal_sample': freeze['formal_sample'],
              'freeze_sha256': sha256(freeze_path), 'status': 'running', 'runs': []}
    write(pair_path, record)
    for arm in item['order']:
        rid = f'{name}-{arm.lower()}'
        root = base / rid
        try:
            if root.exists():
                raise FileExistsError(root)
            check_freeze(freeze, freeze_path.parent)
            free = shutil.disk_usage(base).free
            if free < freeze['minimum_free_bytes']:
                raise RuntimeError(f'insufficient free space: {free} < {freeze["minimum_free_bytes"]}')
            idle_snapshot(freeze['devices'], base / f'{rid}-gpu-preflight.json')
            case = {'minimal': True, 'arm': arm, 'seed': item['seed'],
                    'devices': freeze['devices'], 'scenario': item['scenario'],
                    'steps': 10, 'f2_ordinal': 2, 'pointwise_autotune_off': True,
                    'formal_sample': freeze['formal_sample'], 'recovery_observation_seconds': 900,
                    'perf_probe': bool(freeze.get('perf_probe'))}
            monitor = DiskMonitor(base)
            try:
                with monitor:
                    run(rid, ft1=case)
            finally:
                if root.exists():
                    write(root / 'disk-peak.json', monitor.result())
            acceptance = json.loads((root / 'acceptance-status.json').read_text())
            if acceptance['result'] != 'functional_verification_written':
                raise RuntimeError(f'acceptance failed: {acceptance["result"]}')
            source = verify_source(root)
            write(root / 'source-verification.json', source)
            keep_full = item['scenario'] == 'F2' and index == freeze['retain_full_pair_index']
            storage = retain_or_clear(root, keep_full=keep_full)
            result = json.loads((root / 'functional-verification.json').read_text())
            record['runs'].append({'arm': arm, 'evidence': str(root),
                                   'classification': result['classification'],
                                   'source': source, 'disk_peak': monitor.result(),
                                   'storage': storage['status']})
            write(pair_path, record)
        except BaseException as exc:
            record.update(status='stopped_for_review', failed_arm=arm, error=repr(exc))
            write(pair_path, record)
            raise
    record['status'] = 'formal_pair_verified'
    write(pair_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--pair-index', type=int)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    if sha256(freeze_path) != args.freeze_sha256:
        raise RuntimeError('formal freeze SHA-256 differs from launch command')
    freeze = json.loads(freeze_path.read_text())
    base = freeze_path.parent / 'minimal_evidence'
    indices = [args.pair_index] if args.pair_index is not None else range(len(freeze['pairs']))
    for index in indices:
        if sha256(freeze_path) != args.freeze_sha256:
            raise RuntimeError('formal freeze changed during execution')
        check_freeze(freeze, freeze_path.parent)
        result = run_pair(freeze_path, freeze, index)
        print(json.dumps({'pair_index': index, 'name': result['name'],
                          'status': result['status']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
