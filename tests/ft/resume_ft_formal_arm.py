"""Rerun only the arm of a formal pair that a harness defect invalidated.

The other arm must already be accepted in the pair record. The invalid arm's
evidence directory is moved aside (kept intact) and the arm reruns from the
base model under an amended freeze that differs only in harness hashes.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_ft_formal import check_freeze, sha256, spent_gpu_hours
from run_training_fault import main as run, write
from run_ft_minimal import DiskMonitor
from check_ft_minimal_source import verify as verify_source
from minimal_storage import retain_or_clear
from gpu_load_monitor import GpuLoadMonitor
from scripts.ft.native_gpu import idle_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True, help='amended per-pair freeze')
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--pair-index', type=int, required=True)
    parser.add_argument('--arm', choices=['A', 'R'], required=True)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    assert sha256(freeze_path) == args.freeze_sha256
    freeze = json.loads(freeze_path.read_text())
    base = freeze_path.parent / 'minimal_evidence'
    item = freeze['pairs'][args.pair_index]
    name, arm = item['name'], args.arm
    pair_path = base / f'{name}-pair.json'
    record = json.loads(pair_path.read_text())
    other = 'R' if arm == 'A' else 'A'
    assert record['status'] == 'stopped_for_review' and record['failed_arm'] == arm
    assert [r['arm'] for r in record['runs']] == [other], 'other arm must be accepted'
    assert (record['seed'], record['order'], record['scenario']) == (item['seed'], item['order'], item['scenario'])
    root = base / f'{name}-{arm.lower()}'
    aside = base / f'{name}-{arm.lower()}-invalid1'
    assert root.exists() and not aside.exists()
    check_freeze(freeze, freeze_path.parent)
    root.rename(aside)
    subprocess.run(['docker', 'network', 'rm', f'rtx-{root.name}'], capture_output=True)
    history = {'arm': arm, 'invalid_evidence': str(aside), 'error': record.pop('error'),
               'reason': args.reason, 'arm_freeze': str(freeze_path),
               'arm_freeze_sha256': sha256(freeze_path),
               'script_sha256': sha256(Path(__file__))}
    record.pop('failed_arm')
    record['status'] = 'running'
    record.setdefault('arm_reruns', []).append(history)
    write(pair_path, record)
    try:
        if shutil.disk_usage(base).free < freeze['minimum_free_bytes']:
            raise RuntimeError('insufficient free space')
        idle_snapshot(freeze['devices'], base / f'{root.name}-gpu-preflight.json')
        case = {'minimal': True, 'arm': arm, 'seed': item['seed'], 'devices': freeze['devices'],
                'scenario': item['scenario'], 'steps': 10, 'f2_ordinal': 2,
                'pointwise_autotune_off': True, 'formal_sample': True,
                'recovery_observation_seconds': 900, 'perf_probe': False}
        monitor = DiskMonitor(base)
        gpu_monitor = (GpuLoadMonitor(root, freeze['devices'], freeze['gpu_hour_cap'] - spent_gpu_hours(freeze, base))
                       if freeze.get('monitor_external_gpu') else nullcontext())
        try:
            with monitor, gpu_monitor:
                run(root.name, ft1=case)
        finally:
            if root.exists():
                write(root / 'disk-peak.json', monitor.result())
        acceptance = json.loads((root / 'acceptance-status.json').read_text())
        if acceptance['result'] != 'functional_verification_written':
            raise RuntimeError(f'acceptance failed: {acceptance["result"]}')
        source = verify_source(root)
        write(root / 'source-verification.json', source)
        keep_full = args.pair_index == freeze['retain_full_pair_index']
        storage = retain_or_clear(root, keep_full=keep_full)
        result = json.loads((root / 'functional-verification.json').read_text())
        run_record = {'arm': arm, 'evidence': str(root), 'classification': result['classification'],
                      'source': source, 'disk_peak': monitor.result(), 'storage': storage['status']}
        record['runs'] = sorted(record['runs'] + [run_record], key=lambda r: item['order'].index(r['arm']))
        record['status'] = 'formal_pair_verified'
        write(pair_path, record)
        print(json.dumps({'pair': name, 'status': record['status']}), flush=True)
    except BaseException as exc:
        record.update(status='stopped_for_review', failed_arm=arm, error=repr(exc))
        write(pair_path, record)
        raise


if __name__ == '__main__':
    main()
