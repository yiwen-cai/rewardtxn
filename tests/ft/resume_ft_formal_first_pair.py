"""Finish the first formal pair after its accepted A arm and coordinator exit."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ft_formal import check_freeze, sha256
from run_training_fault import main as run, write
from run_ft_minimal import DiskMonitor
from check_ft_minimal_source import verify as verify_source
from minimal_storage import retain_or_clear
from scripts.ft.native_gpu import idle_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--max-wait-seconds', type=int, default=172800)
    args = parser.parse_args()
    freeze_path = args.freeze.resolve()
    assert sha256(freeze_path) == args.freeze_sha256
    freeze = json.loads(freeze_path.read_text())
    base = freeze_path.parent / 'minimal_evidence'
    first = freeze['pairs'][0]
    assert first['order'] == ['A', 'R']
    pair_path = base / f'{first["name"]}-pair.json'
    root_a = base / f'{first["name"]}-a'
    root_r = base / f'{first["name"]}-r'
    deadline = time.monotonic() + args.max_wait_seconds
    last_name = freeze['pairs'][-1]['name']
    while True:
        last_path = base / f'{last_name}-pair.json'
        if last_path.exists() and json.loads(last_path.read_text()).get('status') == 'formal_pair_verified':
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('remaining formal pairs did not complete')
        time.sleep(60)
    record = json.loads(pair_path.read_text())
    if record.get('status') == 'formal_pair_verified':
        return
    assert record['status'] == 'running' and record['order'] == first['order']
    assert record['freeze_sha256'] == args.freeze_sha256
    assert record['devices'] == freeze['devices'] and len(record['runs']) == 1
    assert record['runs'][0]['arm'] == 'A' and not root_r.exists()
    assert json.loads((root_a / 'ft1-case.json').read_text())['formal_sample'] is True
    assert json.loads((root_a / 'storage-cleanup.json').read_text())['status'] == 'shards_deleted'
    check_freeze(freeze, freeze_path.parent)
    while True:
        if shutil.disk_usage(base).free < freeze['minimum_free_bytes']:
            raise RuntimeError('insufficient free space for first-pair R')
        try:
            idle_snapshot(freeze['devices'], base / f'{first["name"]}-r-gpu-preflight.json')
            break
        except RuntimeError as exc:
            if time.monotonic() >= deadline:
                raise TimeoutError('first-pair GPUs never became idle') from exc
            print(f'waiting for first-pair GPUs: {exc}', flush=True)
            time.sleep(30)
    case = {'minimal': True, 'arm': 'R', 'seed': first['seed'],
            'devices': freeze['devices'], 'scenario': first['scenario'],
            'steps': 10, 'f2_ordinal': 2, 'pointwise_autotune_off': True,
            'formal_sample': True, 'recovery_observation_seconds': 900}
    monitor = DiskMonitor(base)
    try:
        try:
            with monitor:
                run(root_r.name, ft1=case)
        finally:
            if root_r.exists():
                write(root_r / 'disk-peak.json', monitor.result())
        acceptance = json.loads((root_r / 'acceptance-status.json').read_text())
        if acceptance['result'] != 'functional_verification_written':
            raise RuntimeError(f'R acceptance failed: {acceptance["result"]}')
        source = verify_source(root_r)
        write(root_r / 'source-verification.json', source)
        storage = retain_or_clear(root_r)
        result = json.loads((root_r / 'functional-verification.json').read_text())
        record['runs'].append({'arm': 'R', 'evidence': str(root_r),
                               'classification': result['classification'],
                               'source': source, 'disk_peak': monitor.result(),
                               'storage': storage['status']})
        record['coordinator_recovery']['resume_script_sha256'] = sha256(Path(__file__))
        record['status'] = 'formal_pair_verified'
        write(pair_path, record)
        print(json.dumps({'pair': first['name'], 'status': record['status']}), flush=True)
    except BaseException as exc:
        record.update(status='stopped_for_review', failed_arm='R', error=repr(exc))
        write(pair_path, record)
        raise


if __name__ == '__main__':
    main()
