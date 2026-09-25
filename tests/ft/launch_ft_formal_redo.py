"""Perform the one permitted full-pair redo for formal F2 pair 2."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ft.native_gpu import idle_snapshot
from tests.ft.launch_ft_formal_dynamic import choose_devices


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--max-wait-seconds', type=int, default=172800)
    args = parser.parse_args()
    original = args.freeze.resolve()
    assert sha256(original) == args.freeze_sha256
    frozen = json.loads(original.read_text())
    base = original.parent
    old_name = frozen['pairs'][1]['name']
    failed = base / 'minimal_evidence' / f'{old_name}-pair.json'
    old = json.loads(failed.read_text())
    report = json.loads((base / 'minimal_evidence' / f'{old_name}-r' / 'fault-verification.json').read_text())
    assert old['status'] == 'stopped_for_review' and old['failed_arm'] == 'R' and old['runs'] == []
    assert old['order'] == ['R', 'A'] and old['seed'] == frozen['pairs'][1]['seed']
    assert report['classification'] == 'technical_invalid' and report['valid_hit'] is False
    name = f'{old_name}-redo1'
    assigned = base / f'FORMAL_FREEZE_{name}.json'
    pair = base / 'minimal_evidence' / f'{name}-pair.json'
    assert not pair.exists(), 'redo already started; manual review required'
    if assigned.exists():
        selected = json.loads(assigned.read_text())['devices']
    else:
        deadline = time.monotonic() + args.max_wait_seconds
        while True:
            if sha256(original) != args.freeze_sha256:
                raise RuntimeError('original formal freeze changed')
            try:
                selected = choose_devices(base / f'FORMAL_GPU_READY_{name}.json')
                break
            except (RuntimeError, subprocess.SubprocessError) as exc:
                if time.monotonic() >= deadline:
                    raise TimeoutError('GPU wait exceeded for full-pair redo') from exc
                print(f'waiting for four idle H100s for {name}: {exc}', flush=True)
                time.sleep(30)
        run_freeze = dict(frozen)
        run_freeze['pairs'] = [dict(item) for item in frozen['pairs']]
        run_freeze['pairs'][1]['name'] = name
        run_freeze['devices'] = selected
        run_freeze['training_gpu'] = selected[-1]
        run_freeze['gpu_assignment_amendment'] = {
            'original_freeze_sha256': args.freeze_sha256,
            'failed_pair': old_name,
            'reason': 'one full-pair redo after documented technical invalidity',
            'policy': 'any four simultaneously idle identical H100s, fixed within R/A pair'}
        run_freeze['extra_sha256'] = dict(frozen['extra_sha256'])
        for source in (Path(__file__), Path(__file__).with_name('launch_ft_formal_dynamic.py')):
            run_freeze['extra_sha256'][str(source.resolve().relative_to(Path.cwd().resolve()))] = sha256(source)
        assigned.write_text(json.dumps(run_freeze, indent=2) + '\n')
    while True:
        try:
            idle_snapshot(selected, base / f'FORMAL_GPU_READY_{name}.json')
            break
        except RuntimeError as exc:
            print(f'waiting for assigned redo GPUs: {exc}', flush=True)
            time.sleep(30)
    print(json.dumps({'starting_redo': name, 'devices': selected,
                      'assigned_freeze_sha256': sha256(assigned)}), flush=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name('run_ft_formal.py')),
                    '--freeze', str(assigned), '--freeze-sha256', sha256(assigned),
                    '--pair-index', '1'], check=True)


if __name__ == '__main__':
    main()
