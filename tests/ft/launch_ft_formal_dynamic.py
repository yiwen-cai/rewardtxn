"""Assign any four idle identical H100s to each remaining formal pair."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ft.native_gpu import idle_snapshot


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_devices(evidence):
    output = subprocess.run(
        ['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], check=True, capture_output=True,
        text=True, timeout=10).stdout
    rows = [[value.strip() for value in row] for row in csv.reader(output.splitlines())]
    eligible = [row[0] for row in rows if 'H100' in row[1]
                and float(row[2]) <= 100 and float(row[3]) == 0]
    for devices in itertools.combinations(eligible, 4):
        try:
            idle_snapshot(list(devices), evidence)
            return list(devices)
        except RuntimeError:
            continue
    raise RuntimeError('fewer than four idle H100 GPUs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--start-pair-index', type=int, default=1)
    parser.add_argument('--max-wait-seconds', type=int, default=43200)
    args = parser.parse_args()
    original = args.freeze.resolve()
    if sha256(original) != args.freeze_sha256:
        raise RuntimeError('original formal freeze SHA-256 mismatch')
    frozen = json.loads(original.read_text())
    base = original.parent
    for index in range(args.start_pair_index, len(frozen['pairs'])):
        name = frozen['pairs'][index]['name']
        assigned = base / f'FORMAL_FREEZE_{name}.json'
        pair = base / 'minimal_evidence' / f'{name}-pair.json'
        if pair.exists():
            record = json.loads(pair.read_text())
            if record.get('status') == 'formal_pair_verified' and sha256(assigned) == record['freeze_sha256']:
                continue
            raise RuntimeError(f'pair incomplete; manual review required: {pair}')
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
                        raise TimeoutError(f'GPU wait exceeded for {name}') from exc
                    print(f'waiting for four idle H100s for {name}: {exc}', flush=True)
                    time.sleep(30)
            run_freeze = dict(frozen)
            run_freeze['devices'] = selected
            run_freeze['training_gpu'] = selected[-1]
            run_freeze['gpu_assignment_amendment'] = {
                'original_freeze_sha256': args.freeze_sha256,
                'policy': 'any four simultaneously idle identical H100s, fixed within each A/R pair',
                'pair_index': index, 'pair_name': name}
            run_freeze['extra_sha256'] = dict(frozen['extra_sha256'])
            run_freeze['extra_sha256']['tests/ft/launch_ft_formal_dynamic.py'] = sha256(Path(__file__))
            assigned.write_text(json.dumps(run_freeze, indent=2) + '\n')
        while True:
            try:
                idle_snapshot(selected, base / f'FORMAL_GPU_READY_{name}.json')
                break
            except RuntimeError as exc:
                print(f'waiting for assigned pair GPUs for {name}: {exc}', flush=True)
                time.sleep(30)
        print(json.dumps({'starting_pair': name, 'devices': selected,
                          'assigned_freeze_sha256': sha256(assigned)}), flush=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name('run_ft_formal.py')),
                        '--freeze', str(assigned), '--freeze-sha256', sha256(assigned),
                        '--pair-index', str(index)], check=True)


if __name__ == '__main__':
    main()
