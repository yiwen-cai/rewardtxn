"""Wait for the frozen four GPUs, then start the formal serial runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ft.native_gpu import idle_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--max-wait-seconds', type=int, default=43200)
    args = parser.parse_args()
    freeze = args.freeze.resolve()
    if hashlib.sha256(freeze.read_bytes()).hexdigest() != args.freeze_sha256:
        raise RuntimeError('formal freeze SHA-256 differs from launch command')
    devices = json.loads(freeze.read_text())['devices']
    deadline = time.monotonic() + args.max_wait_seconds
    attempt = 0
    while time.monotonic() < deadline:
        if hashlib.sha256(freeze.read_bytes()).hexdigest() != args.freeze_sha256:
            raise RuntimeError('formal freeze changed while waiting for GPUs')
        try:
            snapshot = freeze.parent / 'FORMAL_GPU_PREFLIGHT_LAST_20260924.json'
            idle_snapshot(devices, snapshot)
        except Exception as exc:
            attempt += 1
            if attempt == 1 or attempt % 10 == 0:
                print(f'waiting for frozen GPUs: {exc}', flush=True)
        else:
            snapshot.replace(freeze.parent / 'FORMAL_GPU_READY_20260924.json')
            print('all frozen GPUs idle; starting formal runner', flush=True)
            os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('run_ft_formal.py')),
                                    '--freeze', str(freeze), '--freeze-sha256', args.freeze_sha256])
        time.sleep(30)
    raise TimeoutError('frozen GPUs were not all idle within the wait limit')


if __name__ == '__main__':
    main()
