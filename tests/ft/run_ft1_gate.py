"""30-step F2 gate pilot for R_SLOWDOWN_FIX_PLAN v2 section 6.2; never a formal sample.

One pair, same seed and four GPUs, F2 killed after the 12th successful optimizer
update (FT-v1 section 6 steady-state cut). Criteria are fixed in GATE_CRITERIA
before any run and evaluated by tests/ft/check_ft1_gate.py.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_training_fault import REPO, main as run

SEED = 431          # not in formal seeds (211-317), FT4 (331/337/347) or FT1 pilots (401-421)
ORDER = ('R', 'A')
STEPS = 30
F2_ORDINAL = 12
GATE_CRITERIA = {
    'r_wall_seconds_max': 2700,             # FT-v1 section 6: 30-step run <= 45 minutes
    'r_step_interval_slope_max': 1.0,       # seconds per step, least squares over committed steps
    'r_recovery_full_hash_generations': 1,  # select_recovery content-hashes only the loaded head
    'acceptance_result': 'functional_verification_written',  # both arms
    'r_disk_peak_bytes_max': 45 * 1024 ** 3,
}


def gate_config(config):
    replacements = {'total_train_steps: 3': f'total_train_steps: {STEPS}', 'seed: 211': f'seed: {SEED}'}
    for old, new in replacements.items():
        if config.count(old) != 1:
            raise ValueError('base configuration drift: ' + old)
        config = config.replace(old, new)
    if config.count('retries: 1  # one native restart after one trainer kill') != 1:
        raise ValueError('common native retry configuration drift')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs=4, required=True)
    args = parser.parse_args()
    if len(set(args.devices)) != 4 or any(not d.startswith('GPU-') for d in args.devices):
        parser.error('four distinct UUIDs required')
    if shutil.disk_usage(REPO).free < 120 * 1024 ** 3:
        raise RuntimeError('gate pair requires 120 GiB free')
    base = REPO / 'docs/experiments/rewardtxn-ft-20260916/p3_evidence'
    names = [f'ft1-gate-f2-s{SEED}-{arm.lower()}-r1' for arm in ORDER]
    if any((base / name).exists() for name in names):
        raise RuntimeError('evidence exists; no overwrite or automatic rerun')
    import run_ft1_faults
    original = run_ft1_faults.fault_config
    run_ft1_faults.fault_config = lambda config, scenario, seed: gate_config(config)
    try:
        for arm, name in zip(ORDER, names):
            run(name, ft1={'arm': arm, 'seed': SEED, 'devices': args.devices, 'scenario': 'F2',
                           'steps': STEPS, 'f2_ordinal': F2_ORDINAL, 'formal_sample': False,
                           'gate': 'r-slowdown-fix-v2', 'gate_criteria': GATE_CRITERIA,
                           'recovery_observation_seconds': 900})
            print(json.dumps({'evidence': str(base / name), 'status': 'pending acceptance and gate check'}), flush=True)
    finally:
        run_ft1_faults.fault_config = original


if __name__ == '__main__':
    main()
