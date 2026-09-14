#!/usr/bin/env python3
"""Two independent same-seed Oracle runs, sequentially on the same four GPUs."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    directory = ROOT / f'runs/oracle-repeat-s11-{stamp}'
    directory.mkdir(exist_ok=False)
    runs = [f'e7restart-oracle-repeat-r{i}-s11-{stamp}' for i in [1, 2]]
    files = ['scripts/e7_restart_run.sh', 'scripts/day2_slime_train.sh',
             'scripts/phase2_run.sh', 'scripts/day2_custom_rm.py',
             'third_party/slime/slime/rollout/fully_async_rollout.py']
    hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
    manifest = dict(purpose='same-method same-seed variability observation', seed=11,
                    group='group_rm', gpus=[1, 2, 3, 4], steps=500,
                    runs=runs, source_sha256=hashes, outcomes=[])
    path = directory / 'manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('MANIFEST', path, flush=True)
    env = dict(os.environ, RTX_FORMAL_GPUS='device=1,2,3,4')
    for run in runs:
        assert hashes == {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}, 'source changed between repeats'
        result = subprocess.run(['bash', str(ROOT / 'scripts/e7_restart_run.sh'),
                                 'pilot', '11', 'group_rm', run], cwd=ROOT, env=env)
        manifest['outcomes'].append(dict(run=run, exit_code=result.returncode))
        path.write_text(json.dumps(manifest, indent=2) + '\n')
    if any(r['exit_code'] for r in manifest['outcomes']):
        raise RuntimeError('One or more diagnostic runs failed; inspect the independent manifest.')
    evaluations = [read(ROOT / 'runs' / r / 'restart_eval/validation.json') for r in runs]
    a, b = [e['results'] for e in evaluations]
    assert [r['source_index'] for r in a] == [r['source_index'] for r in b]
    batches = [read(ROOT / 'runs' / r / 'diagnosis_summary.json')['consumed_rollouts'] for r in runs]
    groups = [{g for batch in bs for g in batch['groups']} for bs in batches]
    comparison = dict(runs=runs, accuracy_pp=[e['accuracy'] * 100 for e in evaluations],
                      r2_minus_r1_pp=100 * (evaluations[1]['accuracy'] - evaluations[0]['accuracy']),
                      r1_only_correct=sum(x['correct'] and not y['correct'] for x, y in zip(a, b)),
                      r2_only_correct=sum(y['correct'] and not x['correct'] for x, y in zip(a, b)),
                      shared_groups=len(groups[0] & groups[1]),
                      same_step_shared_group_fraction=sum(len(set(x['groups']) & set(y['groups']))
                                                          for x, y in zip(*batches)) / 2000,
                      limitation='Two repetitions at one seed are descriptive; they do not establish a variance estimate or explain directional method effects.')
    (directory / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
