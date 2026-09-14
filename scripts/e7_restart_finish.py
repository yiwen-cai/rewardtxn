#!/usr/bin/env python3
"""Capture, audit and evaluate one completed or live restart run; never overwrite evaluations."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from e7_restart_checks import ROOT, SPEC, check_run, read, require


def call(args, **kwargs):
    return subprocess.run(args, check=True, cwd=ROOT, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run')
    p.add_argument('--stage', choices=['pilot', 'formal'], required=True)
    p.add_argument('--gpu', required=True)
    a = p.parse_args()
    require(a.run.startswith('e7restart-') and '/' not in a.run, 'invalid restart run name')
    spec = read(SPEC)
    run = ROOT / 'runs' / a.run
    image = read(run / 'meta.json')['image']
    container = 'rtx-p2-' + a.run
    with (run / 'logs/train.log').open('wb') as stream:
        call(['docker', 'logs', '-f', container], stdout=stream, stderr=subprocess.STDOUT)
    call(['docker', 'wait', container], stdout=subprocess.DEVNULL)
    state = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]['State']
    (run / 'logs/terminal_state.json').write_text(json.dumps(state, indent=2) + '\n')
    require(state['ExitCode'] == 0 and not state['OOMKilled'], 'training failed; see terminal_state.json')
    docker = ['docker', 'run', '--rm', '-v', f'{ROOT}:/workspace', '-w', '/workspace']
    call(docker + [image, 'python3', 'scripts/e7_diagnosis_summarize.py', f'/workspace/runs/{a.run}', '--expected-steps', '500'])
    call(['docker', 'run', '--rm', '-v', f'{run}:/run-output', image, 'chown', '-R', f'{os.getuid()}:{os.getgid()}', '/run-output'])
    call(['python3', 'scripts/resource_gate.py', 'check', '--gpus', f'device={a.gpu}', '--out', str(run / 'logs/eval_resource_gate.json')])
    # Evaluate all planned endpoints before applying quality gating; do not selectively omit failed models.
    for name, split in [('validation', spec['validation_split'])] + ([('test', spec['test_split'])] if a.stage == 'formal' else []):
        call(docker + ['--gpus', f'device={a.gpu}', image, 'python3', 'scripts/e7_diagnosis_eval.py',
                      '--checkpoint', f'/workspace/runs/{a.run}/checkpoints/iter_0000499_hf',
                      '--split', '/workspace/' + split, '--output', f'/workspace/runs/{a.run}/restart_eval/{name}.json'])
    # Evaluation creates new root-owned outputs after the earlier ownership repair.
    call(['docker', 'run', '--rm', '-v', f'{run / "restart_eval"}:/eval-output', image, 'chown', '-R', f'{os.getuid()}:{os.getgid()}', '/eval-output'])
    result = check_run(run, spec, a.stage)
    (run / 'restart_eval/quality.json').write_text(json.dumps(result, indent=2) + '\n')
    require(result['quality_pass'], 'fixed endpoint quality failed; no success classification')


if __name__ == '__main__':
    main()
