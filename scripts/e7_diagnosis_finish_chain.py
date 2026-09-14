#!/usr/bin/env python3
"""Bounded E -> 500-step Oracle -> 500-step RewardTxn confirmation.

Stops on failed execution, failed audit, failed quality gate or occupied GPUs.
No automatic retry, deletion, margin relaxation or paper experiment launch.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = 'slimerl/slime:v0.3.1'
DIAG = ROOT/'runs/diagnosis-20260910'


def run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=ROOT, check=True, **kwargs)


def state(name):
    return json.loads(subprocess.check_output(['docker','inspect',name]))[0]['State']


def wait(name):
    while True:
        s = state(name)
        if s['Status'] == 'exited':
            if s['ExitCode'] != 0:
                raise RuntimeError(f'{name} failed: {s}')
            return s
        if s['Status'] != 'running':
            raise RuntimeError(f'Unexpected state, inspect before retry: {name}: {s}')
        time.sleep(15)


def docker_cpu(command):
    run(['docker','run','--rm','-v',f'{ROOT}:/workspace','-w','/workspace',IMAGE,*command])


def audit(name, steps):
    print(f'Auditing {name}',flush=True)
    docker_cpu(['python3','scripts/e7_diagnosis_summarize.py',f'/workspace/runs/{name}',
                '--expected-steps',str(steps)])
    evidence=json.loads((ROOT/'runs'/name/'diagnosis_summary.json').read_text())
    assert evidence['train_steps']==list(range(steps))
    assert evidence['all_logged_metrics_finite'] and evidence['consumed_split_overlap']==0
    assert len(evidence['consumed_rollouts'])==steps
    run(['docker','run','--rm','-v',f'{ROOT}/runs/{name}:/diagnosis',IMAGE,
         'chown','-R',f'{os.getuid()}:{os.getgid()}','/diagnosis'])


def evaluate(name, steps):
    # All evaluations finish before allocating the four training GPUs again.
    run([sys.executable,'scripts/resource_gate.py','check','--gpus','device=0,5',
         '--out',str(ROOT/'runs'/name/'logs/eval_resource_gate.json')])
    jobs=[]
    for gpu, points, suffix in [(0,[(steps,100)],'final'),(5,[(x,20) for x in [5,10,20,30]],'interim')]:
        if steps == 500 and suffix == 'interim':
            # Long confirmation has checkpoints every50, not the diagnostic5.
            points=[(50,20),(100,20),(250,20)]
        commands=[]
        for completed,n in points:
            ckpt=f'/workspace/runs/{name}/checkpoints/iter_{completed-1:07d}_hf'
            output=f'/workspace/runs/{name}/diagnostic_eval/step{completed}_n{n}.json'
            assert not (ROOT/'runs'/name/'diagnostic_eval'/f'step{completed}_n{n}.json').exists()
            commands.append(f'python3 scripts/e7_diagnosis_eval.py --checkpoint {ckpt} '
                            f'--split /workspace/runs/diagnosis-20260910/validation_split.json '
                            f'--limit {n} --output {output}')
        container=f'rtx-confirm-{name}-{suffix}'
        run(['docker','run','-d','--name',container,'--gpus',f'device={gpu}','--shm-size','8g',
             '-v',f'{ROOT}:/workspace','-w','/workspace',IMAGE,'bash','-c',' && '.join(commands)])
        jobs.append(container)
    for job in jobs:
        wait(job)
    print(f'Evaluations completed: {name}',flush=True)


def quality(name, steps):
    d=json.loads((ROOT/'runs'/name/f'diagnostic_eval/step{steps}_n100.json').read_text())
    base=json.loads((DIAG/'base_validation.json').read_text())
    assert d['n_total']==100 and d['split_sha256']==base['split_sha256']
    assert d['checkpoint']==f'/workspace/runs/{name}/checkpoints/iter_{steps-1:07d}_hf'
    assert [x['source_index'] for x in d['results']]==[x['source_index'] for x in base['results']]
    assert d['n_correct']==sum(x['correct'] for x in d['results'])
    passed=d['n_correct']>=base['n_correct']-10 and d['truncated_fraction']<=base['truncated_fraction']+.1+1e-12
    print(f'QUALITY {name}: {d["n_correct"]}/100 truncated={d["truncated_fraction"]} pass={passed}',flush=True)
    if not passed:
        raise RuntimeError(f'Quality gate failed for {name}; no subsequent training launched')


def main():
    name='diagnosis-E7-E-s29-20260910'
    wait('rtx-p2-'+name)
    audit(name,50)
    evaluate(name,50)
    quality(name,50)
    for group in ['oracle','rewardtxn']:
        env=dict(os.environ,RTX_GPUS='device=0,5,6,7')
        run(['bash','scripts/e7_diagnosis_stability.sh',group],env=env)
        name=f'diagnosis-E7-D3-{group}-s29-20260910'
        run([sys.executable,'scripts/e7_diagnosis_capture.py',name])
        wait('rtx-p2-'+name)
        audit(name,500)
        evaluate(name,500)
        quality(name,500)
    print('Confirmation execution completed; independent report/completion audit still required.',flush=True)


if __name__=='__main__':
    main()
