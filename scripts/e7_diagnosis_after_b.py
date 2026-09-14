#!/usr/bin/env python3
"""Bounded B -> C execution chain; never retries or launches long experiments."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = 'diagnosis-E7-B-s29-20260910'
container = 'rtx-p2-' + run
while True:
    state = json.loads(subprocess.check_output(['docker','inspect',container]))[0]['State']
    if state['Status'] == 'exited':
        if state['ExitCode'] != 0:
            raise SystemExit('B failed; C NOT launched')
        break
    if state['Status'] != 'running':
        raise SystemExit(f'Unexpected B state: {state}; C NOT launched')
    time.sleep(15)
print('B exited successfully; auditing all consumed batches', flush=True)
image = 'slimerl/slime:v0.3.1'
subprocess.run(['docker','run','--rm','-v',f'{root}:/workspace','-w','/workspace',image,
                'python3','scripts/e7_diagnosis_summarize.py',f'/workspace/runs/{run}'],check=True)
subprocess.run(['docker','run','--rm','-v',f'{root}/runs/{run}:/diagnosis',image,
                'chown','-R',f'{os.getuid()}:{os.getgid()}','/diagnosis'],check=True)
assert not (root/'runs/diagnosis-E7-C-s29-20260910').exists(), 'C directory exists; inspect before launching'
subprocess.run(['bash','scripts/e7_diagnosis_launch.sh','C'],cwd=root,check=True)
print('C launched; capturing authoritative log until terminal',flush=True)
subprocess.run([sys.executable,'scripts/e7_diagnosis_capture.py','diagnosis-E7-C-s29-20260910'],
               cwd=root,check=True)
