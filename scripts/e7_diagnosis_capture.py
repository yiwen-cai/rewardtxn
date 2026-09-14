#!/usr/bin/env python3
"""Capture Docker's authoritative full log without relying on orphaned nohup."""
import argparse
import json
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('run_name')
args = p.parse_args()
assert args.run_name.startswith('diagnosis-E7-') and '/' not in args.run_name
root = Path(__file__).resolve().parents[1]
container = 'rtx-p2-' + args.run_name
info = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]
assert info['Name'] == '/' + container
out = root/'runs'/args.run_name/'logs'
with (out/'train.log').open('wb') as stream:
    subprocess.run(['docker','logs','-f',container], stdout=stream, stderr=subprocess.STDOUT, check=True)
state = json.loads(subprocess.check_output(['docker','inspect',container]))[0]['State']
(out/'terminal_state.json').write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(state), flush=True)
