"""Frozen FT1 one-fault paired pilots; no formal samples or alternate cuts."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_training_fault import REPO,main as run

CASES={'F4':(417,('A','R')),'F2':(419,('R','A')),'F1':(421,('A','R'))}


def fault_config(config,scenario,seed):
    if scenario not in CASES or seed!=CASES[scenario][0]:raise ValueError('unfrozen FT1 fault case')
    replacements={'total_train_steps: 3':'total_train_steps: 10','seed: 211':f'seed: {seed}'}
    for old,new in replacements.items():
        if config.count(old)!=1:raise ValueError('base configuration drift: '+old)
        config=config.replace(old,new)
    if config.count('retries: 1  # one native restart after one trainer kill')!=1:
        raise ValueError('common native retry configuration drift')
    return config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=tuple(CASES),required=True)
    parser.add_argument('--devices',nargs=4,required=True)
    args=parser.parse_args();seed,order=CASES[args.scenario]
    if len(set(args.devices))!=4 or any(not d.startswith('GPU-') for d in args.devices):parser.error('four distinct UUIDs required')
    if shutil.disk_usage(REPO).free<120*1024**3:raise RuntimeError('pair requires 120 GiB free')
    base=REPO/'docs/experiments/rewardtxn-ft-20260916/p3_evidence'
    names=[f'ft1-{args.scenario.lower()}-s{seed}-{arm.lower()}-r1' for arm in order]
    if any((base/name).exists() for name in names):raise RuntimeError('evidence exists; no overwrite or automatic rerun')
    for arm,name in zip(order,names):
        run(name,ft1={'arm':arm,'seed':seed,'devices':args.devices,'scenario':args.scenario,
                     'steps':10,'formal_sample':False,'recovery_observation_seconds':900})
        print(json.dumps({'evidence':str(base/name),'oracle_status':'pending independent audit'}),flush=True)


if __name__=='__main__':main()
