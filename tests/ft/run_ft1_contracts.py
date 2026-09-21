"""CPU-only real namespace/scorer contracts; scheduler and optimizer are fixtures."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from scripts.ft.container_run import supervise
from scripts.ft.ft1_fault_hooks import contract
IMAGE='sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    base=args.output.resolve();base.mkdir()
    files=[Path(__file__).resolve(),*REPO.glob('scripts/ft/ft1*.py'),REPO/'tests/ft/ft1_score_fault_fixture.py',REPO/'tests/ft/ft1_scheduler_fixture.py']
    (base/'source-sha256.json').write_text(json.dumps({str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    results=[]
    suites=[('score','ft1_score_fault_fixture.py',('F4','late','duplicate','cross_group','race','F2')),
            ('generator','ft1_scheduler_fixture.py',('hit','prefix_only','finished','stale_trainer','no_completed_response','already_claimed','ambiguous','wrong_source','stale_nonce','old_incarnation','dead_trainer','both_wrong_identity'))]
    for suite,fixture,modes in suites:
        for mode in modes:
            name=suite+'-'+mode;scenario='F1' if mode=='hit' else mode if mode in ('F2','F4') else None
            config={'argv':['/opt/.venv/bin/python','/workspace/tests/ft/'+fixture,mode],
                    'env':{'PYTHONPATH':'/workspace:/workspace/third_party/areal','PATH':'/opt/.venv/bin:/usr/bin:/bin','OMP_NUM_THREADS':'1','USER':'cpu','LOGNAME':'cpu'},
                    'timeouts':{'run':90,'handshake':10,'lease':20},'schedule':[contract(scenario)] if scenario else []}
            path=base/(name+'.json');path.write_text(json.dumps(config,indent=2))
            result=supervise(path,REPO,base/name,IMAGE,'/opt/.venv/bin/python',startup_timeout=30)
            assert result['failure'] is None and result['cleanup_confirmed'],result
            assert (base/name/'fixture-result.json').exists()
            events=[json.loads(line) for line in (base/name/'events.jsonl').read_text().splitlines()]
            assert sum(e['kind']=='signal_sent' for e in events)==(1 if scenario else 0)
            assert [e['launcher_exit_code'] for e in events if e['kind']=='method_observation']==[0]
            results.append({'case':name,'passed':True,'container_id':result['container_id']})
            print(name,'passed',flush=True)
    (base/'verification.json').write_text(json.dumps({'scope':__doc__,'cases':results},indent=2))


if __name__=='__main__':main()
