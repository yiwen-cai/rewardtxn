"""Real worker/driver exits followed by >=30s evaluator, default control timeouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

SCRIPTS=Path(__file__).resolve().parents[2]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from ablation_control import Client

TERMINAL=dict(tasks=0,requests=0,writers=0)


def seal(root,client,name):
    artifact=root/(name+'.state.json')
    artifact.write_text(json.dumps(TERMINAL))
    manifest=root/(name+'.manifest.json')
    manifest.write_text(json.dumps(dict(nonce='walltime',fatal=False,
        files=[dict(path=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())])))
    permit=client.prepare_close(manifest,TERMINAL)
    client.close_transport()
    (root/(name+'.permit.json')).write_text(json.dumps(permit))
    return permit


def child(root,role):
    client=Client(str(root/'socket'),'walltime',role)
    if role=='evaluator':
        start=time.monotonic()
        time.sleep(30.1)
        snapshot=client.check()
        for name in ('rm-worker','driver'):
            assert snapshot['components'][name]['state']=='OFFLINE_CONFIRMED'
        (root/'evaluation_duration.json').write_text(json.dumps(dict(seconds=time.monotonic()-start)))
    seal(root,client,role)


def run(root):
    root.mkdir(parents=True,exist_ok=False)
    server=subprocess.Popen([sys.executable,str(SCRIPTS/'ablation_control.py'),'serve',
        '--socket',str(root/'socket'),'--nonce','walltime','--output',str(root/'control'),
        '--expected','driver,rm-worker,evaluator'])
    client=None
    try:
        deadline=time.monotonic()+5
        while not (root/'socket').exists():
            assert server.poll() is None and time.monotonic()<deadline
            time.sleep(.01)
        client=Client(str(root/'socket'),'walltime','launcher')
        permits=[]
        def worker():
            c=Client(str(root/'socket'),'walltime','rm-worker')
            permits.append(seal(root,c,'rm-worker'))
        thread=threading.Thread(target=worker)
        thread.start();thread.join(5)
        assert not thread.is_alive() and len(permits)==1
        client.confirm_exit('rm-worker',permits[0],dict(exited=True,oom=False,
            joined=True,thread_alive=False,**TERMINAL))
        for role in ('driver','evaluator'):
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),str(root),'--child',role],timeout=40)
            assert result.returncode==0
            permit=json.loads((root/(role+'.permit.json')).read_text())
            client.confirm_exit(role,permit,dict(exited=True,oom=False,waited=True,exitcode=result.returncode,**TERMINAL))
        duration=json.loads((root/'evaluation_duration.json').read_text())['seconds']
        assert duration>=30
        decision=client.finalize_run(['rm-worker','driver','evaluator'])
        receipt=client.shutdown_supervisor(decision)
        assert server.wait(timeout=5)==0
        assert json.loads((root/'control/shutdown.json').read_text())==receipt
        report=dict(passed=True,evaluation_seconds=duration,heartbeat_seconds=1,heartbeat_timeout_seconds=10,
            scope='CPU protocol: real RM thread join, real driver/evaluator process waits; no training',
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (SCRIPTS/'ablation_control.py',Path(__file__))})
        (root/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report))
    finally:
        if client: client.close_transport()
        if server.poll() is None: server.terminate();server.wait(timeout=5)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--child',choices=['driver','evaluator']);args=parser.parse_args()
    root=args.output.resolve()
    child(root,args.child) if args.child else run(root)
