"""Isolated four-GPU engineering probe; one controller launch, native retries."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.ft.native_gpu import idle_snapshot
from scripts.ft.areal_training_fault import EVENT, EVIDENCE


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main(rid, scenario='post-optimizer'):
    if scenario == 'midwrite':
        from scripts.ft.areal_midwrite_fault import EVENT, EVIDENCE
        entry = '/workspace/scripts/ft/areal_midwrite_fault.py'
    elif scenario == 'post-optimizer':
        from scripts.ft.areal_training_fault import EVENT, EVIDENCE
        entry = '/workspace/scripts/ft/areal_training_fault.py'
    else:
        raise ValueError('unsupported fault scenario')
    if not re.fullmatch('[a-z0-9-]+', rid):
        raise ValueError('invalid fresh run name')
    output = REPO / 'docs/experiments/rewardtxn-ft-20260916/p3_evidence' / rid
    output.mkdir()
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = [r[0].strip() for r in csv.reader(query.splitlines()) if float(r[1]) <= 100 and float(r[2]) == 0][:4]
    if len(devices) != 4:
        raise RuntimeError('four idle GPUs unavailable')
    idle_snapshot(devices, output / 'gpu-idle.json')
    write(output / 'gpu-uuids.json', devices)
    config = (REPO / 'docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml').read_text()
    config = config.replace('trial_name: native-trainer0', 'trial_name: r-fault-integration').replace('async_save: false', 'async_save: true')
    (output / 'training.yaml').write_text(config)
    for name in ('areal', 'name_resolve', 'tmp'):
        (output / name).mkdir()
    nonce = uuid.uuid4().hex
    env = {'PYTHONPATH':'/workspace:/workspace/third_party/areal', 'HOME':'/tmp',
        'USER':'caiyiwen','LOGNAME':'caiyiwen', 'PATH':'/opt/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin',
        'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','WANDB_MODE':'disabled',
        'AREAL_CACHE_DIR':'/tmp/areal-r','CUDA_HOME':'/usr/local/cuda','CUDA_VISIBLE_DEVICES':'0,1,2,3',
        'OMP_NUM_THREADS':'4','LD_LIBRARY_PATH':'/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
    contract = {'argv':['/opt/.venv/bin/python','-m','areal.infra.launcher.local',
                         entry,'--config','/output/training.yaml'],
        'env':env, 'schedule':[{'event_id':EVENT,'target':'trainer','waiters':['trainer'],'evidence':EVIDENCE}],
        'timeouts':{'run':1200,'handshake':10,'lease':20}}
    write(output / 'controller-config.json', contract)
    write(output / 'identity.json', {'nonce':nonce,'host_pidns':os.stat('/proc/self/ns/pid').st_ino})
    guardian = '''import hashlib,json,os,subprocess
from pathlib import Path
from scripts.ft.namespace_run import control
root=Path('/output')
code=2
try:
 frozen=json.loads((root/'source-sha256.json').read_text())
 assert all(hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
 with (root/'preflight.log').open('w') as log:
  subprocess.run(['/opt/.venv/bin/python','-m','scripts.ft.native_gpu','--preflight'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
 cfg=json.loads((root/'controller-config.json').read_text())
 from areal.infra.utils.launcher import BASE_ENVIRONS
 cfg['env'].update(BASE_ENVIRONS)
 identity=json.loads((root/'identity.json').read_text())
 digest=hashlib.sha256((root/'controller-config.json').read_bytes()).hexdigest()
 code=control(cfg,digest,root,identity['nonce'],identity['host_pidns'])
except BaseException as exc:
 print(repr(exc),flush=True)
finally:
 os._exit(code)
'''
    (output / 'guardian.py').write_text(guardian)
    files = [*REPO.glob('scripts/ft/*.py'), Path(__file__).resolve(),
        REPO/'tests/ft/check_training_fault.py', REPO/'tests/ft/check_training_fault_supplement.py',
        REPO/'third_party/areal/areal/infra/launcher/local.py',
        REPO/'third_party/areal/areal/trainer/rl_trainer.py',
        REPO/'third_party/areal/areal/utils/recover.py',
        REPO/'third_party/areal/areal/engine/megatron_utils/checkpointer.py']
    write(output / 'source-sha256.json', {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    image = 'sha256:c0573bb8412a8db753d44b02c9e04504d2392c54f019707f58acf26dd68d2469'
    network = subprocess.check_output(['docker','network','create','--internal','--label','rewardtxn.fault='+rid,'rtx-'+rid],text=True).strip()
    (output/'network.id').write_text(network)
    cid = None
    stopped = threading.Event()
    def lease():
        counter=0
        while not stopped.is_set():
            write(output/'host_lease.json',{'nonce':nonce,'counter':counter})
            counter+=1
            stopped.wait(.5)
    thread=threading.Thread(target=lease)
    started=time.monotonic()
    try:
        args=['docker','create','--network='+network,'--read-only','--cap-drop=ALL',
            '--security-opt=no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}',
            '--cpus=32','--memory=128g','--pids-limit=4096','--shm-size=16g','--workdir','/workspace',
            '--gpus','"device='+','.join(devices)+'"','--mount',f'type=bind,src={REPO},dst=/workspace,readonly',
            '--mount',f'type=bind,src={output},dst=/output','--mount',f'type=bind,src={output}/tmp,dst=/tmp']
        for k,v in env.items(): args.extend(['--env',k+'='+v])
        args.extend(['--entrypoint','/opt/.venv/bin/python',image,'/output/guardian.py'])
        write(output/'launch.json',args)
        cid=subprocess.check_output(args,text=True).strip(); (output/'container.id').write_text(cid)
        (output/'inspect-created.json').write_bytes(subprocess.check_output(['docker','inspect',cid]))
        idle_snapshot(devices,output/'gpu-idle-start.json')
        write(output/'host_lease.json',{'nonce':nonce,'counter':0})
        thread.start()
        subprocess.run(['docker','start',cid],check=True,capture_output=True)
        (output/'exitcode').write_text(subprocess.check_output(['docker','wait',cid],text=True,timeout=1400).strip())
    finally:
        stopped.set()
        if thread.ident: thread.join()
        if cid:
            info=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            if info['State']['Running']: subprocess.run(['docker','kill',cid],check=True)
            (output/'inspect-final.json').write_bytes(subprocess.check_output(['docker','inspect',cid]))
            r=subprocess.run(['docker','logs',cid],text=True,capture_output=True)
            (output/'container.log').write_text(r.stdout+r.stderr)
            subprocess.run(['docker','rm',cid],check=True,capture_output=True)
            r=subprocess.run(['docker','inspect',cid],text=True,capture_output=True)
            write(output/'cleanup.json',{'id':cid,'returncode':r.returncode,'stderr':r.stderr})
        subprocess.run(['docker','network','rm',network],check=True,capture_output=True)
        r=subprocess.run(['docker','network','inspect',network],text=True,capture_output=True)
        write(output/'network-cleanup.json',{'id':network,'returncode':r.returncode,'stderr':r.stderr})
        elapsed=time.monotonic()-started
        write(output/'cost.json',{'wall_seconds':elapsed,'allocated_gpu_hours':elapsed*4/3600,
                                 'scope':'engineering run; no comparative performance claim'})
        print('evidence',output)


if __name__=='__main__':
    main(*sys.argv[1:])
