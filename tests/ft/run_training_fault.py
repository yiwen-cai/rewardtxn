"""Isolated four-GPU engineering probe; one controller launch, native retries."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
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


def main(rid, scenario='post-optimizer', *, ft1=None):
    if ft1 is not None:
        entry = '/workspace/scripts/ft/areal_ft1.py'
    elif scenario == 'midwrite':
        from scripts.ft.areal_midwrite_fault import EVENT, EVIDENCE
        entry = '/workspace/scripts/ft/areal_midwrite_fault.py'
    elif scenario == 'post-optimizer':
        from scripts.ft.areal_training_fault import EVENT, EVIDENCE
        entry = '/workspace/scripts/ft/areal_training_fault.py'
    else:
        raise ValueError('unsupported fault scenario')
    if not re.fullmatch('[a-z0-9-]+', rid):
        raise ValueError('invalid fresh run name')
    evidence_subdir = 'minimal_evidence' if ft1 is not None and ft1.get('minimal') is True else 'p3_evidence'
    output = REPO / 'docs/experiments/rewardtxn-ft-20260916' / evidence_subdir / rid
    output.parent.mkdir(exist_ok=True)
    output.mkdir()
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    devices = [r[0].strip() for r in csv.reader(query.splitlines()) if float(r[1]) <= 100 and float(r[2]) == 0][:4]
    if ft1 is not None:
        devices = ft1['devices']
        if len(set(devices)) != 4:
            raise ValueError('FT1 requires four fixed, distinct GPU UUIDs')
    if len(devices) != 4:
        raise RuntimeError('four idle GPUs unavailable')
    idle_snapshot(devices, output / 'gpu-idle.json')
    write(output / 'gpu-uuids.json', devices)
    config = (REPO / 'docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml').read_text()
    config = config.replace('trial_name: native-trainer0', 'trial_name: r-fault-integration').replace('async_save: false', 'async_save: true')
    if ft1 is not None:
        if ft1.get('minimal') is True:
            from run_ft_minimal import minimal_config
            config = minimal_config(config, ft1['scenario'], ft1['seed'])
        elif ft1['scenario']=='no_fault':
            from run_ft1 import smoke_config
            config=smoke_config(config,ft1['seed'])
        else:
            from run_ft1_faults import fault_config
            config=fault_config(config,ft1['scenario'],ft1['seed'])
        if ft1['scenario']=='F1':
            shutil.copy2(REPO/'docs/experiments/rewardtxn-ft-20260916/ft1-f1-target.json',output/'f1-target.json')
        write(output/'ft1-case.json',ft1)
    (output / 'training.yaml').write_text(config)
    for name in ('areal', 'name_resolve', 'tmp'):
        (output / name).mkdir()
    nonce = uuid.uuid4().hex
    env = {'PYTHONPATH':'/workspace:/workspace/third_party/areal', 'HOME':'/tmp',
        'USER':'caiyiwen','LOGNAME':'caiyiwen', 'PATH':'/opt/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin',
        'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','WANDB_MODE':'disabled',
        'AREAL_CACHE_DIR':'/tmp/areal-r','CUDA_HOME':'/usr/local/cuda','CUDA_VISIBLE_DEVICES':'0,1,2,3',
        'OMP_NUM_THREADS':'4','LD_LIBRARY_PATH':'/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
    schedule=[]
    if ft1 is not None:
        env['FT1_ARM']=ft1['arm']
        env['FT1_SCENARIO']=ft1['scenario']
        env['FT1_STEPS']=str(ft1.get('steps',10))
        if ft1.get('minimal') is True and not ft1.get('dcp_diag'):
            env['FT_MINIMAL_BLOCKING_D2H']='1'
        if ft1.get('minimal') is True and ft1.get('pointwise_autotune_off'):
            env['FT_MINIMAL_AUTOTUNE_POINTWISE']='0'
        if ft1.get('pinned_host_register'):
            env['PYTORCH_ALLOC_CONF']='pinned_use_cuda_host_register:True,pinned_num_register_threads:8'
        if ft1.get('dcp_diag'):
            env['FT_DCP_DIAG_PATH']='/output/dcp-diag.jsonl'
            env['FT_DCP_DIAG_PRESYNC']='1' if ft1.get('dcp_diag_presync') else '0'
            if ft1.get('dcp_diag_blocking_copy'):
                env['FT_DCP_DIAG_BLOCKING_COPY']='1'
        if ft1.get('cuda_launch_blocking'):
            env['CUDA_LAUNCH_BLOCKING']='1'
        if 'f2_ordinal' in ft1:env['FT1_F2_ORDINAL']=str(ft1['f2_ordinal'])
        if ft1['scenario']!='no_fault':
            from scripts.ft.ft1_fault_hooks import contract as fault_contract
            previous=os.environ.pop('FT1_F2_ORDINAL',None)
            if 'f2_ordinal' in ft1:os.environ['FT1_F2_ORDINAL']=str(ft1['f2_ordinal'])
            try:schedule=[fault_contract(ft1['scenario'])]
            finally:
                os.environ.pop('FT1_F2_ORDINAL',None)
                if previous is not None:os.environ['FT1_F2_ORDINAL']=previous
    contract = {'argv':['/opt/.venv/bin/python','-m','areal.infra.launcher.local',
                         entry,'--config','/output/training.yaml'],
        'env':env, 'schedule':schedule if ft1 is not None else [{'event_id':EVENT,'target':'trainer','waiters':['trainer'],'evidence':EVIDENCE}],
        # 30-step gate: FT-v1 section 6 caps a 30-step run at 45 minutes.
        'timeouts':{'run':(2700 if ft1.get('steps',10)==30 else 2400) if ft1 is not None else 1200,'handshake':10,'lease':20}}
    write(output / 'controller-config.json', contract)
    inputs = ['training.yaml', 'controller-config.json']
    if ft1 is not None and ft1['scenario'] == 'F1': inputs.append('f1-target.json')
    write(output / 'input-sha256.json', {name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in inputs})
    write(output / 'identity.json', {'nonce':nonce,'host_pidns':os.stat('/proc/self/ns/pid').st_ino})
    guardian = '''import hashlib,json,os,subprocess
from pathlib import Path
from scripts.ft.namespace_run import control
root=Path('/output')
code=2
try:
 frozen=json.loads((root/'source-sha256.json').read_text())
 assert all(hashlib.sha256((Path('/workspace')/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
 assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==sha for name,sha in json.loads((root/'input-sha256.json').read_text()).items())
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
    if ft1 is not None:
        files.extend(REPO/p for p in ('tests/ft/check_ft1_smoke.py','tests/ft/run_ft1.py','tests/ft/run_ft1_faults.py',
            'tests/ft/check_ft1_input_audit.py','tests/ft/check_ft1_load.py','tests/ft/check_ft1_chain.py',
            'tests/ft/check_ft1_fault.py','tests/ft/finalize_ft1_fault.py','tests/ft/run_ft1_acceptance.py','docs/experiments/rewardtxn-ft-20260916/ft1-f1-target.json',
            'third_party/areal/areal/v2/inference_service/sglang/scheduler.py',
            'third_party/areal/areal/api/reward_api.py','third_party/areal/areal/utils/strict_reward.py',
            'third_party/areal/areal/infra/remote_inf_engine.py','third_party/areal/areal/infra/workflow_executor.py',
            'third_party/areal/areal/reward/gsm8k.py','third_party/areal/areal/reward/__init__.py'))
        if ft1.get('minimal') is True:
            files.extend(REPO/p for p in ('tests/ft/run_ft_minimal.py',
                'tests/ft/check_ft_minimal_source.py', 'tests/ft/minimal_storage.py'))
            if ft1.get('dcp_diag'):
                files.append(REPO/'tests/ft/run_ft_dcp_diag.py')
    if ft1 is not None: files.extend((REPO/'third_party/areal/areal').rglob('*.py'))
    files = sorted(set(files))
    write(output / 'source-sha256.json', {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    for source in files:
        archived=output/'source-archive'/source.relative_to(REPO)
        archived.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,archived)
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
        (output/'exitcode').write_text(subprocess.check_output(['docker','wait',cid],text=True,timeout=(2900 if ft1.get('steps',10)==30 else 2600) if ft1 is not None else 1400).strip())
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
        if ft1 is not None:
            import sys as _sys
            _sys.path.insert(0, str(REPO / 'tests' / 'ft'))
            from run_ft1_acceptance import accept
            gpu = ft1['devices'][0]
            try:
                report = accept(output, gpu_uuid=gpu, run_docker=True)
            except Exception as exc:
                write(output/'acceptance-status.json', {
                    'result': 'acceptance_runner_exception',
                    'error': str(exc),
                })
                report = {'result': 'acceptance_runner_exception', 'error': str(exc)}
            print('acceptance', report.get('result'), flush=True)


if __name__=='__main__':
    main(*sys.argv[1:])
