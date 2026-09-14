#!/usr/bin/env python3
"""Frozen diagnostic launcher. Explicit phases; no retry or checkpoint deletion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
SPEC=ROOT/'docs/experiments/rewardtxn-ablation-20260911/design.json'
IMAGE='slimerl/slime:v0.3.1'


def write(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def output(cmd): return subprocess.check_output(cmd,text=True).strip()


def source_hashes():
    spec=json.loads(SPEC.read_text())
    names=set(spec['source_sha256_at_design'])
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'third_party/slime').rglob('*')
                 if p.is_file() and p.suffix in ('.py','.sh','.json','.toml','.yaml','.yml','.cpp','.cu','.h')
                 and '.git' not in p.parts and '__pycache__' not in p.parts)
    names.update(str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('ablation_*.py'))
    names.update(['scripts/resource_gate.py','scripts/e7_diagnosis_summarize.py','scripts/e7_diagnosis_eval.py',
                  'scripts/e7_eval_checkpoint_incontainer.py','docs/experiments/rewardtxn-ablation-20260911/PLAN.md',str(SPEC.relative_to(ROOT))])
    return {n:digest(ROOT/n) for n in sorted(names)}


def prepare(batch):
    from ablation_overlay import build
    from resource_gate import _gate_check
    if batch.exists(): raise FileExistsError(batch)
    batch.mkdir(parents=True)
    check,ok=_gate_check('device=1,2,3,4',str(batch)); write(batch/'initial_resource.json',check)
    if not ok: raise RuntimeError('original resource gates failed')
    available=shutil.disk_usage(batch).free/2**30
    if available<620: raise RuntimeError('full-retention target filesystem requires >=620 GiB')
    overlay=batch/'slime'; build(overlay)
    model=ROOT/'models/Qwen2.5-0.5B-Instruct'
    data=[ROOT/'runs/diagnosis-20260910'/n for n in ['train.jsonl','train_split.json','validation_split.json']]
    data.append(ROOT/'models/datasets/gsm8k/dapo-gsm8k-train.jsonl')
    weights={str(p.relative_to(ROOT)):digest(p) for directory in [model,Path(str(model)+'_torch_dist')]
             for p in sorted(directory.rglob('*')) if p.is_file()}
    if not weights: raise RuntimeError('model files missing')
    freeze=dict(status='prepared_not_started',batch=str(batch),created=time.time(),
                source=source_hashes(),weights=weights,data={str(p.relative_to(ROOT)):digest(p) for p in data},
                image=output(['docker','image','inspect',IMAGE,'--format','{{.Id}}']),
                gpu_map=check['checks']['gpu_memory']['gpus'],gpus=[1,2,3,4],
                placement={'actor_container_gpu':0,'rollout_container_gpus':[1,2,3]},
                overlay=json.loads((overlay/'diagnostic_overlay.json').read_text()),
                overlay_sources={str(p.relative_to(overlay)):digest(p) for p in overlay.rglob('*')
                    if p.is_file() and p.suffix in ('.py','.sh','.json','.toml','.yaml','.yml','.cpp','.cu','.h') and '__pycache__' not in p.parts},
                storage={'strategy':'all_checkpoints_retained','path':str(batch),'initial_free_GiB':available,
                         'filesystem':output(['findmnt','-T',str(batch),'-J']),
                         'relocation':'all arms artifacts and CAS on root NVMe; new O/R reference; historical two-disk comparison descriptive only'},
                limits={'log_bytes':128*2**20,'db_bytes':128*2**20,'seen_ids':100000,'rss_bytes':16*2**30,'events':1000000},
                schedule=json.loads(SPEC.read_text())['schedule'])
    write(batch/'freeze.json',freeze)
    return freeze


def verify(batch):
    f=json.loads((batch/'freeze.json').read_text())
    if f['source']!=source_hashes(): raise RuntimeError('source drift; freeze new batch')
    for name,h in f['data'].items():
        if digest(ROOT/name)!=h: raise RuntimeError('data drift '+name)
    for name,h in f['weights'].items():
        if digest(ROOT/name)!=h: raise RuntimeError('model/tokenizer drift '+name)
    for name,h in f['overlay_sources'].items():
        if digest(batch/'slime'/name)!=h: raise RuntimeError('overlay drift '+name)
    return f


def docker_base(batch,run,gpus):
    return ['docker','run','--gpus','"device='+','.join(map(str,gpus))+'"','--shm-size=64g',
            '--ulimit','memlock=-1','--ulimit','stack=67108864',
            '-v',f'{ROOT}:/workspace:ro','-v',f'{batch}:{batch}',
            '-v',f'{batch}/slime:/root/slime','-v',f'{ROOT}/models:/root/models:ro',
            '-v',f'{run}/ray:/rtx-ray',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONUNBUFFERED=1',
            '-e','PYTHONPATH=/workspace/scripts:/root/slime:/root/Megatron-LM',
            '-w','/root/slime']


def environment(run,arm,seed,steps,nonce,sock,limits):
    oracle=arm in ('O','LITE')
    return {'NUM_GPUS':'4','CUDA_VISIBLE_DEVICES':'0,1,2,3','ACTOR_GPUS':'1','ROLLOUT_GPUS':'3',
            'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True',
            'ABLATION_RUN_DIR':str(run),'ABLATION_VARIANT':arm,'ABLATION_STEPS':str(steps),
            'ABLATION_RUN_NONCE':nonce,'ABLATION_CONTROL_SOCKET':str(sock),'ABLATION_RM_LIMITS':json.dumps(limits),
            'RTX_RUN_DIR':str(run),'RTX_CAS_INDEX_DIR':str(run/'cas'),
            'RTX_BASELINE_MODE':'b1','RTX_CUSTOM_RM':'ablation_rm.rm_function',
            'RTX_FAULT':'none','RTX_FAULT_START':'-1','RTX_FAULT_END':'-1','RTX_FAULT_WINDOWS':'',
            'RTX_SEAL':str(int(not oracle)),'RTX_SEAL_AUTO_FIX':str(int(not oracle)),'RTX_GROUP_RM':'1',
            'RTX_GROUP_SIZE':'8','RTX_SEED':str(seed),'RTX_NUM_ROLLOUT':'500',
            'RTX_SAVE_INTERVAL':'50','RTX_SAVE_HF':'1','RTX_NO_SAVE_OPTIM':'1','RTX_CKPT_KEEP':'0',
            'RTX_LR':'1e-6','RTX_FULLY_ASYNC':'1','RTX_TRAIN_ENTRY':'train_async.py',
            'RTX_MAX_TOKENS_PER_GPU':'3072','RTX_SGLANG_MEM_FRACTION_STATIC':'0.45','RTX_SGLANG_CONCURRENCY':'24',
            'RTX_PAPER_MODE':'0','RTX_RAY_TMP_DIR':'/rtx-ray',
            'RTX_RAY_SPILL_DIR':str(run/'spill'),'RTX_RAY_OBJECT_STORE_MEMORY':'17179869184',
            'MODEL_DIR':'/root/models/Qwen2.5-0.5B-Instruct','RTX_MODEL_CONFIG':'qwen2.5-0.5B.sh',
            'DATA_PATH':'/workspace/runs/diagnosis-20260910/train.jsonl','SAVE_DIR':str(run/'checkpoints'),
            'RTX_EXTRA_MODEL_ARGS':f'--use-rollout-logprobs --save-debug-rollout-data {run}/rollout_debug/{{rollout_id}}.pt --check-weight-update-equal'}


def wait_container(name,client,supervisor,run,training=False):
    next_resource=0
    next_gpu=0
    driver_confirmed=False
    while True:
        if supervisor.poll() is not None: raise RuntimeError('supervisor unexpectedly exited')
        client.check()
        state=json.loads(output(['docker','inspect',name,'--format','{{json .State}}']))
        if training and not driver_confirmed and (run/'driver_job_exit.json').exists():
            receipt=json.loads((run/'driver_job_exit.json').read_text())
            permit=json.loads((run/'driver_permit.json').read_text())
            client.confirm_exit('driver',permit,dict(exited=True,waited=receipt['waited'],exitcode=receipt['exitcode'],
                                oom=state['OOMKilled'],tasks=0,requests=0,writers=0,wait_source=receipt['kind']))
            driver_confirmed=True
        if not state['Running']:
            write(run/(name+'-terminal.json'),state)
            if state['ExitCode']!=0 or state['OOMKilled']: raise RuntimeError(('container failed',state))
            return state
        if time.monotonic()>=next_resource:
            if shutil.disk_usage(run).free/2**30<200:
                client.fatal('target_disk_below_200GiB'); raise RuntimeError('target disk gate')
            if shutil.disk_usage('/public').free/2**30<200:
                client.fatal('public_disk_below_200GiB'); raise RuntimeError('public storage gate')
            health={'time':time.time(),'target_free_GiB':shutil.disk_usage(run).free/2**30,
                    'public_free_GiB':shutil.disk_usage('/public').free/2**30,'load_average':os.getloadavg()}
            with (run/'logs/host_health.jsonl').open('a') as stream: stream.write(json.dumps(health)+'\n')
            next_resource=time.monotonic()+5
        if time.monotonic()>=next_gpu:
            from resource_gate import _nvidia_compute_processes
            frozen=json.loads((run.parents[1]/'freeze.json').read_text())
            own={int(line.strip()) for line in output(['docker','top',name,'-eo','pid']).splitlines()[1:] if line.strip().isdigit()}
            uuids={x['uuid'] for x in frozen['gpu_map']}
            foreign=[x for x in _nvidia_compute_processes() if 'error' in x or x.get('gpu_uuid') in uuids and x['pid'] not in own]
            if foreign:
                client.fatal('foreign_gpu_process',processes=foreign); raise RuntimeError(('GPU contention',foreign))
            next_gpu=time.monotonic()+60
        time.sleep(1)


def run_one(batch,arm,seed,steps,index,phase_override=None,total_runs=21):
    from resource_gate import _gate_check
    from ablation_control import Client
    f=verify(batch)
    phase=phase_override or ('smoke' if steps==10 else 'screen')
    run=batch/'runs'/f'{phase}-{index:02}-{arm}-s{seed}'
    if run.exists(): raise FileExistsError(run)
    run.mkdir(parents=True); (run/'logs').mkdir()
    gate,ok=_gate_check('device=1,2,3,4',str(run)); write(run/'resource_gate.json',gate)
    if not ok: raise RuntimeError('resource gate failed')
    target_free=shutil.disk_usage(run).free/2**30
    required=620 if steps==10 else 200+(total_runs-index)*19+20
    if target_free<required: raise RuntimeError(('target storage preflight',target_free,required))
    if [g['uuid'] for g in gate['checks']['gpu_memory']['gpus']] != [g['uuid'] for g in f['gpu_map']]:
        raise RuntimeError('GPU identities changed')
    nonce=uuid.uuid4().hex; sock=run/'control.sock'
    if len(str(sock))>100: raise RuntimeError('Unix socket path too long')
    env=environment(run,arm,seed,steps,nonce,sock,f['limits'])
    env['ABLATION_ACTOR_GPU_UUID']=f['gpu_map'][0]['uuid']
    write(run/'meta.json',dict(arm=arm,seed=seed,steps=steps,environment=env,image=f['image'],freeze_sha256=digest(batch/'freeze.json')))
    control_log=(run/'logs/control.log').open('w')
    supervisor=subprocess.Popen([sys.executable,str(ROOT/'scripts/ablation_control.py'),'serve','--socket',str(sock),
                                 '--nonce',nonce,'--output',str(run/'control'),'--expected','driver,rm-host,rm-worker,evaluator'],
                                stdout=control_log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+10
    while not sock.exists() and time.monotonic()<deadline:
        if supervisor.poll() is not None: raise RuntimeError('supervisor startup failed')
        time.sleep(.05)
    client=Client(str(sock),nonce,'launcher',role='launcher')
    name='rtxa-'+nonce[:12]; evalname=name+'-eval'; started=[]
    try:
        command=docker_base(batch,run,f['gpus'])+['-d','--name',name]
        for k,v in env.items(): command+=['-e',k+'='+v]
        command+=[f['image'],'bash','-c','pip install -e . --no-deps -q && bash /root/slime/diagnostic_day2.sh']
        write(run/'launch_command.json',command)
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL); started.append(name)
        with (run/'logs/train.log').open('w') as log:
            logger=subprocess.Popen(['docker','logs','-f',name],stdout=log,stderr=subprocess.STDOUT)
            state=wait_container(name,client,supervisor,run,training=True); logger.wait(timeout=10)
        command=docker_base(batch,run,[f['gpus'][0]])+['-d','--name',evalname]
        for k in ['ABLATION_RUN_DIR','ABLATION_STEPS','ABLATION_RUN_NONCE','ABLATION_CONTROL_SOCKET']:
            command+=['-e',k+'='+env[k]]
        command+=[f['image'],'python','/workspace/scripts/ablation_evaluate.py']
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL); started.append(evalname)
        with (run/'logs/evaluate.log').open('w') as log:
            logger=subprocess.Popen(['docker','logs','-f',evalname],stdout=log,stderr=subprocess.STDOUT)
            state=wait_container(evalname,client,supervisor,run); logger.wait(timeout=10)
        permit=json.loads((run/'evaluator_permit.json').read_text())
        client.confirm_exit('evaluator',permit,dict(exited=True,waited=True,exitcode=state['ExitCode'],oom=state['OOMKilled'],tasks=0,requests=0,writers=0))
        decision=client.finalize_run(['driver','rm-host','rm-worker','evaluator'])
        shutdown=client.shutdown_supervisor(decision)
        if supervisor.wait(timeout=10)!=0: raise RuntimeError('supervisor failed at final shutdown')
        if json.loads((run/'control/shutdown.json').read_text())!=shutdown: raise RuntimeError('shutdown receipt mismatch')
        write(run/'run_success.json',dict(decision=decision,shutdown=shutdown,status='technical_success',arm=arm,seed=seed,steps=steps))
        return str(run)
    except BaseException as exc:
        client.fatal('launcher_failure',error=repr(exc))
        write(run/'failure.json',dict(error=repr(exc),status='technical_failure',time=time.time()))
        for name in started:
            subprocess.run(['docker','stop','--time','5',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        raise
    finally:
        client.close_transport()
        if supervisor.poll() is None:
            supervisor.terminate()
            supervisor.wait(timeout=10)
        control_log.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['prepare','smoke','screen','confirm']); p.add_argument('--batch',required=True,type=Path)
    p.add_argument('--arm',choices=['O','R','DBM','LOGM','BOTHM','PAYLOAD','LITE']); a=p.parse_args()
    if a.phase=='prepare': prepare(a.batch); return
    f=verify(a.batch)
    gates=a.batch/'cpu_acceptance.json'
    if not gates.exists() or not json.loads(gates.read_text()).get('pass') or json.loads(gates.read_text()).get('source')!=f['source']: raise RuntimeError('CPU acceptance not available')
    if a.phase=='smoke':
        order=['O','R','DBM','LOGM','BOTHM','PAYLOAD','LITE']
        if a.arm: order=[a.arm]
        for i,arm in enumerate(order): run_one(a.batch,arm,11,10,i)
    elif a.phase=='screen':
        smoke=json.loads((a.batch/'smoke_acceptance.json').read_text())
        if (not smoke.get('pass') or smoke.get('source')!=f['source']
                or smoke.get('freeze_sha256')!=digest(a.batch/'freeze.json')):
            raise RuntimeError('seven-arm smoke acceptance failed or freeze mismatch')
        index=0
        for block in f['schedule']:
            for arm in block['order']:
                run_one(a.batch,arm,block['seed'],500,index); index+=1
    else:
        confirmation=json.loads((a.batch/'confirmation_plan.json').read_text())
        if confirmation['screen_report_sha256']!=digest(a.batch/'screen_report.json'):
            raise RuntimeError('confirmation evidence changed')
        if confirmation['source']!=f['source'] or confirmation['type']!='one_primary_intervention_contrast':
            raise RuntimeError('confirmation plan not frozen to this implementation')
        if len(confirmation['schedule'])!=12 or sum(x['order'][0]=='R' for x in confirmation['schedule'])!=6:
            raise RuntimeError('confirmation needs 12 paired new seeds, 6 R-first')
        if [x['seed'] for x in confirmation['schedule']]!=json.loads(SPEC.read_text())['confirmation_seeds']:
            raise RuntimeError('confirmation seed mismatch')
        index=0
        for block in confirmation['schedule']:
            if set(block['order'])!={'R',confirmation['candidate']}: raise RuntimeError('confirmation contrast mismatch')
            for arm in block['order']:
                run_one(a.batch,arm,block['seed'],500,index,phase_override='confirm',total_runs=24);index+=1


if __name__=='__main__': main()
