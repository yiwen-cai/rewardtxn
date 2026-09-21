import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from scripts.ft.descendants import snapshot
from scripts.ft.ft1_scheduler_observer import attach

root=Path(os.environ['FT_CONTROL_SOCKET']).parent
mode=sys.argv[1]
active={'trainer':snapshot(os.getpid()),'task_id':0,'source_row_id':5518,'run_nonce':os.environ['FT_RUN_NONCE'],'ambiguous':False}
complete={**active,'sample_idx':0,'monotonic_ns':time.monotonic_ns(),'run_nonce':os.environ['FT_RUN_NONCE']}
contract={'source_row_id':5518,'unique_tokenized_source_rows':1,'input_tokens':[11,12,13],'prompt_sha256':'fixture'}
if mode=='ambiguous':active['ambiguous']=True
if mode=='wrong_source':complete['source_row_id']=0
if mode=='stale_nonce':active['run_nonce']='old'
if mode=='stale_trainer':complete['trainer']={**active['trainer'],'start_time':'other'}
(root/'f1-target.json').write_text(json.dumps(contract))

def worker():
    # CPU contract mock only; real SGLang/GPU identity requires the GPU cell.
    sys.modules['torch']=SimpleNamespace(cuda=SimpleNamespace(current_device=lambda:0,get_device_properties=lambda n:SimpleNamespace(uuid='CPU-fixture')))
    calls=[]
    scheduler=SimpleNamespace(gpu_id=0,tp_rank=0,dp_rank=None,run_batch=lambda batch,*args,**kw:calls.append((batch,args,kw)) or 123)
    req=SimpleNamespace(origin_input_ids=[11,12] if mode=='prefix_only' else [11,12,13],
                        finished=lambda:mode=='finished',rid='real-local-fixture-id',output_ids=[14])
    batch=SimpleNamespace(reqs=[req])
    attach(scheduler)
    active['monotonic_ns']=time.monotonic_ns()
    complete['monotonic_ns']=time.monotonic_ns()
    if mode=='old_incarnation':active['monotonic_ns']=0
    if mode=='dead_trainer':
        active['trainer']={**active['trainer'],'pid':99999999}
        complete['trainer']=active['trainer']
    if mode=='both_wrong_identity':
        active['trainer']={**active['trainer'],'start_time':'0'}
        complete['trainer']=active['trainer']
    (root/'f1-active.json').write_text(json.dumps(active))
    if mode!='no_completed_response':(root/'f1-complete.json').write_text(json.dumps(complete))
    if mode=='already_claimed':(root/'f1-claimed.json').write_text('{}')
    assert scheduler.run_batch(batch,'extra',test=5)==123
    assert calls==[(batch,('extra',),{'test':5})]

p=multiprocessing.get_context('fork').Process(target=worker);p.start();p.join(20)
assert not p.is_alive()
expected=-9 if mode=='hit' else 0
assert p.exitcode==expected,(p.exitcode,expected)
(root/'fixture-result.json').write_text(json.dumps({'mode':mode,'child_exit_code':p.exitcode,'scope':'CPU fake scheduler contract'}))
