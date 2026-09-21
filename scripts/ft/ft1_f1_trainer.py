"""F1 group observation shared by both arms; markers never supply recovery data."""
import fcntl
import functools
import json
import os
from pathlib import Path
import threading
import time


def install(observer_module):
    from areal import workflow_context
    from areal.workflow.rlvr import RLVRWorkflow
    from scripts.ft.descendants import snapshot
    root=Path(os.environ['FT_CONTROL_SOCKET']).parent
    target=json.loads((root/'f1-target.json').read_text())
    assert target['source_row_id']==5518 and target['unique_tokenized_source_rows']==1
    trainer=snapshot(os.getpid())
    nonce=os.environ['FT_RUN_NONCE']
    lock=threading.Lock()
    active=None
    def write(name,value):
        path=root/name
        temp=path.with_suffix('.'+str(os.getpid())+'.tmp')
        temp.write_text(json.dumps(value));temp.replace(path)
    original=RLVRWorkflow._collect_samples
    @functools.wraps(original)
    async def collect(self,engine,req,prompt_str,task_data):
        nonlocal active
        if task_data['source_row_id']==target['source_row_id']:
            ctx=workflow_context.get()
            assert req.input_ids==target['input_tokens'],'target tokenization drift'
            with lock, (root/'f1-marker.lock').open('a') as marker_lock:
                fcntl.flock(marker_lock,fcntl.LOCK_EX)
                if active is None:
                    active={'trainer':trainer,'task_id':ctx.task_id,'source_row_id':5518,
                            'run_nonce':nonce,'prompt_sha256':target['prompt_sha256'],
                            'ambiguous':False,'monotonic_ns':time.monotonic_ns()}
                    write('f1-active.json',active)
                elif ctx.task_id!=active['task_id']:
                    active={**active,'ambiguous':True}
                    write('f1-active.json',active)
        return await original(self,engine,req,prompt_str,task_data)
    RLVRWorkflow._collect_samples=collect
    emit=observer_module.observe
    def observe(event,**fields):
        cut=time.monotonic_ns()
        emit(event,**fields)
        if event=='generation_complete' and fields.get('source_row_id')==5518:
            with lock, (root/'f1-marker.lock').open('a') as marker_lock:
                fcntl.flock(marker_lock,fcntl.LOCK_EX)
                if active is not None and not active['ambiguous'] and fields['task_id']==active['task_id']:
                    assert fields['input_tokens']==target['input_tokens']
                    # A real returned response, from this trainer and task.
                    write('f1-complete.json',{**active,'sample_idx':fields['sample_idx'],
                                             'monotonic_ns':cut})
    observer_module.observe=observe
