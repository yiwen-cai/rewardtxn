"""One-shot F1 cut in the real GPU scheduler; normal batch selection unchanged."""
import fcntl
import json
import os
from pathlib import Path
import time


def attach(scheduler):
    from scripts.ft.descendants import Client,snapshot,exited
    root=Path(os.environ['FT_CONTROL_SOCKET']).parent
    attached_at=time.monotonic_ns()
    contract=json.loads((root/'f1-target.json').read_text())
    if contract['source_row_id']!=5518 or contract['unique_tokenized_source_rows']!=1:
        raise RuntimeError('F1 requires frozen unique first source prompt')
    original=scheduler.run_batch
    def claim(batch):
        if (root/'f1-claimed.json').exists():return None
        # Only marker snapshots are locked; no inference, scoring or fault wait.
        with (root/'f1-marker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            active_path=root/'f1-active.json';complete_path=root/'f1-complete.json'
            if not active_path.exists() or not complete_path.exists():return None
            active=json.loads(active_path.read_text());complete=json.loads(complete_path.read_text())
            if (active['ambiguous'] or complete['ambiguous']
                    or active['run_nonce']!=os.environ['FT_RUN_NONCE']
                    or complete['run_nonce']!=os.environ['FT_RUN_NONCE']
                    or active['trainer']!=complete['trainer']
                    or active['task_id']!=complete['task_id']
                    or active['source_row_id']!=5518 or complete['source_row_id']!=5518
                    or active['task_id']!=0 or active['monotonic_ns']<attached_at):return None
            try:
                fd=os.pidfd_open(active['trainer']['pid'],0)
                try:
                    if snapshot(active['trainer']['pid'])!=active['trainer'] or exited(fd):return None
                finally:os.close(fd)
            except (OSError,ValueError):return None
            requests=[req for req in batch.reqs if req.origin_input_ids==contract['input_tokens'] and not req.finished()]
            if not requests:return None
            cut=time.monotonic_ns()
            if complete['monotonic_ns']>=cut:raise RuntimeError('invalid F1 event order')
            import torch
            witness={'identity':snapshot(os.getpid()),'scheduler_attached_ns':attached_at,
                     'cut_monotonic_ns':cut,'completed':complete,'active':active,
                     'request_ids':[req.rid for req in requests],
                     'output_lengths':[len(req.output_ids) for req in requests],
                     'prompt_sha256':contract['prompt_sha256'],
                     'gpu_uuid':str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                     'tp_rank':scheduler.tp_rank,'dp_rank':scheduler.dp_rank,
                     'scope':'group-level full-prompt mapping; no sample_idx attribution'}
            try:
                with (root/'f1-claimed.json').open('x') as stream:
                    stream.write(json.dumps(witness));stream.flush();os.fsync(stream.fileno())
            except FileExistsError:return None
            return witness
    def run_batch(batch,*args,**kwargs):
        witness=claim(batch)
        if witness is None:return original(batch,*args,**kwargs)
        client=Client('generator',event_id='ft1-f1-generator')
        try:
            if client.injection['status']=='already_fired':return original(batch,*args,**kwargs)
            evidence={'phase':'generator_active_after_one_response','source_row_id':5518,'k':8}
            witness['incarnation']=client.incarnation
            (root/'f1-worker-witness.json').write_text(json.dumps(witness,indent=2))
            client.ready('ft1-f1-generator',evidence)
            client.wait_release('ft1-f1-generator')
            raise RuntimeError('SIGKILL target survived')
        finally:client.close()
    scheduler.run_batch=run_batch
