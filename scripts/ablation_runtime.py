"""Diagnostic overlay hooks. Baseline files are never imported with global patches."""
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
import time

from ablation_control import Client

_driver = None
_host = None
_worker = None
_adapter = None
_worker_permit = None
_attempt = 0
_sample_attempts = {}


def run_dir():
    return Path(os.environ['ABLATION_RUN_DIR'])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fd=os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return str(path)


def component_manifest(name, terminal):
    state_path=Path(write_json(run_dir()/(name+'_state.json'),terminal))
    return write_json(run_dir()/(name+'_manifest.json'), {
        'run_nonce':os.environ['ABLATION_RUN_NONCE'], 'fatal':False,
        'files':[{'path':str(state_path),'sha256':hashlib.sha256(state_path.read_bytes()).hexdigest()}]})


def empty_terminal():
    return dict(tasks=0, requests=0, writers=0)


def training_steps(args):
    steps = int(os.environ['ABLATION_STEPS'])
    if args.num_rollout != 500 or steps not in (10,500):
        raise ValueError('only 10-step smoke or full 500 with num_rollout=500')
    return steps


def driver_start(args):
    global _driver
    _driver = Client.from_env('driver')
    training_steps(args)
    write_json(run_dir()/'driver_args.json', {'argv':sys.argv,
               'parsed':json.loads(json.dumps(vars(args),default=str))})
    _driver.event('driver_start', steps=training_steps(args), schedule_num_rollout=args.num_rollout)


def driver_check(step):
    _driver.check()
    _driver.event('training_step', step=step)


def driver_failed(exc):
    if _driver is not None:
        _driver.fatal('driver_exception', error=repr(exc))


def manager_start():
    global _host
    _host = Client.from_env('rm-host')
    _host.event('manager_started', pid=os.getpid())


def worker_start(worker, tasks):
    global _worker, _adapter
    import ablation_rm
    _worker = worker
    _adapter = ablation_rm.install()
    _adapter.observe('worker_start', pid=os.getpid())
    # Unexpected loop exceptions would otherwise be logged and swallowed.
    loop = asyncio.get_running_loop()
    def handler(loop, context):
        worker_failed(context.get('exception', RuntimeError(context.get('message'))))
    loop.set_exception_handler(handler)


def worker_failed(exc):
    if _adapter is not None:
        _adapter.client.fatal('worker_exception', error=repr(exc))
    elif _host is not None:
        _host.fatal('worker_start_exception', error=repr(exc))


def ids(samples):
    result=[]
    for sample in samples:
        if isinstance(sample,list): result.extend(ids(sample))
        else:
            result.append(':'.join(str(getattr(sample,k,None)) for k in ['group_index','index','rollout_id']))
    return result


async def generate_group(original, args, group, **kwargs):
    global _attempt
    import ablation_rm
    attempt = _attempt; _attempt += 1
    token = ablation_rm.RM_ATTEMPT_ID.set(attempt)
    _adapter.observe('attempt_start', attempt=attempt, ids=ids(group))
    try:
        result = await original(args, group, **kwargs)
        flat = [s for item in result for s in (item if isinstance(item,list) else [item])]
        for sample in flat: _sample_attempts[id(sample)] = attempt
        if len(_sample_attempts)>100000: raise MemoryError('attempt identity registry capacity')
        _adapter.observe('attempt_end', attempt=attempt, ids=ids(flat),
                         statuses=[str(s.status) for s in flat],
                         versions=[getattr(s,'weight_versions',None) for s in flat])
        return result
    except asyncio.CancelledError:
        _adapter.observe('attempt_cancel', attempt=attempt, ids=ids(group))
        if _worker.running:
            worker_failed(RuntimeError('unexpected active-task cancellation'))
        raise
    except BaseException as exc:
        worker_failed(exc)
        raise
    finally:
        ablation_rm.RM_ATTEMPT_ID.reset(token)


def queued(gid, result, size):
    _adapter.observe('queue_complete', gid=gid, ids=ids(result), queue_before=size)


def consumed(data, step):
    if _adapter is None:
        raise RuntimeError('consumption before diagnostic worker registration')
    flat=[s for item in data for s in (item if isinstance(item,list) else [item])]
    attempts=[_sample_attempts.pop(id(s)) for s in flat]
    _adapter.observe('consumed', step=step, ids=ids(data), attempt_ids=attempts)


def request_count(data):
    if isinstance(data,list):
        if not data: raise ValueError('empty load response')
        return sum(request_count(x) for x in data)
    if not isinstance(data,dict): raise ValueError('invalid load response')
    if 'loads' in data: return request_count(data['loads'])
    for key in ['num_reqs','num_total_reqs','total_reqs']:
        if key in data:
            if type(data[key]) is not int or data[key]<0: raise ValueError(data)
            return data[key]
    for a,b in [('num_running_reqs','num_waiting_reqs'),('total_running_reqs','total_waiting_reqs')]:
        if a in data and b in data:
            if any(type(data[k]) is not int or data[k]<0 for k in [a,b]): raise ValueError(data)
            return data[a]+data[b]
    raise ValueError('unrecognized remote load schema: '+repr(data))


async def remote_idle(args, cancel=False):
    from slime.utils.http_utils import get, post
    response = await get(f'http://{args.sglang_router_ip}:{args.sglang_router_port}/workers')
    urls = [x['url'] for x in response['workers']]
    if len(set(urls)) != 3: raise RuntimeError(('expected three rollout engines', urls))
    while True:
        if cancel:
            await asyncio.gather(*(post(url+'/abort_request', {'abort_all':True}) for url in urls))
        loads = await asyncio.gather(*(get(url+'/v1/loads?include=core') for url in urls))
        counts=[request_count(x) for x in loads]
        if not any(counts):
            _adapter.observe('remote_idle', urls=urls, loads=loads)
            return
        if not cancel: raise RuntimeError(('remote nonidle after local task drain',counts))
        await asyncio.sleep(.25)


async def worker_finalize(worker, tasks):
    global _worker_permit
    try:
        pending = {t for t in tasks if not t.done()}
        _adapter.observe('quiesce', tasks=len(pending), queue_size=worker.queue_size())
        if pending:
            _, pending = await asyncio.wait(pending, timeout=30)
        if pending:
            # Abort remote generation and await explicit zero-load acknowledgement.
            _adapter.begin_cancellation()
            worker.state.aborted = True
            _adapter.client.event('PHASE_BEGIN', phase='cancel', timeout=30)
            async def cancel_and_confirm():
                await remote_idle(worker.args, cancel=True)
                for task in pending: task.cancel()
                _, outstanding=await asyncio.wait(pending, timeout=1)
                if outstanding: raise TimeoutError('task ignored cancellation')
            await asyncio.wait_for(cancel_and_confirm(), timeout=30)
            _adapter.client.event('PHASE_END', phase='cancel')
        await asyncio.sleep(0)  # run completion callbacks before sealing events
        if any(not t.done() for t in tasks): raise RuntimeError('outstanding tasks at export')
        await asyncio.wait_for(remote_idle(worker.args), timeout=10)
        _adapter.client.check()
        import ablation_rm
        _adapter.client.event('PHASE_BEGIN', phase='export', timeout=120)
        ablation_rm.finalize_export(run_dir()/'diagnostic_export')
        _adapter.client.event('PHASE_END', phase='export')
        _worker_permit = _adapter.client.prepare_close(str(_adapter.manifest_path), empty_terminal())
        _adapter.client.close_transport()
        write_json(run_dir()/'worker_permit.json', _worker_permit)
    except BaseException as exc:
        worker_failed(exc)
        raise


def manager_finalize():
    _host.check()
    if _worker is None: raise RuntimeError('missing diagnostic worker')
    _worker.running = False
    _worker.worker_thread.join(timeout=240)
    if _worker.worker_thread.is_alive():
        _host.fatal('worker_join_timeout')
        raise TimeoutError('worker join timeout')
    if _worker_permit is None:
        _host.fatal('worker_export_missing')
        raise RuntimeError('no worker export permit')
    _host.confirm_exit('rm-worker', _worker_permit,
                       dict(exited=True, oom=False, joined=True, thread_alive=False, **empty_terminal()))
    return {'manifest':str(_adapter.manifest_path),'pid':os.getpid()}


def manager_close():
    path=component_manifest('manager_closed', dict(pid=os.getpid(), **empty_terminal()))
    permit = _host.prepare_close(path,empty_terminal())
    _host.close_transport()
    import ray
    context=ray.get_runtime_context()
    return {'pid':os.getpid(), 'starttime':Path('/proc/self/stat').read_text().split()[21],
            'actor_id':context.get_actor_id(), 'worker_id':context.get_worker_id(), 'permit':permit}


def manager_exit(permit):
    import ray
    if permit != _host.permit:
        raise RuntimeError('exit permit does not match registered host')
    ray.actor.exit_actor()


def driver_finalize(manager):
    import ray
    _driver.check()
    summary=ray.get(manager.diagnostic_finalize.remote(), timeout=250)
    close=ray.get(manager.diagnostic_close.remote(), timeout=10)
    from google.protobuf.json_format import MessageToDict
    from ray.core.generated import gcs_pb2
    accessor=ray._private.state.state._connect_and_get_accessor()
    manager.diagnostic_exit.remote(close['permit'])
    deadline=time.monotonic()+9
    while time.monotonic()<deadline:
        a=gcs_pb2.ActorTableData.FromString(accessor.get_actor_info(ray.ActorID.from_hex(close['actor_id'])))
        w=gcs_pb2.WorkerTableData.FromString(accessor.get_worker_info(ray.WorkerID.from_hex(close['worker_id'])))
        try:
            fields=Path(f"/proc/{close['pid']}/stat").read_text().split()
            gone=fields[21]!=close['starttime'] or fields[2]=='Z'
        except FileNotFoundError: gone=True
        if a.state==gcs_pb2.ActorTableData.DEAD and not w.is_alive and gone: break
        time.sleep(.05)
    else: raise TimeoutError('RM host controlled exit not confirmed')
    actor=MessageToDict(a,preserving_proto_field_name=True)
    worker=MessageToDict(w,preserving_proto_field_name=True)
    cause=actor.get('death_cause',{})
    if cause.get('oom_context') or cause.get('actor_died_error_context',{}).get('reason')!='WORKER_DIED':
        raise RuntimeError(('unexpected actor exit cause',cause))
    if w.pid!=close['pid']: raise RuntimeError('worker identity mismatch')
    evidence=dict(exited=True,oom=False,exit_kind='ray_intended_actor_exit',
                  worker_id=close['worker_id'],actor_id=close['actor_id'],is_alive=w.is_alive,
                  exit_type=worker.get('exit_type'),exit_detail=w.exit_detail,
                  actor_state=actor.get('state'),num_restarts=a.num_restarts,registered_process_gone=gone,
                  raw_actor=actor,raw_worker=worker,**empty_terminal())
    write_json(run_dir()/'rm_host_exit_evidence.json',evidence)
    _driver.confirm_exit('rm-host',close['permit'],evidence)
    _driver.event('rm_host_closed', manifest=summary['manifest'])


def driver_close():
    path=component_manifest('driver_closed', empty_terminal())
    permit=_driver.prepare_close(path,empty_terminal())
    write_json(run_dir()/'driver_permit.json',permit)
    _driver.close_transport()


def record_scheduler(args, scheduler):
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    import torch
    properties=torch.cuda.get_device_properties(torch.cuda.current_device())
    gpu_uuid=str(properties.uuid)
    if not gpu_uuid.startswith('GPU-'): gpu_uuid='GPU-'+gpu_uuid
    if gpu_uuid!=os.environ['ABLATION_ACTOR_GPU_UUID']:
        raise RuntimeError(('actor physical GPU mismatch',gpu_uuid))
    values = {'gpu_uuid':gpu_uuid,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'rank':rank, 'train_iters':args.train_iters, 'num_rollout':args.num_rollout,
              'scheduler':{k:v for k,v in vars(scheduler).items() if isinstance(v,(str,int,float,bool,type(None)))}}
    write_json(run_dir()/f'scheduler_rank{rank}.json',values)
    if args.train_iters != 500 or args.lr_decay_iters != 500:
        raise RuntimeError('diagnostic scheduler changed from 500 steps')


def record_engine(rank, args):
    write_json(run_dir()/f'engine_rank{rank}.json', {'rank':rank, 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'server_args':{k:v for k,v in args.items() if isinstance(v,(str,int,float,bool,type(None),list,dict))}})
