"""CPU integration: real control socket + runtime owner-thread export, no GPU."""
import asyncio
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import ablation_control as ctl
import ablation_runtime as rt


class RuntimeTests(unittest.TestCase):
    def test_cancellation_confirmation_and_deadline(self):
        async def case(remote_delay, should_fail):
            task=asyncio.create_task(asyncio.sleep(60))
            adapter=Mock(manifest_path='unused')
            worker=SimpleNamespace(args=None,state=SimpleNamespace(aborted=False),queue_size=lambda:0)
            original_wait,original_wait_for=asyncio.wait,asyncio.wait_for
            async def wait(tasks,timeout):
                # Skip the natural-drain interval; retain real task cancellation.
                if timeout==30: return set(),set(tasks)
                return await original_wait(tasks,timeout=timeout)
            async def wait_for(coro,timeout):
                return await original_wait_for(coro,timeout=.02 if timeout==30 else timeout)
            async def remote(*args,**kwargs):
                if kwargs.get('cancel'): await asyncio.sleep(remote_delay)
            export=Mock()
            with patch.object(rt,'_adapter',adapter),patch.object(rt,'remote_idle',remote),\
                    patch.object(rt.asyncio,'wait',wait),patch.object(rt.asyncio,'wait_for',wait_for),\
                    patch.object(rt,'worker_failed') as failed,patch.object(rt,'write_json'),\
                    patch.dict(sys.modules,ablation_rm=SimpleNamespace(finalize_export=export)):
                try:
                    if should_fail:
                        with self.assertRaises(asyncio.TimeoutError):
                            await rt.worker_finalize(worker,{task})
                        failed.assert_called_once()
                        export.assert_not_called()
                        adapter.client.prepare_close.assert_not_called()
                    else:
                        await rt.worker_finalize(worker,{task})
                        self.assertTrue(task.cancelled())
                        export.assert_called_once()
                        adapter.client.prepare_close.assert_called_once()
                    adapter.begin_cancellation.assert_called_once()
                    self.assertTrue(worker.state.aborted)
                finally:
                    task.cancel()
                    await asyncio.gather(task,return_exceptions=True)
                    rt._worker_permit=None
        with tempfile.TemporaryDirectory() as td,patch.dict(os.environ,ABLATION_RUN_DIR=td):
            asyncio.run(case(0,False))
            asyncio.run(case(.1,True))

    def test_strict_remote_load(self):
        self.assertEqual(rt.request_count({'loads':[{'num_running_reqs':0,'num_waiting_reqs':0}]}),0)
        self.assertEqual(rt.request_count({'num_reqs':3}),3)
        for value in [{},[],{'num_reqs':'0'},{'num_reqs':-1},{'loads':[{}]}]:
            with self.assertRaises(ValueError): rt.request_count(value)

    def test_owner_thread_long_tail_export(self):
        with tempfile.TemporaryDirectory(prefix='rtxa-cpu-') as td:
            root=Path(td); sock=root/'s'; nonce='runtime-test'
            proc=multiprocessing.Process(target=ctl.serve,args=(str(sock),nonce,str(root/'control')))
            proc.start()
            launcher=None
            try:
                deadline=time.monotonic()+5
                while not sock.exists() and time.monotonic()<deadline: time.sleep(.01)
                with patch.dict(os.environ,ABLATION_CONTROL_SOCKET=str(sock),ABLATION_RUN_NONCE=nonce,ABLATION_RUN_DIR=td):
                    launcher=ctl.Client(str(sock),nonce,'launcher',role='launcher')
                    rt.manager_start()
                    state={}
                    class Adapter:
                        def observe(self,*a,**k): pass
                    adapter=Adapter()
                    def install():
                        adapter.client=ctl.Client.from_env('rm-worker')
                        state['owner']=threading.get_ident()
                        return adapter
                    def export(output_dir):
                        self.assertEqual(state['owner'],threading.get_ident())
                        self.assertTrue(state.get('tail_complete'))
                        adapter.manifest_path=rt.component_manifest('rm_export',dict(tasks=0,requests=0,writers=0))
                    async def idle(*a,**k): return None
                    worker=SimpleNamespace(running=True,args=SimpleNamespace(),queue_size=lambda:0)
                    def main():
                        async def loop():
                            tasks=set(); rt.worker_start(worker,tasks)
                            async def tail():
                                await asyncio.sleep(5.2)
                                state['tail_complete']=True
                            tasks.add(asyncio.create_task(tail()))
                            worker.running=False
                            await rt.worker_finalize(worker,tasks)
                        asyncio.run(loop())
                    with patch.dict(sys.modules,ablation_rm=SimpleNamespace(install=install,finalize_export=export)),patch.object(rt,'remote_idle',idle):
                        worker.worker_thread=threading.Thread(target=main)
                        began=time.monotonic(); worker.worker_thread.start()
                        while rt._worker is not worker: time.sleep(.01)
                        result=rt.manager_finalize()
                        self.assertGreater(time.monotonic()-began,5)
                        self.assertTrue(Path(result['manifest']).exists())
                        self.assertFalse(worker.worker_thread.is_alive())
                        snapshot=launcher.check()
                        self.assertEqual(snapshot['components']['rm-worker']['state'],'OFFLINE_CONFIRMED')
            finally:
                # No success is requested in this isolated partial-lifecycle test.
                for client in [rt._host,launcher]:
                    if client:
                        try: client.close_transport()
                        except Exception: pass
                proc.terminate();proc.join(5)
                rt._host=rt._worker=rt._adapter=rt._worker_permit=None


if __name__=='__main__': unittest.main()
