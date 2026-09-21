import asyncio
import json
import os
import multiprocessing
import threading
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from scripts.ft.ft1_fault_hooks import install
from scripts.ft.descendants import snapshot

root=Path(os.environ['FT_CONTROL_SOCKET']).parent
race_request=multiprocessing.Event()
race_done=multiprocessing.Event()

def emit(event,**fields):
    if sys.argv[1]=='race' and event=='score_execution_started' and fields.get('sample_idx')==5:
        race_request.set()
        assert race_done.wait(5)
    with (root/f'observer-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps({'event':event,'pid':os.getpid(),**fields})+'\n')

observer=SimpleNamespace(observe=emit)

def score(prompt,completion,prompt_ids,completion_ids,*,sample_idx,source_row_id,**kwargs):
    observer.observe('score_execution_started',sample_idx=sample_idx,source_row_id=source_row_id,
                     task_id=0,identity=snapshot(os.getpid()))
    from areal.reward.gsm8k import gsm8k_reward_fn
    return gsm8k_reward_fn(prompt,completion,prompt_ids,completion_ids,**kwargs)

score.strict_scoring=True

async def scoring(mode):
    os.environ['HOME']='/tmp'
    from areal.api import AsyncRewardWrapper
    install('F4',observer)
    for idx in [i for i in range(8) if not (mode in ('late','race','cross_group') and i==6)]:
        observer.observe('generation_complete',source_row_id=5518,sample_idx=idx,task_id=0)
    if mode=='cross_group':observer.observe('generation_complete',source_row_id=5518,sample_idx=6,task_id=99)
    if mode=='race':
        def late_generation():
            assert race_request.wait(20)
            observer.observe('generation_complete',source_row_id=5518,sample_idx=6,task_id=0)
            race_done.set()
        thread=threading.Thread(target=late_generation);thread.start()
    wrapper=AsyncRewardWrapper(score)
    slots=(7,4,0,0) if mode=='duplicate' else (7,4,0,5)
    results=[]
    for idx in slots:
        results.append(await wrapper('', '4',[],[],answer='4',source_row_id=5518,sample_idx=idx))
    if mode=='race':thread.join(timeout=5);assert not thread.is_alive()
    assert results==[1,1,1,1]
    (root/'fixture-result.json').write_text(json.dumps({'mode':mode,'scores':results}))


def trainer():
    class Actor:
        ordinal=0
        def optimizer_step(self):
            self.ordinal+=1
            return {'update_successful':0 if self.ordinal==1 else 1}
    sys.modules['areal.engine.megatron_engine']=SimpleNamespace(MegatronPPOActor=Actor)
    sys.modules['scripts.ft.areal_pilot_hooks']=SimpleNamespace(writer=lambda:SimpleNamespace(flush=lambda:emit('flushed')))
    client=install('F2',observer)
    try:
        actor=Actor()
        for i in range(3):actor.optimizer_step()
    finally:client.close()


if sys.argv[1]=='trainer':trainer()
elif sys.argv[1]=='F2':
    codes=[subprocess.run([sys.executable,__file__,'trainer'],timeout=30).returncode for _ in range(2)]
    assert codes==[-9,0],codes
    (root/'fixture-result.json').write_text(json.dumps({'codes':codes}))
else:asyncio.run(scoring(sys.argv[1]))
