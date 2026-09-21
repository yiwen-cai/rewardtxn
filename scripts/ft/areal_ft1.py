"""FT1 ten-step pilot with shared observation and one-shot fault cuts."""
import dataclasses
import functools
import json
import os
from pathlib import Path
import sys
import time


TRAINER_IDENTITY = None

def observe(event, **fields):
    root=Path(os.environ['FT1_OBSERVE']);root.mkdir(exist_ok=True)
    record={'event':event,'pid':os.getpid(),'trainer_identity':TRAINER_IDENTITY,'monotonic_ns':time.monotonic_ns(),**fields}
    with (root/f'{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n');stream.flush()


def install_observer():
    from areal import workflow_context
    from areal.workflow.rlvr import RLVRWorkflow
    from areal.utils import strict_reward
    from scripts.ft.descendants import snapshot
    original=RLVRWorkflow._compute_rewards
    @functools.wraps(original)
    async def rewards(self,resp,prompt_str,task_data):
        ctx=workflow_context.get()
        observe('generation_complete',source_row_id=task_data['source_row_id'],
            task_id=ctx.task_id,sample_idx=ctx.sample_idx,input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,output_versions=resp.output_versions)
        return await original(self,resp,prompt_str,task_data)
    RLVRWorkflow._compute_rewards=rewards
    worker=strict_reward._worker
    def scoring(channel,scorer,args,kwargs):
        ctx=workflow_context.get()
        observe('score_execution_started',source_row_id=kwargs['source_row_id'],
            task_id=ctx.task_id,sample_idx=ctx.sample_idx,identity=snapshot(os.getpid()))
        return worker(channel,scorer,args,kwargs)
    strict_reward._worker=scoring



def install_load_observer():
    from areal.engine.megatron_engine import MegatronPPOActor
    from areal.utils.recover import RecoverInfo
    from scripts.ft.training_adapter import native_snapshot
    save=MegatronPPOActor.save
    def engine_save(self,meta):
        observe('checkpoint_save_state',path=meta.path,update_id=getattr(self,'_pilot_update',None),state=native_snapshot(self))
        return save(self,meta)
    MegatronPPOActor.save=engine_save
    load=MegatronPPOActor.load
    def engine_load(self,meta):
        result=load(self,meta)
        observe('checkpoint_loaded_state',path=meta.path,state=native_snapshot(self))
        return result
    MegatronPPOActor.load=engine_load
    recover_load=RecoverInfo.load.__func__
    @classmethod
    def metadata_load(cls,path):
        result=recover_load(cls,path)
        observe('recover_info_loaded',path=str(path),last_step_info=dataclasses.asdict(result.last_step_info))
        return result
    RecoverInfo.load=metadata_load


def main(args):
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig,load_expr_config
    from areal.utils.seeding import set_random_seed,get_seed
    from scripts.ft.areal_pilot import load_pilot_dataset
    from scripts.ft.areal_pilot_hooks import install_hooks,close_events,file_manifest,ObservedRLVRWorkflow
    from scripts.ft.training_adapter import Runtime,install,native_snapshot
    from scripts.ft.reward_return import verifier_fingerprint
    config,_=load_expr_config(args,GRPOConfig)
    global TRAINER_IDENTITY
    from scripts.ft.descendants import snapshot
    TRAINER_IDENTITY=snapshot(os.getpid())
    arm=os.environ['FT1_ARM'];root=Path(config.cluster.fileroot).parent
    scenario=os.environ.get('FT1_SCENARIO','no_fault')
    if scenario not in ('no_fault','F1','F2','F4'):raise RuntimeError('unmapped FT1 scenario')
    if arm not in ('A','R') or config.total_train_steps!=10 or config.recover.retries!=(0 if scenario=='no_fault' else 1):
        raise RuntimeError('unsupported FT1 smoke configuration')
    if not config.actor.megatron.async_save or config.gconfig.n_samples!=8 or config.train_dataset.batch_size!=4:
        raise RuntimeError('FT1 requires common async DCP K8/U4')
    os.environ['AREAL_PILOT_EVENTS']=str(root/'observer-pilot')
    os.environ['FT1_OBSERVE']=str(root/'observer-ft1')
    set_random_seed(config.seed,'ft1-trainer')
    runtime=None
    if arm=='R':
        runtime=Runtime(config,root/'rewardtxn');install(runtime)
    install_hooks()
    install_observer()
    client=None
    if scenario!='no_fault':
        from scripts.ft.ft1_fault_hooks import install as install_fault
        client=install_fault(scenario,sys.modules[__name__])
        install_load_observer()
    observe('configured',arm=arm,scenario=scenario,config=dataclasses.asdict(config),effective_seed=get_seed(),
            verifier_sha256=verifier_fingerprint())
    try:
        dataset=load_pilot_dataset(config.train_dataset.path)
        with PPOTrainer(config,train_dataset=dataset,valid_dataset=None) as trainer:
            if arm=='R':trainer.train(workflow=runtime.bridge)
            else:
                from areal.reward.gsm8k import gsm8k_reward_fn
                trainer.train(workflow=ObservedRLVRWorkflow,
                    workflow_kwargs={'reward_fn':gsm8k_reward_fn,'gconfig':config.gconfig,
                                     'tokenizer':config.tokenizer_path})
            # Training is finished. Native destroy also drains; this explicit
            # end-only drain permits observation before model destruction.
            trainer.actor.checkpointer.wait_async_saves()
            signature=native_snapshot(trainer.actor)
            (root/'final-native-state.json').write_text(json.dumps(signature,sort_keys=True))
            from areal.utils.saver import Saver
            from areal.utils.recover import RecoverHandler
            if arm=='A':
                cfg=config.recover;params=(cfg.experiment_name,cfg.trial_name,cfg.fileroot)
                checkpoint=Saver.get_recover_checkpoint_path(*params)
                metadata=RecoverHandler.recover_info_path(*params)
                observe('final_checkpoint',checkpoint=checkpoint,metadata=metadata,
                        files=file_manifest(checkpoint),recover_files=file_manifest(metadata))
            else:
                observe('retained_head',head=json.loads((runtime.owner.root/'control.json').read_text())['head'])
        observe('training_returned',arm=arm)
    finally:
        if runtime is not None:runtime.close()
        close_events()
        if client is not None:client.close()


if __name__=='__main__':main(sys.argv[1:])
