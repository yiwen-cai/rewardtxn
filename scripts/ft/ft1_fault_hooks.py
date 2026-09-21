"""Shared one-shot FT1 semantic cuts; native launcher/scorer own retries."""
import multiprocessing
import os
import time


def contract(scenario):
    if scenario=='F1':
        return {'event_id':'ft1-f1-generator','target':'generator','waiters':['generator'],
                'evidence':{'phase':'generator_active_after_one_response','source_row_id':5518,'k':8}}
    if scenario=='F4':
        return {'event_id':'ft1-f4-score-worker','target':'reward','waiters':['reward'],
                'evidence':{'phase':'fourth_score_worker_entry','source_row_id':5518,'k':8,'ordinal':4}}
    if scenario=='F2':
        return {'event_id':'ft1-f2-trainer','target':'trainer','waiters':['trainer'],
                'evidence':{'phase':'post_optimizer_pre_save','successful_ordinal':2}}
    raise ValueError('unmapped FT1 fault')


def install(scenario, observer_module):
    from scripts.ft.descendants import Client,snapshot
    frozen=contract(scenario)
    emit=observer_module.observe
    client=None
    if scenario=='F1':
        from scripts.ft.ft1_f1_trainer import install as install_f1
        install_f1(observer_module)
    elif scenario=='F4':
        # Shared only between this trainer and its forked score invocations.
        lock=multiprocessing.Lock()
        generated=multiprocessing.Value('i',0,lock=False)
        executions=multiprocessing.Value('i',0,lock=False)
        started=multiprocessing.Value('i',0,lock=False)
        target_task=multiprocessing.Value('q',-1,lock=False)
        def observe(event,**fields):
            if fields.get('source_row_id')!=5518 or event not in ('generation_complete','score_execution_started'):
                return emit(event,**fields)
            idx=fields['sample_idx'];task=fields['task_id']
            assert type(idx) is int and 0<=idx<8 and type(task) is int and task>=0
            if event=='generation_complete':
                emit(event,**fields)
                with lock:
                    if target_task.value<0:target_task.value=task
                    if target_task.value==task:generated.value |= 1<<idx
                return
            with lock:
                bound=target_task.value==task
                if bound:
                    executions.value+=1
                    repeated=bool(started.value & (1<<idx))
                    started.value |= 1<<idx
                    ordinal=executions.value
                    ready=ordinal==4 and generated.value==255 and started.value.bit_count()==4 and not repeated
                    witness={'generated_mask':generated.value,'started_mask':started.value,'ordinal':ordinal,
                             'target_task_id':target_task.value,'cut_monotonic_ns':time.monotonic_ns()}
            if not bound:return emit(event,**fields)
            # Condition is sampled before writing the score-start observation;
            # later generation completions cannot make this cut eligible.
            emit(event,**fields,monotonic_ns=witness['cut_monotonic_ns'])
            if ordinal!=4:return
            if not ready:
                emit('fault_cut_missed',scenario=scenario,**witness)
                return
            with_client=Client('reward',event_id=frozen['event_id'])
            try:
                if with_client.injection['status']=='already_fired':
                    emit('fault_injection_already_fired',scenario=scenario)
                    return
                emit('fault_ready',scenario=scenario,identity=snapshot(os.getpid()),
                     incarnation=with_client.incarnation,evidence=frozen['evidence'],**witness)
                with_client.ready(frozen['event_id'],frozen['evidence'])
                with_client.wait_release(frozen['event_id'])
                raise RuntimeError('SIGKILL target survived')
            finally:
                with_client.close()
        observer_module.observe=observe
    else:
        from areal.engine.megatron_engine import MegatronPPOActor
        from scripts.ft.areal_pilot_hooks import writer
        client=Client('trainer',event_id=frozen['event_id'])
        emit('fault_target_registered',scenario=scenario,identity=snapshot(os.getpid()),
             incarnation=client.incarnation,assignment=client.injection)
        original=MegatronPPOActor.optimizer_step
        ordinal=0
        def optimizer_step(self):
            nonlocal ordinal
            result=original(self)
            if result.get('update_successful')==1:ordinal+=1
            if ordinal==2 and client.injection['status']=='pending':
                writer().flush()
                emit('fault_ready',scenario=scenario,identity=snapshot(os.getpid()),
                     incarnation=client.incarnation,evidence=frozen['evidence'])
                client.ready(frozen['event_id'],frozen['evidence'])
                client.wait_release(frozen['event_id'])
                raise RuntimeError('SIGKILL target survived')
            return result
        MegatronPPOActor.optimizer_step=optimizer_step
    return client
