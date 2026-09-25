"""Shared one-shot FT1 semantic cuts; native launcher/scorer own retries."""
import multiprocessing
import os
import time


def f2_ordinal():
    # Default 2 is the frozen FT1 pilot cut; 30-step gate pilots set 12 (FT-v1 section 6).
    value=int(os.environ.get('FT1_F2_ORDINAL','2'))
    if value not in (2,12):raise ValueError('unfrozen F2 ordinal')
    return value


def contract(scenario):
    if scenario=='F1':
        return {'event_id':'ft1-f1-generator','target':'generator','waiters':['generator'],
                'evidence':{'phase':'generator_active_after_one_response','source_row_id':5518,'k':8}}
    if scenario=='F4':
        return {'event_id':'ft1-f4-score-worker','target':'reward','waiters':['reward'],
                'evidence':{'phase':'fourth_score_worker_entry','source_row_id':5518,'k':8,'ordinal':4}}
    if scenario=='F2':
        return {'event_id':'ft1-f2-trainer','target':'trainer','waiters':['trainer'],
                'evidence':{'phase':'post_optimizer_pre_save','successful_ordinal':f2_ordinal()}}
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
        from pathlib import Path as _Path
        from areal.engine.megatron_engine import MegatronPPOActor
        from scripts.ft.areal_pilot_hooks import writer
        client=Client('trainer',event_id=frozen['event_id'])
        emit('fault_target_registered',scenario=scenario,identity=snapshot(os.getpid()),
             incarnation=client.incarnation,assignment=client.injection)
        original=MegatronPPOActor.optimizer_step
        original_save=MegatronPPOActor.save
        ordinal=0
        predecessor={'path':None}

        def save(self, meta):
            result=original_save(self, meta)
            # Record recover checkpoint path; completeness checked after wait_async_saves.
            predecessor['path']=meta.path
            return result

        def _require_complete_dcp(path):
            root=_Path(path)
            if path is None or not root.is_dir():
                raise RuntimeError('F2 cut lacks prior recover checkpoint directory: %s' % (path,))
            names={p.name for p in root.iterdir()}
            if '.metadata' not in names:
                raise RuntimeError('F2 cut lacks DCP .metadata after wait_async_saves: %s' % (root,))
            if not any(name.endswith('.distcp') and (root/name).stat().st_size>0 for name in names):
                raise RuntimeError('F2 cut lacks non-empty .distcp after wait_async_saves: %s' % (root,))
            return sorted(names)

        def optimizer_step(self):
            nonlocal ordinal
            result=original(self)
            if result.get('update_successful')==1:ordinal+=1
            if ordinal==frozen['evidence']['successful_ordinal'] and client.injection['status']=='pending':
                writer().flush()
                # FT1 keeps async_save=true; drain step-0 DCP before the cut so
                # native recover does not load a partial recover_checkpoint.
                self.checkpointer.wait_async_saves()
                files=_require_complete_dcp(predecessor['path'])
                emit('fault_ready',scenario=scenario,identity=snapshot(os.getpid()),
                     incarnation=client.incarnation,evidence=frozen['evidence'],
                     predecessor_checkpoint=predecessor['path'],predecessor_files=files)
                client.ready(frozen['event_id'],frozen['evidence'])
                client.wait_release(frozen['event_id'])
                raise RuntimeError('SIGKILL target survived')
            return result
        MegatronPPOActor.save=save
        MegatronPPOActor.optimizer_step=optimizer_step
    return client
