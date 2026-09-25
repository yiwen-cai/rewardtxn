"""Independent FT1 fault-hit and continuation audit; authority audited separately."""
import json
from pathlib import Path
import re
import sys


def gpu_id(value):
    value=str(value)
    return value[4:] if value.startswith('GPU-') else value


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line]
def one(events,key,value):
    selected=[e for e in events if e.get(key)==value]
    assert len(selected)==1,(key,value,len(selected))
    return selected[0]


def verify(root):
    root=Path(root);case=read(root/'ft1-case.json');scenario=case['scenario']
    journal=rows(root/'events.jsonl');signals=[e for e in journal if e['kind']=='signal_sent']
    result={'scenario':scenario,'arm':case['arm'],'formal_sample':False,
            'scope':'kernel fault target, semantic cut and native continuation; not reward authority or full retained-chain oracle'}
    cleanup=read(root/'cleanup.json');assert cleanup['returncode']==1 and 'No such object' in cleanup['stderr']
    assert read(root/'inspect-final.json')[0]['Id']==cleanup['id']
    assert read(root/'network-cleanup.json')['returncode']==1
    assert len([e for e in journal if e['kind']=='launcher_created'])==1
    if len(signals)!=1:
        return {**result,'valid_hit':False,'classification':'technical_invalid','reason':'no unique planned signal',
                'controller_result':one(journal,'kind','result')}
    sent=signals[0];ready=one(journal,'kind','ready');exited=one(journal,'kind','process_exit_observed')
    assert sent['signal']==9 and sent['identity']==ready['identity']==exited['identity']
    assert sent['incarnation']==ready['incarnation']==exited['incarnation']
    assert ready['controller_monotonic_ns']<=sent['controller_monotonic_ns']<=exited['controller_monotonic_ns']
    expected=read(root/'controller-config.json')['schedule'][0];assert ready['evidence']==expected['evidence']
    events=[e for p in (root/'observer-ft1').glob('*.jsonl') for e in rows(p)]
    configs=[e for e in events if e['event']=='configured']
    assert configs and all(e['config']==configs[0]['config'] and e['effective_seed']==configs[0]['effective_seed'] and e['verifier_sha256']==configs[0]['verifier_sha256'] for e in configs)
    pilot=sorted([e for p in (root/'observer-pilot').glob('*.jsonl') for e in rows(p)],key=lambda e:e['monotonic_ns'])
    if scenario=='F4':
        witness=one(events,'event','fault_ready');assert witness['identity']==sent['identity']
        assert witness['generated_mask']==255 and witness['started_mask'].bit_count()==4 and witness['ordinal']==4
        group=[e for e in events if e.get('source_row_id')==5518 and e.get('task_id')==witness['target_task_id']
               and e.get('trainer_identity')==witness['trainer_identity'] and e['monotonic_ns']<=witness['cut_monotonic_ns']]
        generations=[e for e in group if e['event']=='generation_complete']
        starts=sorted([e for e in group if e['event']=='score_execution_started'],key=lambda e:e['monotonic_ns'])
        assert {e['sample_idx'] for e in generations}==set(range(8))
        assert len(starts)==4 and len({e['sample_idx'] for e in starts})==4
        assert starts[-1]['identity']==sent['identity'] and starts[-1]['monotonic_ns']==witness['cut_monotonic_ns']
        assert max(e['monotonic_ns'] for e in generations)<witness['cut_monotonic_ns']<sent['controller_monotonic_ns']
        logs=[]
        for path in (root/'areal').rglob('trainer.log'):
            for line in path.read_text(errors='replace').splitlines():
                if 'scoring_attempt ' in line:
                    logs.append(json.JSONDecoder().raw_decode(line.split('scoring_attempt ',1)[1])[0])
        killed=[e for e in logs if e['pid']==sent['identity']['pid'] and e['exit_code']==-9]
        assert len(killed)==1 and killed[0]['status']=='error'
        retries=[e for e in logs if e['input_sha256']==killed[0]['input_sha256'] and e['attempt']==2 and e['status']=='scored']
        result['same_input_native_score_retry_succeeded']=len(retries)==1
    elif scenario=='F2':
        witness=one(events,'event','fault_ready');assert witness['identity']==sent['identity']
        updates=[e for e in pilot if e['event']=='optimizer_end' and e['pid']==sent['identity']['pid']
                 and e['monotonic_ns']<sent['controller_monotonic_ns'] and e['stats']['update_successful']==1]
        assert len(updates)==case.get('f2_ordinal',2) and updates[-1]['monotonic_ns']<witness['monotonic_ns']
        assert not any(e['event']=='checkpoint_save_start' and e['update_id']==updates[-1]['update_id'] for e in pilot)
        result['killed_uncheckpointed_update_id']=updates[-1]['update_id']
    elif scenario=='F1':
        witness=read(root/'f1-worker-witness.json');assert witness['identity']==sent['identity']
        assert witness['request_ids'] and len(witness['request_ids'])==len(witness['output_lengths'])
        active,complete=witness['active'],witness['completed']
        assert active['trainer']==complete['trainer'] and active['task_id']==complete['task_id']==0
        assert active['source_row_id']==complete['source_row_id']==5518
        assert not active['ambiguous'] and not complete['ambiguous']
        assert witness['scheduler_attached_ns']<=active['monotonic_ns']<=complete['monotonic_ns']<witness['cut_monotonic_ns']<sent['controller_monotonic_ns']
        returned=[e for e in events if e['event']=='generation_complete' and e['trainer_identity']==complete['trainer']
                  and e['source_row_id']==5518 and e['task_id']==0 and e['sample_idx']==complete['sample_idx']]
        assert returned and min(e['monotonic_ns'] for e in returned)<witness['cut_monotonic_ns']
        result['target_gpu_uuid']=witness['gpu_uuid']
        assert gpu_id(witness['gpu_uuid']) in {gpu_id(gpu) for gpu in read(root/'gpu-uuids.json')}
    else:raise AssertionError('unmapped scenario')
    jobs=[read(p) for p in (root/'areal').rglob('job-lifecycle/*.json')]
    trained=[e for e in pilot if e['event']=='optimizer_end' and e['stats']['update_successful']==1]
    after=[e for e in trained if e['monotonic_ns']>sent['controller_monotonic_ns']]
    returned=[e for e in events if e['event']=='training_returned']
    timeout=any(e['kind']=='method_observation' and e.get('outcome')=='timeout' for e in journal)
    run_ids=re.findall(r'run_id=(\d+), is_recover_run=(?:True|False)',(root/'launcher.log').read_text(errors='replace'))
    return {**result,'valid_hit':True,'classification':'timeout' if timeout else ('training_returned' if returned else 'safe_stop_or_error'),
            'signals':1,'physical_optimizer_successes':len(trained),'successful_updates_after_fault':len(after),
            'native_run_ids':run_ids,'job_receipts':len(jobs),'all_jobs_cleaned':bool(jobs) and all(j['cleanup']['empty'] for j in jobs),
            'training_returned':bool(returned),'affected_work_recovery':'pending independent retained-input/checkpoint audit'}


if __name__=='__main__':
    report=verify(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(report,indent=2));print(json.dumps(report))
