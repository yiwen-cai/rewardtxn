"""Combine separately verified fault, input, retained-chain and load evidence."""
import json
from pathlib import Path
import sys
from check_ft1_input_audit import batch_rows,fingerprint,rows


def read(path):return json.loads(path.read_text())
def finalize(root,input_dir,load_dir):
    fault=read(root/'fault-verification.json')
    if not fault['valid_hit']:return fault
    inputs=read(input_dir/'input-verification.json');chain=read(root/'chain-verification.json')
    loaded=read(load_dir/'load-verification.json')
    assert inputs['verified'] and chain['verified'] and loaded['engine_exact_match']
    assert chain['arm']==inputs['arm']==loaded['arm']==fault['arm']
    pilot=sorted([e for p in (root/'observer-pilot').glob('*.jsonl') for e in rows(p)],key=lambda e:e['monotonic_ns'])
    observed=[e for p in (root/'observer-ft1').glob('*.jsonl') for e in rows(p)]
    if fault['scenario']=='F2':
        trains=[e for e in pilot if e['event']=='train_batch'];batches=[e for e in pilot if e['event']=='batch_taken']
        batch=next(b for b,t in zip(batches,trains) if t['update_id']==fault['killed_uncheckpointed_update_id'])
        target=set()
        generated={}
        for e in observed:
            if e['event']=='generation_complete':
                n=len(e['input_tokens']);key=fingerprint(e['input_tokens']+e['output_tokens'],[-1]*n+e['output_versions'],[0]*n+[1]*len(e['output_tokens']))
                generated.setdefault(key,set()).add(e['source_row_id'])
        for key,reward in batch_rows(batch['batches']):
            assert key in generated and len(generated[key])==1
            target.update(generated[key])
        assert len(target)==4
    else:target={5518}
    retained=set(chain['retained_source_rows']);recovered=target<=retained
    sent=next(e for e in rows(root/'events.jsonl') if e['kind']=='signal_sent')['controller_monotonic_ns']
    endpoint=None;exact=False
    if recovered:
        if fault['arm']=='R':
            accumulated=set()
            for commit in (e for e in rows(root/'rewardtxn/events.jsonl') if e['event']=='committed'):
                manifest=read(root/'rewardtxn/state/generations'/commit['generation']/'manifest.json')
                accumulated.update(g['prompt']['source_row_id'] for u in manifest['updates'] for g in u['groups'])
                if target<=accumulated:
                    endpoint=commit['monotonic_ns'];exact=True;break
        else:
            endpoint=next(e['monotonic_ns'] for e in observed if e['event']=='final_checkpoint')
        assert endpoint is not None and endpoint>sent
    seconds=(endpoint-sent)/1e9 if endpoint is not None else None
    return {**fault,'classification':'correct_recovered' if recovered else 'safe_discard',
            'safety_verified_for_retained_chain':True,'training_continuation_verified':fault['successful_updates_after_fault']>0,
            'affected_work_recovery':'verified' if recovered else 'target not retained','affected_work_recovered':recovered,'target_source_rows':sorted(target),
            'retained_target_source_rows':sorted(target&retained),'full_native_reload_verified':True,
            'fault_to_verified_persistence_seconds':seconds,
            'persistence_time_kind':'commit observation' if exact else 'final-checkpoint upper bound' if recovered else None,
            'recovery_within_900_seconds':True if recovered and seconds<=900 else False if exact or not recovered else None,
            'formal_rto_eligible':False,
            'scope':'FT1 single-rank pilot; A first durable completion is not independently timestamped, no formal RTO comparison'}


if __name__=='__main__':
    root,inputs,loaded=map(Path,sys.argv[1:4]);report=finalize(root,inputs,loaded)
    (root/'functional-verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
