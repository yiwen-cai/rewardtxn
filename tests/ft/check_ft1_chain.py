"""Offline no-fault checkpoint/consumption audit, independent of method state code."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
from check_ft1_input_audit import batch_rows,fingerprint,rows


def read(path):return json.loads(path.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def canon(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def verify(root):
    started=time.monotonic();root=Path(root)
    case=read(root/'ft1-case.json');arm=case['arm']
    no_fault=case['scenario']=='no_fault'
    repo=Path(__file__).resolve().parents[2]
    for name,digest in read(root/'source-sha256.json').items():
        archived=root/'source-archive'/name
        assert sha(archived if archived.exists() else repo/name)==digest,name
    events=[e for p in (root/'observer-ft1').glob('*.jsonl') for e in rows(p)]
    pilot=sorted([e for p in (root/'observer-pilot').glob('*.jsonl') for e in rows(p)],key=lambda e:e['monotonic_ns'])
    batches=[e for e in pilot if e['event']=='batch_taken'];trained=[e for e in pilot if e['event']=='train_batch']
    applied=[e for e in pilot if e['event']=='optimizer_end']
    assert len(batches)==len(trained)==len(applied)
    if no_fault:assert len(applied)==10
    for b,t,a in zip(batches,trained,applied):
        assert b['monotonic_ns']<t['monotonic_ns']<a['monotonic_ns'] and t['update_id']==a['update_id']
        assert Counter(k for k,r in batch_rows(b['batches']))==Counter(k for k,r in batch_rows(t['batches'],training=True))
    size=0;files_checked=0
    if arm=='A':
        record=next(e for e in events if e['event']=='final_checkpoint')
        for key,items in [('checkpoint','files'),('metadata','recover_files')]:
            base=root/Path(record[key]).relative_to('/output')
            actual={str(p.relative_to(base)):p for p in base.rglob('*') if p.is_file()}
            assert set(actual)=={r['path'] for r in record[items]}
            for info in record[items]:
                p=actual[info['path']]
                assert not p.is_symlink() and p.stat().st_size==info['size'] and sha(p)==info['sha256']
                size+=info['size'];files_checked+=1
        assert read(root/Path(record['metadata']).relative_to('/output')/'step_info.json')['global_step']==9
        retained=[]
        incarnations=list(dict.fromkeys(e['pid'] for e in applied))
        for pid in incarnations:
            local=[a for a in applied if a['pid']==pid]
            loads=[e for e in events if e['event']=='recover_info_loaded' and e['pid']==pid]
            if loads:
                assert len(loads)==1
                base_step=loads[0]['last_step_info']['global_step']+1
                state_load=[e for e in events if e['event']=='checkpoint_loaded_state' and e['pid']==pid]
                assert len(state_load)==1 and base_step<=len(retained)
                saved=[e for e in events if e['event']=='checkpoint_save_state' and e.get('update_id')==retained[base_step-1]['update_id']]
                assert len(saved)==1 and saved[0]['state']==state_load[0]['state'],'checkpoint state/step mismatch'
                retained=retained[:base_step]
            else:retained=[]
            retained.extend(local)
        assert len(retained)==10
        actual_by_update={t['update_id']:b for b,t in zip(batches,trained)}
        retained_slots=[]
        for update in retained:
            for batch in actual_by_update[update['update_id']]['batches']:
                sources=batch['pilot_source_row_id'];slots=batch['pilot_sample_idx']
                assert len(sources)==len(slots)==batch['batch_size']
                retained_slots.extend(zip(sources,slots))
        assert len(retained_slots)==len(set(retained_slots))==320
        source_rows={row for row,idx in retained_slots}
        assert len(source_rows)==40
        extra={'final_checkpoint_retains_step':9,'retained_updates':10,
               'retained_physical_update_ids':[e['update_id'] for e in retained],
               'discarded_physical_updates':len(applied)-len(retained),'retained_source_rows':sorted(source_rows)}
    else:
        method=root/'rewardtxn';head=read(method/'state/control.json')['head'];chain=[]
        while head is not None:
            directory=method/'state/generations'/head['generation']
            token=read(directory/'token.json');manifest=read(directory/'manifest.json');intent=read(directory/'intent.json')
            assert sha(directory/'token.json')==head['token_sha256']
            assert sha(directory/'manifest.json')==token['manifest_sha256'] and sha(directory/'intent.json')==token['intent_sha256']
            assert token['parent']==manifest['parent']==intent['parent']
            assert manifest['updates']==intent['updates'] and len(manifest['updates'])==1
            checkpoint=directory/'checkpoint'
            assert not any(p.is_symlink() for p in checkpoint.rglob('*'))
            files={str(p.relative_to(checkpoint)):p for p in checkpoint.rglob('*') if p.is_file()}
            assert set(files)==set(manifest['files'])
            assert {'native/.metadata','native-state.json','policy.json'}<=set(files)
            for name,info in manifest['files'].items():
                assert files[name].stat().st_size==info['size'] and sha(files[name])==info['sha256'],name
                size+=info['size'];files_checked+=1
            chain.append((directory,manifest));head=token['parent']
        chain.reverse();assert len(chain)==10
        log=rows(method/'events.jsonl');commits=[e for e in log if e['event']=='committed']
        assert [e['generation'] for e in commits]==[d.name for d,m in chain]
        retained=[]
        for directory,manifest in chain:
            method_update=[e for e in log if e.get('generation')==directory.name and e['event']=='optimizer_applied']
            assert len(method_update)==1
            opt=method_update[0]
            matched=[(b,t,a) for b,t,a in zip(batches,trained,applied) if a['pid']==opt['pid'] and t['monotonic_ns']<opt['monotonic_ns']<a['monotonic_ns']]
            assert len(matched)==1
            retained.append((matched[0][0],matched[0][2]))
        consumed=set();source_rows=set()
        original=rows(repo/'runs/diagnosis-20260910/train.jsonl')
        def blob(digest):
            path=method/'artifacts/blobs'/digest
            assert sha(path)==digest
            return read(path)
        for step,((directory,manifest),batch,update,commit) in enumerate(zip(chain,[x[0] for x in retained],[x[1] for x in retained],commits)):
            assert read(directory/'checkpoint/policy.json')['step_info']['global_step']==step==commit['global_step']
            assert update['monotonic_ns']<commit['monotonic_ns']
            expected=[];samples=[]
            for group in manifest['updates'][0]['groups']:
                assert group['k']==8 and len(group['samples'])==8
                assert {s['sample_index'] for s in group['samples']}==set(range(8))
                prompt=group['prompt'];row=prompt['source_row_id'];source_rows.add(row)
                assert prompt['messages']==original[row]['prompt'] and prompt['answer']==original[row]['label']
                assert hashlib.sha256(canon(prompt)).hexdigest()==group['prompt_sha256']
                for sample in group['samples']:
                    name=sample['sample'];assert name==group['logical_group_id']+':'+str(sample['sample_index'])
                    samples.append(name);receipt=sample['receipt'];assert receipt['attempt']['sample']==name
                    response=blob(receipt['payload']['response_sha256']);reward=blob(receipt['payload']['reward_sha256'])
                    assert response['binding']['attempt']==reward['binding']['attempt']==receipt['attempt']
                    assert reward['payload']['response_sha256']==receipt['payload']['response_sha256']
                    envelope=reward['payload']['return'];assert envelope['status']=='scored' and envelope['schema']==2
                    payload=response['payload'];n=len(payload['input_tokens'])
                    key=fingerprint(payload['input_tokens']+payload['output_tokens'],[-1]*n+payload['output_versions'],[0]*n+[1]*len(payload['output_tokens']))
                    expected.append((key,envelope['score']))
            assert len(samples)==32 and len(set(samples))==32 and not consumed.intersection(samples)
            assert Counter(expected)==Counter(batch_rows(batch['batches']))
            consumed.update(samples)
            data=manifest['data'];assert set(data['consumed'])==consumed and len(data['consumed'])==len(consumed)
            assert consumed<=set(data['drawn']) and {p['sample'] for p in data['pending']}==set(data['drawn'])-consumed
        assert len(consumed)==320 and len(source_rows)==40
        extra={'retained_generations':10,'unique_consumed_samples':320,'unique_source_rows':40,'retained_source_rows':sorted(source_rows)}
    return {'verified':True,'arm':arm,'scope':'observed optimizer/input to checkpoint linkage; reward re-score and native reload reported separately',
            'checkpoint_bytes_hashed':size,'files_hashed':files_checked,'wall_seconds':time.monotonic()-started,**extra}


if __name__=='__main__':
    result=verify(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(result,indent=2));print(json.dumps(result))
