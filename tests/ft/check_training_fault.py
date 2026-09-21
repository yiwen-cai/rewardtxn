"""Independent single-fault acceptance from kernel, lifecycle and retained files."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_bytes())


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def verify(root):
    assert (root/'exitcode').read_text().strip()=='0'
    final=read(root/'inspect-final.json')[0]; cleanup=read(root/'cleanup.json')
    assert final['State']['ExitCode']==0 and final['Id']==cleanup['id']
    assert cleanup['returncode']==1 and 'No such object' in cleanup['stderr']
    network=read(root/'network-cleanup.json')
    assert network['returncode']==1 and 'not found' in network['stderr']
    journal=rows(root/'events.jsonl'); events=rows(root/'rewardtxn/events.jsonl')
    def one(seq, key, value):
        selected=[e for e in seq if e[key]==value]
        assert len(selected)==1,(key,value,len(selected))
        return selected[0]
    assert one(journal,'kind','result')['classification']=='execution_complete'
    assert one(journal,'kind','method_observation')['launcher_exit_code']==1
    assert len([e for e in journal if e['kind']=='launcher_created'])==1
    registrations=[e for e in journal if e['kind']=='descendant_registered']
    trainers=[e for e in events if e['event']=='trainer_registered']
    assert len(registrations)==len(trainers)==2
    for reg,trainer in zip(registrations,trainers):
        assert reg['role']=='trainer' and reg['identity']==trainer['identity']
        assert reg['incarnation']==trainer['incarnation'] and trainer['pid']==reg['identity']['pid']
    first,second=trainers
    assert first['pid']!=second['pid']
    assert [e['assignment']['status'] for e in trainers]==['pending','already_fired']
    assert first['assignment']['event_nonce']==second['assignment']['event_nonce']
    sent=one(journal,'kind','signal_sent'); exited=one(journal,'kind','process_exit_observed')
    ready=one(journal,'kind','ready'); witness=one(events,'event','fault_ready')
    assert sent['signal']==9
    for e in [sent,exited,ready]:
        assert e['identity']==first['identity'] and e['incarnation']==first['incarnation']
    expected=read(root/'controller-config.json')['schedule'][0]['evidence']
    assert expected in [{'successful_ordinal':2,'saved_global_step':0,'phase':p}
                        for p in ('post_optimizer_pre_save','checkpoint_midwrite')]
    midwrite=expected['phase']=='checkpoint_midwrite'
    assert ready['evidence']==witness['evidence']==expected
    assert witness['pid']==first['pid']
    commits=[e for e in events if e['event']=='committed']
    assert [e['global_step'] for e in commits]==[0,1,2]
    assert [e['pid'] for e in commits]==[first['pid'],second['pid'],second['pid']]
    optimizers=[e for e in events if e['event']=='optimizer_applied']
    assert len(optimizers)==4 and all(e['stats']['update_successful']==1.0 for e in optimizers)
    assert [e['pid'] for e in optimizers]==[first['pid'],first['pid'],second['pid'],second['pid']]
    abandoned=witness['generation']
    assert optimizers[1]['generation']==abandoned
    assert commits[0]['generation']==witness['retained']['generation']
    assert commits[0]['monotonic_ns']<optimizers[1]['monotonic_ns']<sent['controller_monotonic_ns']<=exited['controller_monotonic_ns']
    loaded=one(events,'event','native_state_loaded')
    assert loaded['exact_match'] and loaded['pid']==second['pid'] and loaded['generation']==commits[0]['generation']
    assert exited['controller_monotonic_ns']<loaded['monotonic_ns']<optimizers[2]['monotonic_ns']
    assert one(events,'event','training_returned')['pid']==second['pid']
    native=(root/'launcher.log').read_text()
    assert re.findall(r'run_id=(\d+), is_recover_run=(?:True|False)',native)==['0','1']
    lifecycle=[read(p) for p in (root/'areal').rglob('job-lifecycle/*.json')]
    jobs=sorted([e for e in lifecycle if e['log_path'].endswith('/trainer.log')],key=lambda e:e['started_ns'])
    assert len(jobs)==2 and jobs[0]['root_exit_code']!=0 and jobs[1]['root_exit_code']==0
    assert all(e['cleanup']['empty'] for e in lifecycle)
    assert jobs[0]['cleanup']['signals'], 'orphan cleanup not observed'
    for job,reg in zip(jobs,registrations):
        ancestors={p['pid'] for p in reg['ancestry']}
        assert job['supervisor_pid'] in ancestors and job['root_pid'] in ancestors
    assert sent['controller_monotonic_ns']<jobs[0]['finished_ns']<jobs[1]['started_ns']<second['monotonic_ns']
    method=root/'rewardtxn'; generations=[]; head=read(method/'state/control.json')['head']; hashed=0
    while head is not None:
        d=method/'state/generations'/head['generation']
        token=read(d/'token.json'); manifest=read(d/'manifest.json')
        assert sha(d/'token.json')==head['token_sha256']
        assert sha(d/'manifest.json')==token['manifest_sha256'] and sha(d/'intent.json')==token['intent_sha256']
        assert token['parent']==manifest['parent']
        assert not any(p.is_symlink() for p in (d/'checkpoint').rglob('*'))
        files={str(p.relative_to(d/'checkpoint')):p for p in (d/'checkpoint').rglob('*') if p.is_file()}
        assert set(files)==set(manifest['files'])
        for name,info in manifest['files'].items():
            assert files[name].stat().st_size==info['size'] and sha(files[name])==info['sha256']
            hashed+=info['size']
        generations.append((d.name,manifest));head=token['parent']
    generations.reverse()
    assert [g for g,m in generations]==[e['generation'] for e in commits]
    assert abandoned not in {g for g,m in generations}
    assert not (method/'state/generations'/abandoned/'token.json').exists()
    assert [e['event'] for e in events if e.get('generation')==abandoned]==(
        ['update_prepared','optimizer_applied','scheduler_applied','async_scheduled','fault_ready']
        if midwrite else ['update_prepared','optimizer_applied','fault_ready'])
    if midwrite:
        writer=witness['writer']; gate=witness['gate']; job=jobs[0]
        assert writer==read(method/'midwrite-witness.json')
        assert writer['size']>0 and writer['remaining_tensors']>0
        assert writer['generation']==gate['generation']==abandoned and gate['pending']
        assert writer['writer']['ppid']==first['pid']
        assert witness['writer_ancestry'][0]==writer['writer'] and witness['writer_ancestry'][-1]==first['identity']
        assert writer['monotonic_ns']<witness['monotonic_ns']<sent['controller_monotonic_ns']
        assert any(e['pid']==writer['writer']['pid'] for e in job['cleanup']['exits'])
        partial=method/'state/generations'/abandoned/'checkpoint/native'
        assert not (partial/'.metadata').exists()
        path=root/Path(writer['path']).relative_to('/output')
        assert path.stat().st_size==writer['size']
        abandoned_dir=partial.parent.parent
        assert not (abandoned_dir/'manifest.json').exists()
        reconciled=one(events,'event','pending_writer_abandoned')
        assert reconciled['gate']==gate and reconciled['pid']==second['pid']
        assert reconciled['recovered_generation']==commits[0]['generation']
        receipt=root/Path(reconciled['receipt']).relative_to('/output')
        assert sha(receipt)==reconciled['receipt_sha256'] and read(receipt)==job
        assert gate['job']['supervisor']['pid']==job['supervisor_pid']
        assert gate['job']['root']['pid']==job['root_pid']
        assert gate['owner']=={k:first['identity'][k] for k in ('pid','start_time','boot_id')}
        assert job['started_ns']<gate['job']['captured_ns']<job['root_exited_ns']<=job['finished_ns']<reconciled['monotonic_ns']<loaded['monotonic_ns']
    consumed=set(); receipts={}
    for gid,manifest in generations:
        samples=[s for u in manifest['updates'] for g in u['groups'] for s in g['samples']]
        ids={s['sample'] for s in samples}
        assert len(samples)==len(ids)==32 and not consumed.intersection(ids)
        for update in manifest['updates']:
            for group in update['groups']:
                assert group['k']==8 and sorted(s['sample_index'] for s in group['samples'])==list(range(8))
        consumed.update(ids); receipts.update({s['sample']:s['receipt'] for s in samples})
        assert set(manifest['data']['consumed'])==consumed
        assert set(manifest['data']['drawn'])==consumed|{p['sample'] for p in manifest['data']['pending']}
        matched=[one([e for e in events if e.get('generation')==gid],'event',name) for name in
            ['update_prepared','optimizer_applied','scheduler_applied','async_scheduled','async_finalized','committed']]
        assert [e['monotonic_ns'] for e in matched]==sorted(e['monotonic_ns'] for e in matched)
        assert len({e['pid'] for e in matched})==1
        assert matched[3]['call_id']==matched[4]['call_id']
        opt=one(manifest['receipts'],'kind','optimizer'); done=one(manifest['receipts'],'kind','finalize')
        assert opt['successful'] and opt['scheduler_applied'] and done['writer_closed']
        assert opt['physical_updates']==[u['physical_update_id'] for u in manifest['updates']]
        assert done['async_call_ids']==[matched[3]['call_id']]
        assert all(r['snapshot_id']==manifest['data_snapshot_id'] for r in manifest['receipts'])
    lost=one([e for e in events if e.get('generation')==abandoned],'event','update_prepared')
    assert set(lost['samples'])<=consumed and not set(lost['samples']).intersection(
        one([e for e in events if e.get('generation')==commits[0]['generation']],'event','update_prepared')['samples'])
    adopted=set(); blobs=method/'artifacts/blobs'
    for path in (method/'artifacts/samples').glob('*/adoption.json'):
        adoption=read(path); dest=adoption['destination']; sample=dest['sample']
        if sample not in receipts or receipts[sample]['attempt']!=dest: continue
        assert adoption['origin']['attempt']['epoch']<dest['epoch']
        pairs={}
        for stage,key in [('response','response_sha256'),('reward','reward_sha256'),('tensor','tensor_input_sha256')]:
            ref=read(path.parent/(stage+'.json'))['blob']; old_sha=adoption['origin']['payload'][key]
            assert ref['sha256']==receipts[sample]['payload'][key]
            assert sha(blobs/ref['sha256'])==ref['sha256'] and sha(blobs/old_sha)==old_sha
            current=read(blobs/ref['sha256']);old=read(blobs/old_sha)
            assert current['binding']['attempt']==dest and old['binding']['attempt']==adoption['origin']['attempt']
            pairs[stage]=(current['payload'],old['payload'])
        assert pairs['response'][0]==pairs['response'][1]
        assert pairs['reward'][0]['return']==pairs['reward'][1]['return']
        assert pairs['tensor'][0]['fields']==pairs['tensor'][1]['fields']
        for field in pairs['tensor'][0]['fields'].values():
            ref=field['raw'];assert sha(blobs/ref['sha256'])==ref['sha256'] and (blobs/ref['sha256']).stat().st_size==ref['size']
        adopted.add(sample)
    assert set(lost['samples'])<=adopted, 'uncommitted update samples did not replay into final consumption'
    return {'verified':True,'scope':'single-rank '+expected['phase']+' SIGKILL; common lifecycle patch',
        'incomplete_writer_cleaned_and_abandoned':midwrite,
        'native_automatic_retries':1,'signals':1,'physical_optimizer_successes':4,'committed_updates':3,
        'unique_consumed_samples':len(consumed),'adopted_samples_committed':len(adopted),
        'abandoned_update_samples_replayed':len(lost['samples']),'abandoned_generation':abandoned,
        'native_full_state_load_exact':True,'checkpoint_bytes_hashed':hashed,
        'fault_to_next_optimizer_seconds':(optimizers[2]['monotonic_ns']-sent['controller_monotonic_ns'])/1e9,
        'fault_to_next_commit_seconds':(commits[1]['monotonic_ns']-sent['controller_monotonic_ns'])/1e9}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('evidence',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.evidence),indent=2))
