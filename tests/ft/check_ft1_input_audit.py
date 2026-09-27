"""Independent no-fault input audit from observer tensors and original labels."""
import asyncio
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def fingerprint(tokens,versions,mask):
    return hashlib.sha256(json.dumps([tokens,versions,mask],separators=(',',':')).encode()).hexdigest()


def batch_rows(batches, training=False):
    result=[]
    for batch in batches:
        assert all(len(batch[k])==len(batch['attention_mask']) for k in ('input_ids','versions','loss_mask'))
        for i,mask in enumerate(batch['attention_mask']):
            assert all(v in (0,1) for v in mask)
            assert all(len(batch[k][i])==len(mask) for k in ('input_ids','versions','loss_mask'))
            assert all(v in (0,1) for v in batch['loss_mask'][i])
            parts=[[v for v,valid in zip(batch[k][i],mask) if valid] for k in ('input_ids','versions','loss_mask')]
            parts[2]=[int(v) for v in parts[2]]
            if training:
                assert parts[2][-1]==0
                parts[2]=[0]+parts[2][:-1]  # PPO next-token shift, actor.py:_compute_advantages
            key=fingerprint(*parts)
            result.append((key,batch.get('rewards',[None]*len(batch['attention_mask']))[i]))
    return result


def drop_killed_update(root,case,pilot,taken,trained,applied):
    """Mirror of check_ft1_chain.killed_update: F4' may SIGKILL between train_batch and optimizer_end."""
    done={a['update_id'] for a in applied}
    open_=[t for t in trained if t['update_id'] not in done]
    if not open_:return taken,trained
    assert case['scenario']=='F4T' and len(open_)==1,'unapplied train_batch outside F4T kill'
    t=open_[0]
    same=[e for e in pilot if e['pid']==t['pid'] and e['process_incarnation']==t['process_incarnation']]
    assert same[-1] is t,'unapplied train_batch is not the last event of its process'
    prior=[e for e in same if e['event']=='batch_taken' and e['monotonic_ns']<t['monotonic_ns']]
    assert prior and prior[-1] in taken
    signal=[e for e in rows(root/'events.jsonl') if e['kind']=='signal_sent']
    assert len(signal)==1 and t['monotonic_ns']<signal[0]['controller_monotonic_ns']
    later=[e for e in pilot if e['monotonic_ns']>t['monotonic_ns']]
    assert later and all(e['monotonic_ns']>signal[0]['controller_monotonic_ns'] for e in later)
    return [b for b in taken if b is not prior[-1]],[x for x in trained if x is not t]


async def main():
    from transformers import AutoTokenizer
    from areal.api import AsyncRewardWrapper
    from areal.reward.gsm8k import gsm8k_reward_fn
    from scripts.ft.reward_return import verifier_fingerprint
    root,out=Path(sys.argv[1]),Path(sys.argv[2])
    raw=Path('/workspace/runs/diagnosis-20260910/train.jsonl')
    data=rows(raw)
    events=[e for p in (root/'observer-ft1').glob('*.jsonl') for e in rows(p)]
    config=next(e for e in events if e['event']=='configured')
    assert config['verifier_sha256']==verifier_fingerprint()
    generated=defaultdict(list)
    for e in events:
        if e['event']=='generation_complete':
            key=fingerprint(e['input_tokens']+e['output_tokens'],[-1]*len(e['input_tokens'])+e['output_versions'],[0]*len(e['input_tokens'])+[1]*len(e['output_tokens']))
            generated[key].append(e)
    pilot=sorted([e for p in (root/'observer-pilot').glob('*.jsonl') for e in rows(p)],key=lambda e:e['monotonic_ns'])
    taken=[e for e in pilot if e['event']=='batch_taken']
    trained=[e for e in pilot if e['event']=='train_batch']
    applied=[e for e in pilot if e['event']=='optimizer_end']
    case=json.loads((root/'ft1-case.json').read_text());no_fault=case['scenario']=='no_fault'
    taken,trained=drop_killed_update(root,case,pilot,taken,trained,applied)
    assert len(taken)==len(trained)==len(applied)
    if no_fault:assert len(applied)==case.get('steps',10)
    assert applied,'no optimizer input available for authority audit'
    cases=[];claimed=Counter();groups=defaultdict(set)
    for batch,train,update in zip(taken,trained,applied):
        assert batch['monotonic_ns']<train['monotonic_ns']<update['monotonic_ns']
        assert train['update_id']==update['update_id'] and update['stats']['update_successful']==1
        inputs=batch_rows(batch['batches']);physical=batch_rows(train['batches'],training=True)
        assert len(inputs)==len(physical)==32
        assert Counter(k for k,_ in inputs)==Counter(k for k,_ in physical)
        for key,reward in inputs:
            possible=generated[key]
            assert possible and claimed[(batch['pid'],key)]<len(possible),'duplicate or unobserved training input'
            event=possible[claimed[(batch['pid'],key)]];claimed[(batch['pid'],key)]+=1
            # Identical responses can be indistinguishable among sample slots;
            # report that ambiguity rather than invent a sample-to-update proof.
            assert len({p['source_row_id'] for p in possible})==1
            groups[event['source_row_id']].add(event['sample_idx'])
            cases.append((event,reward))
    tokenizer=AutoTokenizer.from_pretrained('/workspace/models/Qwen2.5-0.5B-Instruct',local_files_only=True)
    reward_fn=AsyncRewardWrapper(gsm8k_reward_fn,max_workers=4)
    async def check(event,reward):
        label=data[event['source_row_id']]['label']
        score=await reward_fn(tokenizer.decode(event['input_tokens']),tokenizer.decode(event['output_tokens']),event['input_tokens'],event['output_tokens'],answer=label)
        assert score==reward,(event['source_row_id'],event['sample_idx'],reward,score)
        return score
    scores=await asyncio.gather(*(check(e,r) for e,r in cases))
    ambiguities=sum(1 for (pid,key),count in claimed.items() if len(generated[key])>1)
    report={'scope':'independent physical optimizer input multiset (rollback may repeat across trainer incarnations), original-label re-scoring, physical optimizer linkage; not persistent-chain/reload proof',
            'verified':True,'arm':config['arm'],'updates':len(applied),'rows':len(cases),'positive_rewards':sum(scores),
            'distinct_source_rows':len(groups),'all_groups_have_eight_distinct_slots':all(v==set(range(8)) for v in groups.values()),
            'identical_response_identity_ambiguities':ambiguities,'data_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
            'verifier_sha256':verifier_fingerprint()}
    (out/'input-verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':asyncio.run(main())
