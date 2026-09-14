#!/usr/bin/env python3
"""Post-training consumption audit and fixed endpoint evaluation under control."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch
from ablation_control import Client
from ablation_runtime import write_json, empty_terminal, component_manifest


def read_jsonl(path):
    # Unicode line separators inside JSON strings are not JSONL delimiters.
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    client=Client.from_env('evaluator')
    run=Path(os.environ['ABLATION_RUN_DIR'])
    steps=int(os.environ['ABLATION_STEPS'])
    try:
        sys.argv=['e7_diagnosis_summarize.py',str(run),'--expected-steps',str(steps)]
        runpy.run_path('/workspace/scripts/e7_diagnosis_summarize.py',run_name='__main__')
        records=read_jsonl(run/'diagnostic_export/rewards.jsonl')
        def key(s): return ':'.join(str(s.get(k)) for k in ['group_index','index','rollout_id'])
        logged={key(s):s for s in records}
        assert len(logged)==len(records), 'duplicate audit ID'
        from day2_custom_rm import _v1_reward
        source=read_jsonl('/workspace/models/datasets/gsm8k/dapo-gsm8k-train.jsonl')
        summary=json.loads((run/'diagnosis_summary.json').read_text())
        source_map={(str(r['rollout_id']),x['group_index'],x['sample_index']):x['source_index']
                    for r in summary['consumed_rollouts'] for x in r['samples']}
        consumed=set()
        for path in (run/'rollout_debug').glob('*.pt'):
            for s in torch.load(path,map_location='cpu',weights_only=False)['samples']:
                k=key(s)
                assert k not in consumed and k in logged,(k,'duplicate/missing binding')
                consumed.add(k)
                assert s['reward']==logged[k]['reward'],(k,'reward')
                assert isinstance(s['response'],str) and s['label'] is not None
                source_id=source_map[(path.stem,s['group_index'],s['index'])]
                assert s['label']==source[source_id]['label'],(k,'source label mismatch')
                assert _v1_reward(s['response'],s['label'])==s['reward'],(k,'rescoring mismatch')
                if 'response' in logged[k]:
                    assert logged[k]['response']==s['response'][:4000] and logged[k]['label']==s['label'][:200]
        assert len(consumed)==steps*32
        audit=dict(consumed=len(consumed),logged=len(logged),tail_count=len(set(logged)-consumed),
                   tail_ids=sorted(set(logged)-consumed),consumed_payload_coverage=1.,
                   tail_payload_guaranteed=False,consumed_reward_mismatches=0)
        events=read_jsonl(run/'diagnostic_export/events.jsonl')
        attempts={}
        consumption={}
        for e in events:
            if e['kind']=='attempt_start':
                assert e['attempt'] not in attempts
                attempts[e['attempt']]={'ids':e['ids'],'terminal':None,'rm_entered':False,'rm_returned':False,'rm_cancelled':False,'consumed_ids':[]}
            elif e['kind'] in ('attempt_end','attempt_cancel'):
                row=attempts[e['attempt']]; assert row['terminal'] is None
                row['terminal']=e['kind']; row['terminal_ids']=e['ids']
            elif e['kind'] in ('reward_start','reward_return','reward_cancelled'):
                row=attempts[e['attempt_id']]
                if e['kind']=='reward_start': row['rm_entered']=True; row['rm_ids']=e['ids']
                elif e['kind']=='reward_return': row['rm_returned']=True
                else: row['rm_cancelled']=True
            elif e['kind']=='consumed':
                for k,attempt in zip(e['ids'],e['attempt_ids']):
                    assert k not in consumption
                    consumption[k]=attempt
                    row=attempts[attempt]; assert row['rm_returned'] and k in row['rm_ids']
                    row['consumed_ids'].append(k)
        assert set(consumption)==consumed
        assert all(r['terminal'] is not None for r in attempts.values())
        audit['attempts']=attempts
        audit['attempt_count']=len(attempts)
        audit['unscored_attempts']=sum(not r['rm_entered'] for r in attempts.values())
        audit['unknown_attempt_terminal_count']=0
        audit['consumed_source_label_and_rescore_mismatches']=0
        scheduler=json.loads((run/'scheduler_rank0.json').read_text())
        assert scheduler['train_iters']==500 and scheduler['num_rollout']==500
        assert scheduler['scheduler']['lr_decay_steps']==16000
        assert scheduler['scheduler']['lr_warmup_steps']==320
        assert scheduler['scheduler']['wd_incr_steps']==16000
        assert scheduler['scheduler']['lr_decay_style']=='constant'
        audit['scheduler']=scheduler
        write_json(run/'consumption_audit.json',audit)
        if steps==500:
            sys.argv=['e7_diagnosis_eval.py','--checkpoint',str(run/'checkpoints/iter_0000499_hf'),
                      '--split','/workspace/runs/diagnosis-20260910/validation_split.json',
                      '--output',str(run/'evaluation.json')]
            runpy.run_path('/workspace/scripts/e7_diagnosis_eval.py',run_name='__main__')
        client.event('audit_and_evaluation_completed',steps=steps)
        artifacts=[run/'consumption_audit.json',run/'diagnosis_summary.json']
        if steps==500: artifacts.append(run/'evaluation.json')
        manifest=write_json(run/'evaluation_manifest.json',dict(run_nonce=os.environ['ABLATION_RUN_NONCE'],fatal=False,
                    files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in artifacts]))
        permit=client.prepare_close(manifest,empty_terminal())
        write_json(run/'evaluator_permit.json',permit)
        client.close_transport()
    except BaseException as exc:
        client.fatal('evaluation_failure',error=repr(exc))
        raise


if __name__=='__main__': main()
