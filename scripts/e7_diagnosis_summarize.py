#!/usr/bin/env python3
"""Summarize a completed diagnostic run including consumed rollout provenance.

Run inside the experiment container (torch required). Historical runs untouched.
"""
import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--expected-steps', type=int, default=50)
    args = parser.parse_args()
    root = Path('/workspace')
    diag = root / 'runs/diagnosis-20260910'
    source = root / 'models/datasets/gsm8k/dapo-gsm8k-train.jsonl'
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    # Debug samples contain already-templated prompts; identify the unique
    # complete user message contained in each, rather than truncated text.
    content_to_index = {r['prompt'][0]['content']: i for i,r in enumerate(rows)}
    assert len(content_to_index) == len(rows)
    train_ids = set(json.loads((diag/'train_split.json').read_text())['eval_indices'])
    forbidden = set(range(len(rows))) - train_ids
    metrics = []
    with (args.run/'logs/train.log').open() as stream:
        for line in stream:
            m = re.search(r'(?:data\.py:\d+ - rollout|model\.py:\d+ - step) (\d+): (\{.*\})', line)
            if m:
                metrics.append({'step': int(m[1]), **ast.literal_eval(m[2])})
    consumed = []
    for file in sorted((args.run/'rollout_debug').glob('*.pt'), key=lambda p: int(p.stem)):
        data = torch.load(file, map_location='cpu', weights_only=False)
        groups = Counter()
        sample_rows = []
        for sample in data['samples']:
            prompt = sample['prompt']
            if isinstance(prompt,list):
                match = [content_to_index[m['content']] for m in prompt
                         if m.get('content') in content_to_index]
            else:
                match = [i for content,i in content_to_index.items() if content in prompt]
            assert len(match) == 1, (file, sample['index'], match)
            source_id = match[0]
            assert source_id not in forbidden, (file,source_id)
            groups[sample['group_index']] += 1
            tokens = sample['tokens']
            length = sample['response_length']
            probs = sample.get('rollout_log_probs')
            sample_rows.append({'source_index': source_id, 'sample_index': sample['index'],
                                'group_index': sample['group_index'], 'reward': sample['reward'],
                                'response_length': length, 'status': sample['status'],
                                'weight_versions': sample.get('weight_versions'),
                                'token_count': len(tokens),
                                'logprob_count': len(probs) if probs is not None else None,
                                'logprobs_finite': all(math.isfinite(x) for x in probs) if probs is not None else None,
                                'response_sha256': hashlib.sha256(sample['response'].encode()).hexdigest()})
        assert len(sample_rows) == 32 and len(groups) == 4 and set(groups.values()) == {8}, (file,groups)
        assert all(s['response_length'] == s['logprob_count'] and s['logprobs_finite'] for s in sample_rows)
        consumed.append({'rollout_id': data['rollout_id'], 'groups': dict(groups), 'samples': sample_rows})
    train_steps = sorted({m['step'] for m in metrics if 'train/loss' in m})
    assert train_steps == list(range(args.expected_steps)), train_steps
    assert sorted(d['rollout_id'] for d in consumed) == train_steps
    output = {'run': str(args.run), 'train_steps': train_steps,
              'metrics': metrics, 'consumed_rollouts': consumed,
              'consumed_split_overlap': 0,
              'all_logged_metrics_finite': all(math.isfinite(v) for m in metrics for v in m.values()
                                               if isinstance(v,(int,float))),
              'note': 'Only serialized consumed batches; callback-only/aborted/dropped groups excluded.'}
    (args.run/'diagnosis_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['metrics','consumed_rollouts']},indent=2))


if __name__ == '__main__':
    main()
