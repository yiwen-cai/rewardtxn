#!/usr/bin/env python3
"""Teacher-forced score-function direction on a single frozen rollout batch."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def score(path, samples):
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    values = []
    with torch.no_grad():
        for s in samples:
            ids = torch.tensor([s['tokens']],device='cuda')
            n = s['response_length']
            start = ids.shape[1]-n
            logits = model(input_ids=ids).logits[0,start-1:-1].float()
            targets = ids[0,start:]
            lp = logits.gather(-1,targets[:,None]).squeeze(-1)-logits.logsumexp(-1)
            assert lp.numel() == n and torch.isfinite(lp).all()
            values.append(float(lp.mean()))
    del model
    torch.cuda.empty_cache()
    return torch.tensor(values,dtype=torch.float64)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    data=Path('/workspace/runs/diagnosis-E7-A-s29-20260910-r1/rollout_debug/0.pt')
    samples=torch.load(data,map_location='cpu',weights_only=False)['samples']
    rewards=torch.tensor([s['reward'] for s in samples],dtype=torch.float64).reshape(4,8)
    advantage=((rewards-rewards.mean(dim=1,keepdim=True))/(rewards.std(dim=1,keepdim=True)+1e-6)).flatten()
    baseline=score('/workspace/models/Qwen2.5-0.5B-Instruct',samples)
    output={'fixed_data': str(data), 'sample_indices':[s['index'] for s in samples],
            'base_mean_logprobs':baseline.tolist(), 'advantages':advantage.tolist(), 'steps':{},
            'note':'Warmup initial LR=0; step1 may be identical. Step2 tests the first nonzero-LR update. '
                   'Repeated fixed data is not a generalization test.'}
    for completed in [1,2,5,10]:
        lp=score(str(args.run/f'checkpoints/iter_{completed-1:07d}_hf'),samples)
        delta=lp-baseline
        output['steps'][completed]={'mean_logprobs':lp.tolist(),
                                    'advantage_weighted_delta':float((advantage*delta).mean()),
                                    'positive_advantage_mean_delta':float(delta[advantage>0].mean()),
                                    'negative_advantage_mean_delta':float(delta[advantage<0].mean()),
                                    'max_abs_delta':float(delta.abs().max())}
        print(completed,output['steps'][completed]['advantage_weighted_delta'],flush=True)
    (args.run/'fixed_score.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__ == '__main__':
    main()
