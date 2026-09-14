#!/usr/bin/env python3
"""CPU parameter-change audit of saved diagnostic HF checkpoints."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def state(path):
    tensors = {}
    for shard in sorted(path.glob('*.safetensors')):
        with safe_open(shard, framework='pt', device='cpu') as stream:
            for key in stream.keys():
                assert key not in tensors
                tensors[key] = stream.get_tensor(key)
    assert tensors
    return tensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    base = state(Path('/workspace/models/Qwen2.5-0.5B-Instruct'))
    result = {}
    for step in [4,9,19,29,49]:
        current = state(args.run/f'checkpoints/iter_{step:07d}_hf')
        assert current.keys() == base.keys()
        stats = {}
        for name, weight in base.items():
            a, b = weight.float(), current[name].float()
            assert a.shape == b.shape
            delta = b-a
            stats[name] = {'base_l2': float(a.norm()), 'delta_l2': float(delta.norm()),
                           'max_abs_delta': float(delta.abs().max()), 'finite': bool(torch.isfinite(b).all())}
        base_norm = sum(s['base_l2']**2 for s in stats.values())**.5
        delta_norm = sum(s['delta_l2']**2 for s in stats.values())**.5
        result[step+1] = {'global_relative_l2': delta_norm/base_norm,
                          'all_finite': all(s['finite'] for s in stats.values()), 'tensors': stats}
        print(step+1, result[step+1]['global_relative_l2'],flush=True)
    (args.run/'weight_delta.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
