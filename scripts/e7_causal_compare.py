#!/usr/bin/env python3
"""Compare actual training tensors and HF updates from fixed-payload replays."""
import ast
import json
import re
from pathlib import Path

import torch
from e7_diagnosis_weight_delta import state

ROOT = Path('/workspace')
OUT = ROOT / 'runs/pilot-causal-audit-20260911'
RUNS = {name: ROOT / f'runs/diagnosis-E7-causal-{name}-s11-20260911'
        for name in ['oracle-r1', 'b6-r1', 'oracle-r2']}


def compare(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        return {'equal': torch.equal(a, b), 'numel': a.numel(),
                'max_abs': float((a.double() - b.double()).abs().max()) if a.numel() else 0.0}
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return {k: compare(v, b[k]) for k, v in a.items()}
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        return [compare(x, y) for x, y in zip(a, b)]
    return {'equal': a == b}


def all_equal(result):
    if isinstance(result, list):
        return all(map(all_equal, result))
    if 'equal' in result:
        return result['equal']
    return all(map(all_equal, result.values()))


def weight_diff(a, b):
    assert a.keys() == b.keys()
    count = 0
    maximum = 0.0
    square = 0.0
    for k in a:
        assert a[k].shape == b[k].shape
        assert torch.isfinite(a[k]).all() and torch.isfinite(b[k]).all()
        delta = a[k].float() - b[k].float()
        count += int(torch.count_nonzero(delta))
        maximum = max(maximum, float(delta.abs().max()))
        square += float(delta.double().square().sum())
    return dict(changed_elements=count, total_elements=sum(x.numel() for x in a.values()),
                max_abs=maximum, l2=square ** .5)


def main():
    torch.set_num_threads(4)
    report = {'runs': {k: str(v) for k, v in RUNS.items()}, 'steps': {}}
    report['logged_metrics'] = {}
    for name, run in RUNS.items():
        metrics = {}
        for line in (run / 'logs/train.log').read_text(errors='replace').splitlines():
            match = re.search(r'model.py:\d+ - step (\d+): (\{.*\})', line)
            if match:
                metrics[int(match[1])] = ast.literal_eval(match[2])
        assert set(metrics) == {0, 1}
        report['logged_metrics'][name] = metrics
    for run in RUNS.values():
        terminal = json.loads((run / 'logs/terminal_state.json').read_text())
        assert terminal['ExitCode'] == 0 and not terminal['OOMKilled']
    base = state(ROOT / 'models/Qwen2.5-0.5B-Instruct')
    for step in [0, 1]:
        data = {k: torch.load(p / f'train_debug/{step}_0.pt', map_location='cpu', weights_only=False)['rollout_data']
                for k, p in RUNS.items()}
        result = {'training_tensor_keys': list(data['oracle-r1']), 'comparisons': {}}
        weights = state(RUNS['oracle-r1'] / f'checkpoints/iter_{step:07d}_hf')
        result['oracle_vs_base'] = weight_diff(weights, base)
        for name in ['b6-r1', 'oracle-r2']:
            details = compare(data['oracle-r1'], data[name])
            other = state(RUNS[name] / f'checkpoints/iter_{step:07d}_hf')
            result['comparisons'][name] = dict(all_training_data_equal=all_equal(details),
                                             training_data=details, weights=weight_diff(weights, other))
            del other
        report['steps'][step] = result
        print(step, {k: v['weights'] for k, v in result['comparisons'].items()}, flush=True)
        del weights
    report['nonzero_update_observed'] = report['steps'][1]['oracle_vs_base']['changed_elements'] > 0
    report['scope'] = 'Fixed first batch, identical source model and seed, same GPU, two steps including warmup; excludes online scheduling and generation.'
    (OUT / 'fixed_update_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['nonzero_update_observed'], 'No representable weight update; extend diagnostic before claiming update parity.'


if __name__ == '__main__':
    main()
