#!/usr/bin/env python3
"""Reproducible, non-destructive E7 diagnostic data and log audit."""
import ast
import csv
import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/diagnosis-20260910'


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'models/datasets/gsm8k/dapo-gsm8k-train.jsonl'
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    spec = json.loads((ROOT / 'prereg/eval_splits/gsm8k_eval500_seed42.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == spec['source_sha256']
    historical = set(spec['eval_indices'])
    remaining = sorted(set(range(len(rows))) - historical)
    random.Random(20260910).shuffle(remaining)
    # New validation/test are unobserved by NEW training only; old runs may have
    # consumed them. The test is reserved, not evaluated during diagnosis.
    validation, test, train = remaining[:100], remaining[100:600], remaining[600:]
    parts = {'train': train, 'validation': validation, 'reserved_test': test,
             'historical_eval': sorted(historical)}
    fingerprints = {k: {json.dumps(rows[i]['prompt'], sort_keys=True) for i in ids}
                    for k, ids in parts.items()}
    for a in parts:
        for b in parts:
            if a != b:
                assert not fingerprints[a] & fingerprints[b], (a, b)
    for name, ids in parts.items():
        split = dict(source=str(source.relative_to(ROOT)), eval_indices=ids,
                     eval_count=len(ids), source_sha256=spec['source_sha256'])
        (OUT / f'{name}_split.json').write_text(json.dumps(split, indent=2) + '\n')
    (OUT / 'train.jsonl').write_text(''.join(json.dumps(rows[i]) + '\n' for i in train))
    isolated = sorted(set(range(len(rows))) - historical)
    (OUT / 'train_excluding_old_eval.jsonl').write_text(
        ''.join(json.dumps(rows[i]) + '\n' for i in isolated))
    result = {'source_rows': len(rows), 'historical_eval_in_source_pool': len(historical),
              'historical_consumption': 'not reconstructed; pool overlap is not proof of updates',
              'counts': {k: len(v) for k, v in parts.items()},
              'prompt_intersections': 0, 'split_seed': 20260910,
              'new_test_caveat': 'Reserved for new runs, not proven unseen by old runs', 'runs': {}}
    batches = [(29,'20260903-194500'),(42,'20260903-194500'),
               (73,'20260904-093948'),(101,'20260904-093948')]
    for seed, batch in batches:
        for group in ['oracle','rewardtxn']:
            name = f'formal-E7-clean-{group}-0.5B-4gpu-s{seed}-{batch}'
            events = []
            for line in (ROOT / 'runs' / name / 'logs/train.log').read_text().splitlines():
                match = re.search(r'(?:data\.py:\d+ - rollout|model\.py:\d+ - step) (\d+): (\{.*\})', line)
                if match and int(match[1]) < 100:
                    values = ast.literal_eval(match[2])
                    events.append({'step': int(match[1]), **values})
            keys = sorted(set().union(*(e.keys() for e in events)))
            with (OUT / f'{name}_metrics.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=keys)
                writer.writeheader()
                writer.writerows(events)
            windows = {}
            for lo, hi in [(0,4),(5,9),(10,19),(20,29),(30,49),(50,99)]:
                window = [e for e in events if lo <= e['step'] <= hi]
                windows[f'{lo}-{hi}'] = {key: sum(e[key] for e in window if key in e) /
                                        sum(key in e for e in window)
                                        for key in keys if key != 'step' and any(key in e for e in window)}
            result['runs'][name] = windows
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'runs'}, indent=2))


if __name__ == '__main__':
    main()
