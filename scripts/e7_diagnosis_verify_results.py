#!/usr/bin/env python3
"""Read-only cross-check of saved diagnostic evaluations (no generation/GPU).

Run in the Slime image. Missing long confirmations remain explicitly incomplete.
This complements, and does not replace, raw debug-batch and runtime audits.
"""
import hashlib
import json
from pathlib import Path

from day2_custom_rm import _v1_reward

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / 'runs/diagnosis-20260910'


def read(path):
    return json.loads(path.read_text())


def main():
    splits = {name: read(DIAG / f'{name}_split.json') for name in
              ['train', 'validation', 'reserved_test', 'historical_eval']}
    source = ROOT / splits['train']['source']
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    all_ids, all_prompts = set(), set()
    for split in splits.values():
        ids = split['eval_indices']
        assert split['source_sha256'] == digest
        assert len(ids) == split['eval_count'] == len(set(ids))
        assert not all_ids.intersection(ids)
        prompts = {json.dumps(rows[i]['prompt'], sort_keys=True, ensure_ascii=False) for i in ids}
        assert not all_prompts.intersection(prompts)
        all_ids.update(ids)
        all_prompts.update(prompts)
    assert all_ids == set(range(len(rows)))
    validation_ids = splits['validation']['eval_indices']
    validation_sha = hashlib.sha256((DIAG / 'validation_split.json').read_bytes()).hexdigest()

    def verify_eval(path, checkpoint, n):
        data = read(path)
        assert data['checkpoint'] == checkpoint
        assert data['split_sha256'] == validation_sha
        assert data['n_total'] == len(data['results']) == n
        assert [r['source_index'] for r in data['results']] == validation_ids[:n]
        for result in data['results']:
            row = rows[result['source_index']]
            assert result['label'] == row.get('label', row.get('answer', ''))
            assert all(message['content'] in result['prompt'] for message in row['prompt'])
            assert result['correct'] == (_v1_reward(result['response'], result['label']) > .5)
            assert 0 <= result['generated_tokens'] <= 2048
            assert not result['truncated'] or result['generated_tokens'] == 2048
            assert 0 <= result['repeated_4gram_fraction'] <= 1
        correct = sum(r['correct'] for r in data['results'])
        truncated = sum(r['truncated'] for r in data['results']) / n
        assert data['n_correct'] == correct and data['accuracy'] == correct / n
        assert data['truncated_fraction'] == truncated
        return {'correct': correct, 'n': n, 'truncated_fraction': truncated,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    output = {'scope': 'saved evaluations regraded; saved consumption summaries cross-checked',
              'source_sha256': digest, 'validation_sha256': validation_sha, 'runs': {}}
    output['base'] = verify_eval(DIAG / 'base_validation.json',
                                '/workspace/models/Qwen2.5-0.5B-Instruct', 100)
    names = [f'diagnosis-E7-{group}-s29-20260910' + ('-r1' if group in 'AD' else '')
             for group in 'ABCDE']
    names += [f'diagnosis-E7-D3-{group}-s29-20260910' for group in ['oracle', 'rewardtxn']]
    for name in names:
        steps = 500 if '-D3-' in name else 50
        run = ROOT / 'runs' / name
        points = [(50, 20), (100, 20), (250, 20)] if steps == 500 else [(x, 20) for x in [5, 10, 20, 30]]
        points.append((steps, 100))
        required = [run / 'diagnosis_summary.json'] + [run / f'diagnostic_eval/step{s}_n{n}.json' for s, n in points]
        missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
        if missing:
            output['runs'][name] = {'status': 'incomplete', 'missing': missing}
            continue
        summary = read(required[0])
        assert summary['train_steps'] == list(range(steps))
        assert summary['all_logged_metrics_finite'] and summary['consumed_split_overlap'] == 0
        assert [r['rollout_id'] for r in summary['consumed_rollouts']] == list(range(steps))
        for batch in summary['consumed_rollouts']:
            assert len(batch['groups']) == 4 and set(batch['groups'].values()) == {8}
            assert len(batch['samples']) == 32
            for sample in batch['samples']:
                assert sample['source_index'] in splits['train']['eval_indices']
                assert sample['logprobs_finite']
                assert sample['response_length'] == sample['logprob_count']
        evaluations = {str(s): verify_eval(run / f'diagnostic_eval/step{s}_n{n}.json',
                       f'/workspace/runs/{name}/checkpoints/iter_{s-1:07d}_hf', n) for s, n in points}
        final = evaluations[str(steps)]
        output['runs'][name] = {'status': 'saved_evidence_verified', 'evaluations': evaluations,
                               'quality_pass': final['correct'] >= output['base']['correct'] - 10
                               and final['truncated_fraction'] <= output['base']['truncated_fraction'] + .1}
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
