#!/usr/bin/env python3.12
"""Read-only CPU audit of the fixed 20260911 pilot evaluations.

Run from any directory (no torch, GPU, model generation or result writes):
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/public/home/caiyiwen/.local/lib/python3.8/site-packages \
  /public/home/caiyiwen/.local/bin/python3.12 \
  /public/home/caiyiwen/rewardtxn/scripts/e7_pilot_eval_audit.py

Python >= 3.10 is required by Slime's math_utils. The path above supplies the
already-installed pure-Python sympy, mpmath and pylatexenc packages. The grader
function is compiled unchanged from its AST to avoid importing Slime/torch.
All comparisons retain the original fixed grader; examples are diagnostic only.
"""
import ast
import collections
import importlib.util
import json
import statistics
from pathlib import Path

from e7_restart_checks import ROOT, SPEC, manifest_report, read, sha


def load_grader():
    path = ROOT / 'third_party/slime/slime/rollout/rm_hub/math_utils.py'
    spec = importlib.util.spec_from_file_location('pilot_audit_math_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = ROOT / 'scripts/day2_custom_rm.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == '_v1_reward')
    namespace = {name: getattr(module, name) for name in
                 ['extract_answer', 'grade_answer_mathd', 'grade_answer_sympy']}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_v1_reward'], module.extract_answer, module.grade_answer_mathd


def main():
    spec = read(SPEC)
    manifest = read(ROOT / spec['pilot_manifest'])
    gate = manifest_report(ROOT / spec['pilot_manifest'], spec, 'pilot')
    grade, extract, mathd = load_grader()
    split = read(ROOT / spec['validation_split'])
    with (ROOT / split['source']).open() as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    base = read(ROOT / spec['base_validation'])
    assert base['checkpoint'] == '/workspace/models/' + spec['model']
    base_correct = {row['source_index'] for row in base['results'] if row['correct']}
    paths = [('base', ROOT / spec['base_validation'])]
    paths += [(Path(run).name, ROOT / run / 'restart_eval/validation.json') for run in manifest['runs']]
    results, evaluations, summaries = {}, {}, {}
    for name, path in paths:
        data = read(path)
        assert data['split_sha256'] == sha(ROOT / spec['validation_split'])
        assert [row['source_index'] for row in data['results']] == split['eval_indices']
        assert data['n_total'] == len(data['results']) == 100
        assert (data['batch_size'], data['max_new_tokens'], data['seed']) == (8, 2048, 29)
        mismatches, missing_boxed, sympy_only = [], [], []
        for index, row in enumerate(data['results']):
            sample = source[row['source_index']]
            assert row['label'] == sample.get('label', sample.get('answer', ''))
            assert all(message['content'] in row['prompt'] for message in sample['prompt'])
            assert row['prompt'] == base['results'][index]['prompt']
            assert type(row['correct']) is bool and type(row['truncated']) is bool
            assert 0 <= row['generated_tokens'] <= 2048
            assert not row['truncated'] or row['generated_tokens'] == 2048
            assert 0 <= row['repeated_4gram_fraction'] <= 1
            if row['correct'] != (grade(row['response'], row['label']) > .5):
                mismatches.append(row['source_index'])
            answer = extract(row['response'])
            if answer is None:
                missing_boxed.append(row['source_index'])
            if row['correct'] and not mathd(answer, row['label']):
                sympy_only.append(row['source_index'])
        correct = {row['source_index'] for row in data['results'] if row['correct']}
        truncated = [row['source_index'] for row in data['results'] if row['truncated']]
        assert data['n_correct'] == len(correct) and data['accuracy'] == len(correct) / 100
        assert data['truncated_fraction'] == len(truncated) / 100
        summaries[name] = {
            'file': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'n_correct': len(correct), 'n_total': 100, 'regrade_mismatches': mismatches,
            'truncated_ids': truncated, 'truncated_fraction': len(truncated) / 100,
            'missing_boxed_ids': missing_boxed, 'sympy_only_correct_ids': sympy_only,
            'gained_vs_base_ids': sorted(correct - base_correct),
            'lost_vs_base_ids': sorted(base_correct - correct),
            'mean_generated_tokens': statistics.mean(row['generated_tokens'] for row in data['results']),
        }
        if name != 'base':
            meta = read(path.parents[1] / 'meta.json')
            results[meta['seed'], meta['baseline_mode']] = data['results']
        evaluations[name] = data

    pairs, losses, gains = [], [], []
    for seed in spec['pilot_seeds']:
        oracle, rtx = results[seed, 'group_rm'], results[seed, 'b6']
        counts = collections.Counter((a['correct'], b['correct']) for a, b in zip(oracle, rtx))
        lost = {a['source_index'] for a, b in zip(oracle, rtx) if a['correct'] and not b['correct']}
        gained = {a['source_index'] for a, b in zip(oracle, rtx) if not a['correct'] and b['correct']}
        losses.append(lost)
        gains.append(gained)
        pairs.append({'seed': seed, 'both_correct': counts[True, True], 'both_wrong': counts[False, False],
                      'oracle_only_correct': len(lost), 'rewardtxn_only_correct': len(gained),
                      'oracle_only_ids': sorted(lost), 'rewardtxn_only_ids': sorted(gained),
                      'identical_full_responses': sum(a['response'] == b['response'] for a, b in zip(oracle, rtx)),
                      'discordant_answers': [
                          {'source_index': a['source_index'], 'label': a['label'],
                           'oracle_answer': extract(a['response']), 'rewardtxn_answer': extract(b['response'])}
                          for a, b in zip(oracle, rtx) if a['correct'] != b['correct']]})

    assets = {}
    checkpoints = [ROOT / 'models' / spec['model']]
    checkpoints += [ROOT / run / 'checkpoints/iter_0000499_hf' for run in manifest['runs']]
    for filename in ['config.json', 'generation_config.json', 'tokenizer_config.json',
                     'tokenizer.json', 'vocab.json', 'merges.txt']:
        hashes = {sha(checkpoint / filename) for checkpoint in checkpoints}
        assert len(hashes) == 1, filename + ' differs between base/pilot checkpoints'
        assets[filename] = hashes.pop()
    common_losses = sorted(set.intersection(*losses))
    examples = {str(index): {
        name: {'correct': row['correct'], 'label': row['label'], 'answer': extract(row['response'])}
        for name, data in evaluations.items() for row in data['results'] if row['source_index'] == index}
        for index in common_losses + [3657]}
    report = {
        'scope': 'saved text regraded and metadata compared; no model regeneration or weight identity verification',
        'pass': not any(value['regrade_mismatches'] for value in summaries.values()),
        'regraded_responses': sum(value['n_total'] for value in summaries.values()),
        'gate': gate, 'evaluations': summaries, 'pairs': pairs,
        'oracle_only_ids_all_seeds': common_losses,
        'rewardtxn_only_ids_all_seeds': sorted(set.intersection(*gains)),
        'identical_base_and_six_checkpoint_asset_sha256': assets,
        'review_examples': examples,
        'interpretation_limits': [
            'All 3 seeds use the same 100 items; 300 pair-item observations are not 300 independent seed replicates.',
            'Power gate is a plug-in approximation at true difference zero; reported zero is not exact TOST power.',
            'Pilot uses validation100 but formal plan uses reserved test500; variance transfer is an assumption.',
            'Example 3657: b6 seed37 reports 3 players and 12 cheerleaders instead of total15; fixed score remains false.',
            'Do not delete difficult items, change the fixed grader, or pool pilot seeds into the formal endpoint.',
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
