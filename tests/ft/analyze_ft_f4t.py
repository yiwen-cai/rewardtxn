"""Recompute the frozen F4' paired retention result from per-run evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


SAFE = {'correct_recovered', 'safe_discard', 'safe_stop'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_sign_p(positive, negative):
    n = positive + negative
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n)


def accepted_freezes(freeze_path):
    """Original freeze plus its amendment chain (A1, A2, ...), each linked to its predecessor."""
    accepted = [sha256(freeze_path)]
    stem = freeze_path.stem
    for path in sorted(freeze_path.parent.glob('FORMAL_F4T_FREEZE_*-A*.json'),
                       key=lambda p: int(p.stem.rsplit('-A', 1)[1])):
        tag = 'amendment_A' + path.stem.rsplit('-A', 1)[1]
        link = json.loads(path.read_text())[tag]
        assert link.get('previous_freeze_sha256', link.get('original_freeze_sha256')) == accepted[-1], path
        accepted.append(sha256(path))
    assert stem.startswith('FORMAL_F4T_FREEZE_')
    return accepted


def attempts_for(base, name):
    """The planned run followed by its full-pair redos, in order."""
    found, k = [], 0
    while True:
        attempt = name if k == 0 else f'{name}-redo{k}'
        path = base / f'{attempt}-pair.json'
        if not path.exists():
            return found
        found.append((attempt, json.loads(path.read_text())))
        k += 1


def analyze(freeze_path):
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_text())
    assert freeze['kind'] == 'f4t_formal' and freeze['formal_sample'] is True
    base = freeze_path.parent / 'minimal_evidence'
    original_sha = sha256(freeze_path)
    accepted = accepted_freezes(freeze_path)
    pairs, incomplete, invalid_attempts = [], [], []
    all_gpu_hours = 0.0
    for planned in freeze['pairs']:
        attempts = attempts_for(base, planned['name'])
        for attempt, _ in attempts:
            for arm in ('a', 'r'):
                cost_path = base / f'{attempt}-{arm}' / 'cost.json'
                if cost_path.exists():
                    all_gpu_hours += json.loads(cost_path.read_text())['allocated_gpu_hours']
        if not attempts:
            incomplete.append({'name': planned['name'], 'reason': 'pair record missing'})
            continue
        *earlier, (name, record) = attempts
        for attempt, old in earlier:
            assert old['status'] == 'stopped_for_review', attempt
            invalid_attempts.append({'name': attempt, 'failed_arm': old.get('failed_arm'), 'error': old.get('error'),
                                     'cause_correction': old.get('cause_correction_A3')})
        if record.get('status') != 'formal_pair_verified':
            incomplete.append({'name': name, 'reason': record.get('status'), 'error': record.get('error')})
            continue
        assigned = freeze_path.parent / f'FORMAL_FREEZE_{name}.json'
        used = assigned if assigned.exists() else freeze_path
        used_freeze = json.loads(used.read_text())
        assert record['freeze_sha256'] == sha256(used)
        if used != freeze_path:
            assert used_freeze['gpu_assignment_amendment']['original_freeze_sha256'] in accepted
        assert (record['name'], record['scenario'], record['seed'], record['order'], record['formal_sample']) == (
            name, 'F4T', planned['seed'], planned['order'], True)
        arms = {}
        assert {run['arm'] for run in record['runs']} == {'A', 'R'}
        for run in record['runs']:
            root = Path(run['evidence'])
            assert root.name == f"{name}-{run['arm'].lower()}"
            source = json.loads((root / 'source-verification.json').read_text())
            if run['classification'] == 'safe_stop':
                # Amendment A2: valid hit, native recovery could not resume; nothing retained.
                assert source.get('outcome') == 'safe_stop' and source['same_execution_reused'] == 0
                fault = json.loads((root / 'fault-verification.json').read_text())
                assert fault['valid_hit'] and fault['classification'] == 'safe_stop_or_error'
                result = {'classification': 'safe_stop'}
            else:
                result = json.loads((root / 'functional-verification.json').read_text())
            monitor = json.loads((root / 'gpu-load-summary.json').read_text())
            cost = json.loads((root / 'cost.json').read_text())
            count = source['completed_untrained_scores_at_signal']
            kept = source['same_execution_reused']
            assert source['verified'] and source['arm'] == run['arm'] and source['scenario'] == 'F4T'
            assert count > 0 and 0 <= kept <= count and source['discarded'] == count - kept
            assert result['classification'] == run['classification'] in SAFE
            if result['classification'] != 'safe_stop':
                assert result['safety_verified_for_retained_chain'] and result['full_native_reload_verified']
            assert monitor['samples'] > 0 and monitor['violation'] is None
            arms[run['arm']] = {'completed': count, 'kept': kept, 'fraction': kept / count,
                               'classification': result['classification'],
                               'wall_seconds': cost['wall_seconds'],
                               'gpu_hours': cost['allocated_gpu_hours'],
                               'generated_tokens_after_fault': source['generated_tokens_after_fault'],
                               'score_returns_after_fault': source['score_returns_after_fault']}
        difference = arms['R']['fraction'] - arms['A']['fraction']
        pairs.append({'name': name, 'planned_name': planned['name'], 'seed': planned['seed'], 'order': planned['order'],
                      'arms': arms, 'retention_fraction_difference': difference})
    differences = [pair['retention_fraction_difference'] for pair in pairs]
    positive = sum(x > 0 for x in differences)
    negative = sum(x < 0 for x in differences)
    complete = not incomplete and len(pairs) == freeze['planned_pair_count']
    return {'freeze_sha256': original_sha, 'accepted_freeze_chain': accepted,
            'technical_invalid_attempts': invalid_attempts,
            'gpu_hours_all_attempts': all_gpu_hours,
            'gpu_hours_verified_pairs': sum(a['gpu_hours'] for p in pairs for a in p['arms'].values()),
            'planned_pairs': freeze['planned_pair_count'],
            'verified_pairs': len(pairs), 'pairs': pairs, 'incomplete': incomplete,
            'positive': positive, 'negative': negative, 'tied': len(pairs) - positive - negative,
            'median_fraction_difference': statistics.median(differences) if differences else None,
            'exact_two_sided_sign_p': exact_sign_p(positive, negative) if complete else None,
            'primary_claim_eligible': complete and positive > negative and exact_sign_p(positive, negative) < 0.05,
            'scope': '10-update single-machine F4T; one paired run is one independent unit'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.freeze)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: result[key] for key in ('planned_pairs', 'verified_pairs', 'incomplete',
                                                   'exact_two_sided_sign_p', 'primary_claim_eligible')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
