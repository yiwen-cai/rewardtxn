#!/usr/bin/env python3
"""Descriptive v4 screening analysis; never launches confirmation or drops failures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile

from paper_statistics import student_t_quantile

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('O', 'R', 'DBM', 'LOGM', 'BOTHM', 'PAYLOAD', 'LITE')
SEEDS = (11, 23, 37)
CONTRASTS = {
    'R-O': {'R': 1, 'O': -1},
    'DBM-R': {'DBM': 1, 'R': -1},
    'LOGM-R': {'LOGM': 1, 'R': -1},
    'BOTHM-R': {'BOTHM': 1, 'R': -1},
    'BOTHM-DBM-LOGM+R': {'BOTHM': 1, 'DBM': -1, 'LOGM': -1, 'R': 1},
    'PAYLOAD-R': {'PAYLOAD': 1, 'R': -1},
    'LITE-BOTHM': {'LITE': 1, 'BOTHM': -1},
    'LITE-O': {'LITE': 1, 'O': -1},
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    xs = list(values.values())
    n = len(xs)
    mean = statistics.mean(xs) if xs else None
    interval = None
    if n >= 2:
        half = student_t_quantile(.975, n - 1) * statistics.stdev(xs) / math.sqrt(n)
        interval = [mean - half, mean + half]
    return dict(n_seed_pairs=n, by_seed_pp=values, mean_pp=mean,
                range_pp=[min(xs), max(xs)] if xs else None,
                paired_t95_ci_pp=interval,
                all_positive=bool(xs) and all(x > 0 for x in xs),
                all_negative=bool(xs) and all(x < 0 for x in xs),
                inference='descriptive; n=3 screening, fixed validation100; no multiplicity-adjusted claim')


def validate_run(run, arm, seed, freeze_hash, split_hash):
    """Require final control evidence and the post-training audit, not a step counter."""
    success, evaluation, audit = [read(run / name) for name in
                                  ('run_success.json', 'evaluation.json', 'consumption_audit.json')]
    meta = read(run / 'meta.json')
    assert not (run / 'failure.json').exists(), 'failure receipt present'
    assert success['status'] == 'technical_success'
    assert (success['arm'], success['seed'], success['steps']) == (arm, seed, 500)
    assert meta['freeze_sha256'] == freeze_hash, 'run references another freeze'
    decision, shutdown = success['decision'], success['shutdown']
    assert decision['fatal'] is False
    assert decision['nonce'] == meta['environment']['ABLATION_RUN_NONCE']
    assert shutdown['decision'] == decision and read(run / 'control/shutdown.json') == shutdown
    assert set(decision['components']) >= {'driver', 'rm-host', 'rm-worker', 'evaluator'}
    manifest_path = run / 'evaluation_manifest.json'
    assert sha(manifest_path) == decision['barriers']['evaluator']['manifest_sha256']
    manifest = read(manifest_path)
    assert manifest['fatal'] is False and manifest['run_nonce'] == decision['nonce']
    covered = set()
    for item in manifest['files']:
        path = Path(item['path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        assert sha(path) == item['sha256'], 'evaluation artifact changed after approval'
        covered.add(path.resolve())
    assert {run.resolve() / 'evaluation.json', run.resolve() / 'consumption_audit.json'} <= covered
    assert evaluation['n_total'] == 100 and len(evaluation['results']) == 100
    assert len({r['source_index'] for r in evaluation['results']}) == 100
    assert all(type(r['correct']) is bool for r in evaluation['results'])
    assert evaluation['n_correct'] == sum(r['correct'] for r in evaluation['results'])
    assert math.isclose(evaluation['accuracy'], evaluation['n_correct'] / 100, abs_tol=1e-12)
    assert (evaluation['seed'], evaluation['batch_size'], evaluation['max_new_tokens']) == (29, 8, 2048)
    assert evaluation['split_sha256'] == split_hash
    assert Path(evaluation['checkpoint']).name == 'iter_0000499_hf'
    assert audit['consumed'] == 16000 and audit['consumed_payload_coverage'] == 1
    for key in ('consumed_reward_mismatches', 'consumed_source_label_and_rescore_mismatches',
                'unknown_attempt_terminal_count'):
        assert audit[key] == 0, key
    assert audit['logged'] - audit['consumed'] == audit['tail_count'] == len(audit['tail_ids'])
    return evaluation, audit


def decide(contrasts, auxiliary, complete):
    if not complete:
        return dict(status='incomplete', positive_candidates=[], negative_candidates=[],
                    interpretation='Do not select candidates before all 21 planned runs are valid.')
    reproduced = contrasts['R-O']['mean_pp'] < 0
    positive, negative = [], []
    for arm in ARMS[2:]:
        s = auxiliary[arm + '-R']
        if s['all_positive'] and s['mean_pp'] >= 2:
            positive.append(arm)
        if s['all_negative'] and s['mean_pp'] <= -2:
            negative.append(arm)
    # LITE changes a bundle whose component count is not unambiguously specified.
    # Preserve the plan's tie rule; do not invent a numerical rank for that bundle.
    return dict(status='screening_complete', contemporary_R_minus_O_negative=reproduced,
                positive_candidates=positive if reproduced else [],
                improvement_signals_without_historical_attribution=positive if not reproduced else [],
                negative_candidates=negative,
                interpretation=('Candidates require independent confirmation; no unique root cause established.'
                                if reproduced else 'Historical negative gap not reproduced; no explanation established.'),
                candidate_priority_rule='Fewer changed components first, then larger mean; explicitly define bundle counting before selection.',
                negative_followup='First audit semantics/implementation, then freeze any negative confirmation direction.',
                interaction_followup='A combined effect needs a four-cell confirmation to establish interaction.',
                confirmation_started=False)


def analyze(batch, history_manifest):
    freeze = read(batch / 'freeze.json')
    schedule = freeze['schedule']
    assert [b['seed'] for b in schedule] == list(SEEDS)
    assert all(len(b['order']) == 7 and set(b['order']) == set(ARMS) for b in schedule)
    split_hash = freeze['data']['runs/diagnosis-20260910/validation_split.json']
    values, rows = {}, []
    for block in schedule:
        seed = block['seed']
        for arm in block['order']:
            run = batch / 'runs' / f'screen-{len(rows):02}-{arm}-s{seed}'
            row = dict(arm=arm, seed=seed, path=str(run), status='missing')
            if run.exists():
                try:
                    ev, audit = validate_run(run, arm, seed, sha(batch / 'freeze.json'), split_hash)
                    values[arm, seed] = ev['n_correct']  # denominator100 => count is percentage points
                    row.update(status='valid', accuracy_pp=ev['n_correct'],
                               truncated_fraction=ev.get('truncated_fraction'), tail_count=audit['tail_count'],
                               artifacts_sha256={n: sha(run / n) for n in
                                                 ('run_success.json', 'evaluation.json', 'consumption_audit.json')})
                except (AssertionError, KeyError, ValueError, TypeError, OSError) as exc:
                    row.update(status='invalid_or_unfinished', reason=repr(exc))
                    if (run / 'failure.json').exists():
                        try:
                            row['failure'] = read(run / 'failure.json')
                        except (ValueError, OSError) as failure:
                            row['failure_read_error'] = repr(failure)
            rows.append(row)
    def contrast(coefficients):
        return summary({str(s): sum(c * values[a, s] for a, c in coefficients.items())
                        for s in SEEDS if all((a, s) in values for a in coefficients)})
    contrasts = {name: contrast(c) for name, c in CONTRASTS.items()}
    auxiliary = {a + '-R': contrast({a: 1, 'R': -1}) for a in ARMS[2:]}
    complete = len(values) == 21
    history = dict(status='unavailable', comparisons={}, limitation=
                   'Historical repeats are descriptive only: changed devices/time/storage/observations; not new independent seeds.')
    try:
        old = read(history_manifest)
        for arm, marker in [('O', '-group_rm-'), ('R', '-b6-')]:
            differences, sources = {}, []
            for seed in SEEDS:
                matches = [r for r in old['runs'] if marker in r and f'-s{seed}-' in r]
                assert len(matches) == 1
                path = ROOT / matches[0] / 'restart_eval/validation.json'
                ev = read(path)
                assert ev['n_total'] == 100 and ev['split_sha256'] == split_hash
                sources.append(dict(seed=seed, path=str(path), sha256=sha(path), accuracy_pp=ev['n_correct']))
                if (arm, seed) in values:
                    differences[str(seed)] = values[arm, seed] - ev['n_correct']
            history['comparisons'][arm + '_screen-' + arm + '_pilot'] = dict(summary(differences), sources=sources)
        history['status'] = 'available'
    except (AssertionError, KeyError, ValueError, TypeError, OSError) as exc:
        history['error'] = repr(exc)
    return dict(status='complete' if complete else 'incomplete', batch=str(batch),
                freeze_sha256=sha(batch / 'freeze.json'), expected_runs=21, valid_runs=len(values),
                runs=rows, contrasts=contrasts, candidate_comparisons=auxiliary, historical=history,
                decision=decide(contrasts, auxiliary, complete),
                limitations=['Screening has three independent seeds, not 21 independent replicates.',
                             '100 validation questions and training steps are not independent training repeats.',
                             'Technical failures remain listed; no automatic rerun, exclusion, seed substitution, or confirmation.',
                             'Quality declines remain results; no accuracy or truncation filter excludes a technically valid run.',
                             'Pointwise Student t intervals are descriptive and do not establish equivalence or a unique cause.'])


def self_test():
    s = summary({'11': 2, '23': 2, '37': 2})
    assert s['mean_pp'] == 2 and s['paired_t95_ci_pp'] == [2, 2]
    assert summary({'11': 1})['paired_t95_ci_pp'] is None
    c = {'R-O': summary({'11': -1, '23': -1, '37': -1})}
    aux = {a + '-R': summary({'11': 0, '23': 3, '37': 3}) for a in ARMS[2:]}
    aux['PAYLOAD-R'] = s
    aux['DBM-R'] = summary({'11': -2, '23': -2, '37': -2})
    d = decide(c, aux, True)
    assert d['positive_candidates'] == ['PAYLOAD'] and d['negative_candidates'] == ['DBM']
    assert decide(c, aux, False)['positive_candidates'] == []
    c['R-O'] = summary({'11': 0, '23': 0, '37': 0})
    assert decide(c, aux, True)['positive_candidates'] == []
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        spec = read(ROOT / 'docs/experiments/rewardtxn-ablation-20260911/design.json')
        (p / 'freeze.json').write_text(json.dumps(dict(schedule=spec['schedule'], data={
            'runs/diagnosis-20260910/validation_split.json': 'missing'})))
        report = analyze(p, p / 'nonexistent_history.json')
        assert report['status'] == 'incomplete' and len(report['runs']) == 21 and report['valid_runs'] == 0
        assert not report['decision']['positive_candidates']
        accuracy = dict(O=75, R=72, DBM=73, LOGM=72, BOTHM=71, PAYLOAD=75, LITE=70)
        for row in report['runs']:
            run = Path(row['path'])
            run.mkdir(parents=True)
            (run / 'control').mkdir()
            arm, seed = row['arm'], row['seed']
            nonce = f'{arm}-{seed}'
            evaluation = dict(n_total=100, n_correct=accuracy[arm], accuracy=accuracy[arm] / 100,
                              results=[dict(source_index=i, correct=i < accuracy[arm]) for i in range(100)],
                              seed=29, batch_size=8, max_new_tokens=2048, split_sha256='missing',
                              checkpoint=str(run / 'checkpoints/iter_0000499_hf'))
            audit = dict(consumed=16000, logged=16000, tail_count=0, tail_ids=[],
                         consumed_payload_coverage=1, consumed_reward_mismatches=0,
                         consumed_source_label_and_rescore_mismatches=0, unknown_attempt_terminal_count=0)
            for name, obj in [('evaluation.json', evaluation), ('consumption_audit.json', audit),
                              ('meta.json', dict(freeze_sha256=sha(p / 'freeze.json'),
                                                 environment={'ABLATION_RUN_NONCE': nonce}))]:
                (run / name).write_text(json.dumps(obj))
            manifest = dict(fatal=False, run_nonce=nonce, files=[
                dict(path=str(run / name), sha256=sha(run / name))
                for name in ('evaluation.json', 'consumption_audit.json')])
            (run / 'evaluation_manifest.json').write_text(json.dumps(manifest))
            decision = dict(nonce=nonce, fatal=False, components=['driver', 'rm-host', 'rm-worker', 'evaluator'],
                            barriers={'evaluator': {'manifest_sha256': sha(run / 'evaluation_manifest.json')}})
            shutdown = dict(decision=decision)
            (run / 'control/shutdown.json').write_text(json.dumps(shutdown))
            (run / 'run_success.json').write_text(json.dumps(dict(
                status='technical_success', arm=arm, seed=seed, steps=500, decision=decision, shutdown=shutdown)))
        report = analyze(p, p / 'nonexistent_history.json')
        assert report['status'] == 'complete' and report['valid_runs'] == 21, report['runs']
        assert report['decision']['positive_candidates'] == ['PAYLOAD']
        assert report['decision']['negative_candidates'] == ['LITE']
        assert report['contrasts']['BOTHM-DBM-LOGM+R']['mean_pp'] == -2
        assert all(s['n_seed_pairs'] == 3 for s in report['contrasts'].values())
        # Post-approval tampering must invalidate the matrix, even if accuracy improves.
        (Path(report['runs'][0]['path']) / 'evaluation.json').write_text('{}')
        report = analyze(p, p / 'nonexistent_history.json')
        assert report['status'] == 'incomplete' and report['valid_runs'] == 20
        assert not report['decision']['positive_candidates']
    print('self-test passed: complete21/incomplete20, artifact tampering, interaction, inclusive2pp, strict directions, zero baseline gap, missing matrix, n<2, constant differences')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--history-manifest', type=Path, default=ROOT / 'runs/e7_restart_0.5B_20260911_pilot.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.batch is None:
            parser.error('--batch is required')
        report = analyze(args.batch.resolve(), args.history_manifest)
        text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        else:
            print(text, end='')
