"""Prove physical generation and scoring origins for the minimal F2 comparison.

Runs only after the existing input, chain, and fresh-process load acceptance.
An identical token string with a different request or sample attempt is redo.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from check_ft1_input_audit import batch_rows, fingerprint, rows


def read(path):
    return json.loads(path.read_text())


def events(root, folder):
    return [event for path in (root / folder).glob('*.jsonl') for event in rows(path)]


def unique(items, label):
    assert len(items) == 1, (label, len(items))
    return items[0]


def check_accepted(root):
    status = read(root / 'acceptance-status.json')
    assert status['result'] == 'functional_verification_written', status['result']
    assert read(root / 'chain-verification.json')['verified']
    result = read(root / 'functional-verification.json')
    assert result['safety_verified_for_retained_chain'] and result['full_native_reload_verified']
    assert result['classification'] in ('correct_recovered', 'safe_discard', 'no_fault_verified')
    return result


def physical_index(observed):
    physical = [e for e in observed if e['event'] == 'engine_generation_returned']
    by_request = {e['request_id']: e for e in physical}
    by_execution = {e['execution_id']: e for e in physical}
    assert len(by_request) == len(by_execution) == len(physical)
    scored = [e for e in observed if e['event'] == 'score_execution_returned']
    assert len({e['execution_id'] for e in scored}) == len(scored)
    return by_request, by_execution, scored


def parent_accepted_scores(root, scored):
    """A child return counts only when strict_reward's parent accepted its exit."""
    logs = list((root / 'areal/logs').rglob('trainer.log'))
    assert len(logs) == 1, ('trainer logs', len(logs))
    decoder = json.JSONDecoder()
    attempts = []
    for line in logs[0].read_text().splitlines():
        if 'scoring_attempt ' in line:
            attempts.append(decoder.raw_decode(line.split('scoring_attempt ', 1)[1])[0])
    by_pid = {record['pid']: record for record in attempts}
    assert len(by_pid) == len(attempts)
    returned = {event['pid']: event for event in scored}
    assert len(returned) == len(scored)
    accepted = {pid for pid, record in by_pid.items()
                if record['status'] == 'scored' and record['exit_code'] == 0}
    assert accepted <= set(returned)
    return [returned[pid] for pid in accepted]


def a_batch(batch):
    answer = []
    for part in batch['batches']:
        attempts = part['pilot_sample_attempt']
        assert len(attempts) == part['batch_size']
        for (fp, reward), attempt, source, task, slot in zip(
                batch_rows([part]), attempts, part['pilot_source_row_id'],
                part['pilot_task_id'], part['pilot_sample_idx']):
            answer.append((attempt, source, task, slot, fp, reward))
    assert len(answer) == 32 and len({r[0] for r in answer}) == 32
    return answer


def audit_a(root, pilot, observed, physical, scored, killed_id, collect=None):
    recorded = events(root, 'observer-pilot')
    generated = [e for e in recorded if e['event'] == 'generation_done']
    generation = {e['sample_attempt']: e for e in generated}
    assert len(generation) == len(generated)
    scores = {}
    for e in scored:
        if e.get('sample_attempt') is not None:
            assert e['sample_attempt'] not in scores
            scores[e['sample_attempt']] = e
    batches = [e for e in pilot if e['event'] == 'batch_taken']
    trains = [e for e in pilot if e['event'] == 'train_batch']
    assert len(batches) == len(trains)
    by_update = {}
    used_attempts = set()
    for batch, train in zip(batches, trains):
        assert batch['pid'] == train['pid'] and batch['monotonic_ns'] < train['monotonic_ns']
        original = a_batch(batch)
        assert Counter(r[0] for r in original) == Counter(r[0] for r in a_batch(train))
        for attempt, source, task, slot, fp, reward in original:
            used_attempts.add(attempt)
            gen = generation[attempt]
            actual = physical[gen['origin_request_id']]
            assert gen['generation_execution_id'] == actual['execution_id']
            assert (source, task, slot) == (gen['source_row_id'], gen['task_id'], gen['sample_idx'])
            assert (source, task, slot) == (actual['source_row_id'], actual['task_id'], actual['sample_idx'])
            n = len(actual['input_tokens'])
            assert fp == fingerprint(actual['input_tokens'] + actual['output_tokens'],
                                     [-1] * n + actual['output_versions'],
                                     [0] * n + [1] * len(actual['output_tokens']))
            assert reward == scores[attempt]['score']
        assert train['update_id'] not in by_update
        by_update[train['update_id']] = original
    retained = read(root / 'chain-verification.json')['retained_physical_update_ids']
    retained_attempts = {row[0] for update in retained for row in by_update[update]}
    assert len(retained_attempts) == 320
    unused_generated = len(set(generation) - used_attempts)
    if collect is not None:
        collect['retained_scores'] = {scores[a]['execution_id'] for a in retained_attempts}
        collect['update_scores'] = {u: {scores[r[0]]['execution_id'] for r in rows_} for u, rows_ in by_update.items()}
    if killed_id is None:
        return 0, 0, 320, unused_generated
    target = by_update[killed_id]
    return sum(row[0] in retained_attempts for row in target), sum(
        len(physical[generation[row[0]]['origin_request_id']]['output_tokens']) for row in target), len(retained_attempts), unused_generated


def audit_r(root, pilot, observed, physical, scored, killed_id, collect=None):
    method = root / 'rewardtxn'
    scores = {}
    for event in scored:
        nonce = event.get('reward_invocation_nonce')
        if nonce is not None:
            scores.setdefault(nonce, []).append(event)

    def blob(sha):
        path = method / 'artifacts/blobs' / sha
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        return json.loads(raw)

    def detail(sample):
        receipt = sample['receipt']; payload = receipt['payload']
        response = blob(payload['response_sha256'])
        reward = blob(payload['reward_sha256'])
        tensor = blob(payload['tensor_input_sha256'])
        assert response['binding']['attempt'] == reward['binding']['attempt'] == tensor['binding']['attempt'] == receipt['attempt']
        assert reward['payload']['response_sha256'] == payload['response_sha256']
        assert tensor['payload']['response_sha256'] == payload['response_sha256']
        assert tensor['payload']['reward_sha256'] == payload['reward_sha256']
        tensor_raw = {}
        for key, field in tensor['payload']['fields'].items():
            reference = field['raw']; raw = (method / 'artifacts/blobs' / reference['sha256']).read_bytes()
            assert len(raw) == reference['size'] and hashlib.sha256(raw).hexdigest() == reference['sha256']
            tensor_raw[key] = reference['sha256']
        answer = response['payload']; score = reward['payload']['return']
        origin = physical[answer['origin_rid']]
        assert (answer['input_tokens'], answer['output_tokens'], answer['output_versions']) == (
            origin['input_tokens'], origin['output_tokens'], origin['output_versions'])
        invocation = unique(scores[score['invocation_nonce']], 'physical score execution')
        assert invocation['score'] == score['score'] and invocation['execution_id']
        assert score['status'] == 'scored'
        return {'request_id': answer['origin_rid'], 'reward_nonce': score['invocation_nonce'],
                'score_execution_id': invocation['execution_id'],
                'score_monotonic_ns': invocation['monotonic_ns'],
                'generation_monotonic_ns': origin['monotonic_ns'],
                'receipt': receipt, 'tensor_raw': tensor_raw,
                'output_tokens': len(answer['output_tokens'])}

    def samples(record):
        return {sample['sample']: detail(sample) for update in record['updates']
                for group in update['groups'] for sample in group['samples']}

    generation_dir = method / 'state/generations'
    head = read(method / 'state/control.json')['head']; manifests = []; retained_generations = set()
    while head is not None:
        directory = generation_dir / head['generation']
        assert head['generation'] not in retained_generations
        retained_generations.add(head['generation'])
        manifests.append(read(directory / 'manifest.json'))
        head = read(directory / 'token.json')['parent']
    manifests.reverse()
    retained = {}
    for manifest in manifests:
        for sample, value in samples(manifest).items():
            assert sample not in retained
            retained[sample] = value
    assert len(retained) == 320
    if collect is not None:
        collect['retained_scores'] = {v['score_execution_id'] for v in retained.values()}
        applied = {e['generation']: int(e['monotonic_ns']) for e in rows(method / 'events.jsonl')
                   if e['event'] == 'optimizer_applied'}
        collect['generation_scores'] = {
            g: (t, {v['score_execution_id'] for v in samples(read(generation_dir / g / 'intent.json')).values()})
            for g, t in applied.items()}
    if killed_id is None:
        return 0, 0, len(retained)

    starts = [e for e in pilot if e['event'] == 'optimizer_start' and e['update_id'] == killed_id]
    ends = [e for e in pilot if e['event'] == 'optimizer_end' and e['update_id'] == killed_id]
    start, end = unique(starts, 'killed optimizer start'), unique(ends, 'killed optimizer end')
    applied = unique([e for e in rows(method / 'events.jsonl') if e['event'] == 'optimizer_applied'
                      and e['pid'] == start['pid'] and start['monotonic_ns'] < e['monotonic_ns'] < end['monotonic_ns']],
                     'killed R generation')
    intent = read(generation_dir / applied['generation'] / 'intent.json')
    assert len(intent['updates']) == 1
    assert not (generation_dir / applied['generation'] / 'token.json').exists()
    assert applied['generation'] not in retained_generations
    target = samples(intent)
    assert len(target) == 32
    assert all(value['generation_monotonic_ns'] < start['monotonic_ns']
               and value['score_monotonic_ns'] < start['monotonic_ns'] for value in target.values())
    adoptions = {}
    for path in (method / 'artifacts/samples').glob('*/adoption.json'):
        record = read(path)
        key = json.dumps(record['destination'], sort_keys=True)
        assert key not in adoptions
        adoptions[key] = record['origin']
    reused = sum(retained.get(sample, {}).get('request_id') == value['request_id']
                 and retained.get(sample, {}).get('reward_nonce') == value['reward_nonce']
                 and retained.get(sample, {}).get('score_execution_id') == value['score_execution_id']
                 and retained.get(sample, {}).get('tensor_raw') == value['tensor_raw']
                 and (retained[sample]['receipt'] == value['receipt']
                      or adoptions.get(json.dumps(retained[sample]['receipt']['attempt'], sort_keys=True)) == value['receipt'])
                 for sample, value in target.items())
    return reused, sum(v['output_tokens'] for v in target.values()), len(retained)


def verify(root):
    root = Path(root)
    case = read(root / 'ft1-case.json')
    assert case['scenario'] in ('F2', 'F4T', 'F1', 'no_fault') and case.get('minimal') is True
    if case['scenario'] in ('F4T', 'F1'):
        return verify_inflight(root, case)
    accepted = check_accepted(root)
    observed = events(root, 'observer-ft1')
    pilot = sorted(events(root, 'observer-pilot'), key=lambda e: e['monotonic_ns'])
    physical, _, scored = physical_index(observed)
    scored = parent_accepted_scores(root, scored)
    killed_id = None if case['scenario'] == 'no_fault' else read(root / 'fault-verification.json')['killed_uncheckpointed_update_id']
    if case['arm'] == 'A':
        reused, target_tokens, retained, unused_generated = audit_a(root, pilot, observed, physical, scored, killed_id)
        if case['scenario'] == 'F2' and accepted['classification'] == 'safe_discard':
            assert reused == 0
    else:
        reused, target_tokens, retained = audit_r(root, pilot, observed, physical, scored, killed_id)
        unused_generated = None
    sent = None if killed_id is None else unique([e for e in rows(root / 'events.jsonl') if e['kind'] == 'signal_sent'], 'signal')['controller_monotonic_ns']
    return {'verified': True, 'arm': case['arm'], 'scenario': case['scenario'],
            'target_rows': 0 if killed_id is None else 32, 'same_execution_source_rows': reused,
            'all_32_reused': reused == 32 if killed_id is not None else None,
            'retained_rows': retained, 'target_output_tokens': target_tokens,
            'unused_generated_attempts': unused_generated,
            'generated_tokens_after_fault': None if sent is None else sum(len(e['output_tokens']) for e in physical.values() if e['monotonic_ns'] > sent),
            'score_returns_after_fault': None if sent is None else sum(e['monotonic_ns'] > sent for e in scored),
            'scope': 'actual engine call return and score execution ID to retained native chain; one run, no statistical claim'}


def verify_inflight(root, case):
    """F4'/F1 main metric (FT_F4F1_PILOT_PLAN section 8): completed work at the
    signal = parent-accepted score executions returned before the signal whose
    sample was not in any update already applied before it; reused = those whose
    same physical score execution enters the final retained chain."""
    check_accepted(root)
    observed = events(root, 'observer-ft1')
    pilot = sorted(events(root, 'observer-pilot'), key=lambda e: e['monotonic_ns'])
    physical, _, scored = physical_index(observed)
    scored = parent_accepted_scores(root, scored)
    sent = unique([e for e in rows(root / 'events.jsonl') if e['kind'] == 'signal_sent'], 'signal')['controller_monotonic_ns']
    collect = {}
    if case['arm'] == 'A':
        audit_a(root, pilot, observed, physical, scored, None, collect)
        ends = {e['update_id']: e['monotonic_ns'] for e in pilot if e['event'] == 'optimizer_end'}
        trained = set().union(*(ids for u, ids in collect['update_scores'].items() if ends.get(u, 1 << 62) < sent))
    else:
        audit_r(root, pilot, observed, physical, scored, None, collect)
        trained = set().union(*(ids for t, ids in collect['generation_scores'].values() if t < sent))
    completed = {e['execution_id'] for e in scored if e['monotonic_ns'] < sent} - trained
    reused = completed & collect['retained_scores']
    return {'verified': True, 'arm': case['arm'], 'scenario': case['scenario'],
            'completed_untrained_scores_at_signal': len(completed),
            'same_execution_reused': len(reused), 'discarded': len(completed) - len(reused),
            'generated_tokens_after_fault': sum(len(e['output_tokens']) for e in physical.values() if e['monotonic_ns'] > sent),
            'score_returns_after_fault': sum(e['monotonic_ns'] > sent for e in scored),
            'scope': 'physical score execution identity to retained native chain; one run, no statistical claim'}


if __name__ == '__main__':
    root = Path(sys.argv[1]); report = verify(root)
    (root / 'source-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
