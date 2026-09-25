"""Directed CPU checks for source identity and the post-load cleanup gate."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from check_ft_minimal_source import audit_a, audit_r, parent_accepted_scores, physical_index
from minimal_storage import retain_or_clear
from run_ft_minimal import minimal_config
from run_ft1_acceptance import accept


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def test_fixed_pilot_config():
    base = (Path(__file__).resolve().parents[2] /
            'docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml').read_text()
    config = minimal_config(base, 'no_fault', 419)
    assert 'total_train_steps: 10' in config
    assert 'seed: 419' in config
    assert 'retries: 0' in config
    assert 'retries: 1' in minimal_config(base, 'F2', 419)
    with pytest.raises(ValueError, match='drift'):
        minimal_config(base.replace('seed: 211', 'seed: 17'), 'F2', 419)


def test_parent_rejects_child_return_before_retry(tmp_path):
    log = tmp_path / 'areal/logs/experiment/trainer.log'
    log.parent.mkdir(parents=True)
    records = [
        {'pid': 2702, 'status': 'error', 'exit_code': -15},
        {'pid': 2704, 'status': 'scored', 'exit_code': 0},
    ]
    log.write_text(''.join('StrictReward INFO: scoring_attempt ' + json.dumps(r) + '\n'
                           for r in records))
    returns = [{'pid': 2702, 'sample_attempt': 7, 'execution_id': 'unaccepted', 'score': 0},
               {'pid': 2704, 'sample_attempt': 7, 'execution_id': 'accepted', 'score': 0},
               {'pid': 2706, 'sample_attempt': 8, 'execution_id': 'no-parent-record', 'score': 0}]
    assert parent_accepted_scores(tmp_path, returns) == [returns[1]]
    with pytest.raises(AssertionError):
        parent_accepted_scores(tmp_path, returns[:1])


def test_same_text_regeneration_is_not_reuse(tmp_path):
    origin = tmp_path / 'observer-pilot' / 'events.jsonl'
    origin.parent.mkdir()
    observations = []
    physical = []
    score = []
    pilot = []
    for update in range(11):
        attempts = list(range(update * 32, (update + 1) * 32))
        part = {'batch_size': 32, 'pilot_sample_attempt': attempts,
                'pilot_source_row_id': [update * 4 + i // 8 for i in range(32)],
                'pilot_task_id': [update * 4 + i // 8 for i in range(32)],
                'pilot_sample_idx': [i % 8 for i in range(32)],
                'input_ids': [[11, 22] for _ in range(32)],
                'versions': [[-1, 0] for _ in range(32)],
                'loss_mask': [[0, 1] for _ in range(32)],
                'attention_mask': [[1, 1] for _ in range(32)],
                'rewards': [1.0] * 32}
        pilot += [{'event': 'batch_taken', 'pid': 1, 'monotonic_ns': update * 2,
                   'batches': [part]},
                  {'event': 'train_batch', 'pid': 1, 'monotonic_ns': update * 2 + 1,
                   'update_id': str(update), 'batches': [part]}]
        for slot, attempt in enumerate(attempts):
            source = update * 4 + slot // 8
            request = f'request-{attempt}'
            execution = f'execution-{attempt}'
            observations += [
                {'event': 'generation_done', 'sample_attempt': attempt,
                 'source_row_id': source, 'task_id': source, 'sample_idx': slot % 8,
                 'origin_request_id': request, 'generation_execution_id': execution},
                {'event': 'reward_done', 'sample_attempt': attempt, 'reward': 1.0}]
            physical.append({'event': 'engine_generation_returned', 'request_id': request,
                             'execution_id': execution, 'source_row_id': source,
                             'task_id': source, 'sample_idx': slot % 8,
                             'input_tokens': [11], 'output_tokens': [22],
                             'output_versions': [0]})
            score.append({'event': 'score_execution_returned',
                          'execution_id': f'score-{attempt}',
                          'sample_attempt': attempt, 'score': 1.0})
    origin.write_text(''.join(json.dumps(item) + '\n' for item in observations))
    put(tmp_path / 'chain-verification.json',
        {'retained_physical_update_ids': [str(i) for i in range(1, 11)]})
    by_request, _, scored = physical_index(physical + score)
    reused, tokens, retained, unused = audit_a(tmp_path, pilot, physical + score,
                                               by_request, scored, '0')
    assert (reused, tokens, retained, unused) == (0, 32, 320, 0)
    observations.append({'event': 'generation_done', 'sample_attempt': 9999})
    origin.write_text(''.join(json.dumps(item) + '\n' for item in observations))
    assert audit_a(tmp_path, pilot, physical + score, by_request, scored, '0') == (0, 32, 320, 1)
    # Same tokens with a forged execution ID must fail the origin join.
    observations[0]['generation_execution_id'] = 'other-execution'
    origin.write_text(''.join(json.dumps(item) + '\n' for item in observations))
    with pytest.raises(AssertionError):
        audit_a(tmp_path, pilot, physical + score, by_request, scored, '0')


def test_r_same_text_new_request_is_not_reuse(tmp_path):
    method = tmp_path / 'rewardtxn'
    blobs = method / 'artifacts/blobs'
    blobs.mkdir(parents=True)

    def blob(value):
        raw = json.dumps(value, sort_keys=True).encode()
        sha = hashlib.sha256(raw).hexdigest()
        (blobs / sha).write_bytes(raw)
        return sha

    physical = {}
    scored = []

    def sample(index, execution):
        name = f'sample-{index}'
        attempt = {'sample': name, 'execution': execution}
        request = f'request-{index}-{execution}'
        nonce = f'nonce-{index}-{execution}'
        physical[request] = {'request_id': request, 'input_tokens': [11],
                             'output_tokens': [22], 'output_versions': [0], 'monotonic_ns': 0}
        scored.append({'event': 'score_execution_returned', 'execution_id': f'score-{index}-{execution}',
                       'monotonic_ns': 0, 'reward_invocation_nonce': nonce, 'score': 1.0})
        binding = {'attempt': attempt}
        response = blob({'binding': binding, 'payload': {'origin_rid': request,
                         'input_tokens': [11], 'output_tokens': [22], 'output_versions': [0]}})
        reward = blob({'binding': binding, 'payload': {'response_sha256': response,
                       'return': {'invocation_nonce': nonce, 'score': 1.0, 'status': 'scored'}}})
        raw = f'raw-{index}-{execution}'.encode()
        raw_sha = hashlib.sha256(raw).hexdigest()
        (blobs / raw_sha).write_bytes(raw)
        tensor = blob({'binding': binding, 'payload': {'response_sha256': response,
                       'reward_sha256': reward, 'fields': {'input_ids': {
                           'raw': {'sha256': raw_sha, 'size': len(raw)}}}}})
        return {'sample': name, 'receipt': {'attempt': attempt,
                'payload': {'response_sha256': response, 'reward_sha256': reward,
                            'tensor_input_sha256': tensor}}}

    retained = [sample(i, 'original') for i in range(320)]
    redo = [sample(i, 'redo') for i in range(32)]
    generation = method / 'state/generations'
    put(method / 'state/control.json', {'head': {'generation': 'committed'}})
    put(generation / 'committed/manifest.json',
        {'updates': [{'groups': [{'samples': retained}]}]})
    put(generation / 'committed/token.json', {'parent': None})
    put(generation / 'abandoned/intent.json',
        {'updates': [{'groups': [{'samples': redo}]}]})
    (method / 'events.jsonl').write_text(json.dumps({
        'event': 'optimizer_applied', 'pid': 7, 'monotonic_ns': 2,
        'generation': 'abandoned'}) + '\n')
    pilot = [{'event': 'optimizer_start', 'pid': 7, 'monotonic_ns': 1, 'update_id': 'killed'},
             {'event': 'optimizer_end', 'pid': 7, 'monotonic_ns': 3, 'update_id': 'killed'}]
    assert audit_r(tmp_path, pilot, [], physical, scored, 'killed') == (0, 32, 320)
    put(generation / 'abandoned/intent.json',
        {'updates': [{'groups': [{'samples': retained[:32]}]}]})
    adopted = []
    for index, old in enumerate(retained[:32]):
        previous = old['receipt']
        attempt = {'sample': old['sample'], 'execution': 'adopted'}
        response = json.loads((blobs / previous['payload']['response_sha256']).read_text())
        response['binding']['attempt'] = attempt
        response['binding']['origin_attempt'] = previous['attempt']
        response_sha = blob(response)
        reward = json.loads((blobs / previous['payload']['reward_sha256']).read_text())
        reward['binding'] = response['binding']
        reward['payload']['response_sha256'] = response_sha
        reward_sha = blob(reward)
        tensor = json.loads((blobs / previous['payload']['tensor_input_sha256']).read_text())
        tensor['binding'] = response['binding']
        tensor['payload']['response_sha256'] = response_sha
        tensor['payload']['reward_sha256'] = reward_sha
        tensor_sha = blob(tensor)
        adopted.append({'sample': old['sample'], 'receipt': {'attempt': attempt, 'payload': {
            'response_sha256': response_sha, 'reward_sha256': reward_sha,
            'tensor_input_sha256': tensor_sha}}})
        put(method / f'artifacts/samples/adopted-{index}/adoption.json',
            {'origin': previous, 'destination': attempt})
    put(generation / 'committed/manifest.json',
        {'updates': [{'groups': [{'samples': adopted + retained[32:]}]}]})
    assert audit_r(tmp_path, pilot, [], physical, scored, 'killed') == (32, 32, 320)
    adoption = method / 'artifacts/samples/adopted-0/adoption.json'
    proof = adoption.read_text(); adoption.unlink()
    assert audit_r(tmp_path, pilot, [], physical, scored, 'killed') == (31, 32, 320)
    adoption.write_text(proof)
    scored.append({'event': 'score_execution_returned', 'execution_id': 'rescored-after-fault',
                   'monotonic_ns': 4, 'reward_invocation_nonce': 'nonce-0-original', 'score': 1.0})
    with pytest.raises(AssertionError):
        audit_r(tmp_path, pilot, [], physical, scored, 'killed')


def test_cleanup_requires_loaded_and_source_verified():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / 'minimal_evidence' / 'pilot-a'
        shard = root / 'areal/checkpoints/step0/shard.distcp'
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b'checkpoint')
        put(root / 'ft1-case.json', {'minimal': True, 'arm': 'A', 'scenario': 'no_fault'})
        put(root / 'acceptance-status.json', {'result': 'functional_verification_written'})
        put(root / 'functional-verification.json',
            {'classification': 'no_fault_verified', 'full_native_reload_verified': False,
             'safety_verified_for_retained_chain': True})
        put(root / 'source-verification.json', {'verified': True})
        with pytest.raises(AssertionError):
            retain_or_clear(root)
        assert shard.exists()
        result = json.loads((root / 'functional-verification.json').read_text())
        result['full_native_reload_verified'] = True
        put(root / 'functional-verification.json', result)
        report = retain_or_clear(root)
        assert report['status'] == 'shards_deleted'
        assert report['shards'][0]['bytes'] == len(b'checkpoint')
        assert not shard.exists()
        assert (root / 'storage-cleanup.json').exists()


def test_no_fault_acceptance_requires_input_and_reload(tmp_path):
    root = tmp_path / 'minimal_evidence' / 'pilot-a'
    put(root / 'ft1-case.json', {'minimal': True, 'arm': 'A', 'scenario': 'no_fault'})
    put(root / 'final-native-state.json', {'model': 'present'})

    def host(script, *args):
        assert script == 'tests/ft/check_ft1_chain.py'
        put(Path(args[1]), {'verified': True})
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    def docker(kind, evidence, out, arm, gpu):
        assert kind in ('input', 'load')
        return {'kind': kind, 'exitcode': 0, 'output': str(out)}

    with patch('check_ft1_smoke.verify', return_value={'smoke_passed': True}), \
         patch('run_ft1_acceptance.run_host', side_effect=host), \
         patch('run_ft1_acceptance.docker_verify', side_effect=docker):
        status = accept(root, gpu_uuid='GPU-fixture')
    assert status['result'] == 'functional_verification_written'
    result = json.loads((root / 'functional-verification.json').read_text())
    assert result['classification'] == 'no_fault_verified'
    assert result['full_native_reload_verified']
