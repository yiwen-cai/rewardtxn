#!/usr/bin/env python3
"""Read-only data, result, pilot and freeze gates for the versioned E7 restart."""
import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from paper_statistics import paired_tost, plan_paired_tost_sample_size

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'prereg/restarts/0.5B-20260911/protocol.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_data(spec):
    splits = [read(ROOT / spec[k]) for k in ['train_split', 'validation_split', 'test_split', 'historical_split']]
    source = ROOT / splits[0]['source']
    with source.open() as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    seen, prompts = set(), set()
    for split in splits:
        require(split['source'] == splits[0]['source'] and split['source_sha256'] == sha(source), 'source hash/path mismatch')
        ids = split['eval_indices']
        require(len(ids) == len(set(ids)) == split['eval_count'], 'duplicate IDs/count mismatch')
        require(not seen.intersection(ids), 'split index overlap')
        content = {json.dumps(rows[i]['prompt'], sort_keys=True) for i in ids}
        require(not prompts.intersection(content), 'split prompt overlap')
        seen.update(ids)
        prompts.update(content)
    with (ROOT / spec['train_file']).open() as stream:
        train = [json.loads(line) for line in stream if line.strip()]
    require(train == [rows[i] for i in splits[0]['eval_indices']], 'actual training rows differ from split')
    require([s['eval_count'] for s in splits] == [6373, 100, 500, 500], 'unexpected split sizes')
    return {'train_count': len(train), 'train_sha256': sha(ROOT / spec['train_file'])}


def check_eval(path, split_path, checkpoint=None):
    d, split = read(path), read(split_path)
    require(d['split_sha256'] == sha(split_path), 'evaluation split hash mismatch')
    require(d['n_total'] == split['eval_count'] == len(d['results']), 'evaluation count mismatch')
    require([x['source_index'] for x in d['results']] == split['eval_indices'], 'evaluation ID/order mismatch')
    require(d['max_new_tokens'] == 2048 and d['batch_size'] == 8, 'evaluation generation settings mismatch')
    if checkpoint:
        require(d['checkpoint'] == checkpoint, 'wrong fixed endpoint checkpoint')
    require(all(type(x['correct']) is bool and type(x['truncated']) is bool for x in d['results']), 'invalid evaluation flags')
    accuracy = sum(x['correct'] for x in d['results']) / d['n_total']
    truncated = sum(x['truncated'] for x in d['results']) / d['n_total']
    require(d['accuracy'] == accuracy and d['n_correct'] == sum(x['correct'] for x in d['results']), 'incorrect accuracy summary')
    require(d['truncated_fraction'] == truncated, 'incorrect truncation summary')
    return d


def check_run(run, spec, stage):
    run = Path(run).resolve()
    meta = read(run / 'meta.json')
    params = meta['params']
    require(params.get('restart_protocol_sha256') == sha(SPEC), 'run protocol hash mismatch')
    require(meta['seed'] in spec[stage + '_seeds'], 'seed not in planned stage')
    require(meta['baseline_mode'] in spec['implemented_clean_groups'], 'unexpected group')
    require(params['lr'] == spec['lr'] and params['use_rollout_logprobs'] is True, 'optimization config mismatch')
    require(params['model'] == spec['model'] and params['num_rollout'] == spec['steps'], 'model/steps mismatch')
    require(params['train_entry'] == spec['train_entry'] and params['fully_async_rollout'], 'training loop mismatch')
    require(params['dataset_path'] == '/workspace/' + spec['train_file'], 'training data path mismatch')
    require(meta['paper_mode'] == (stage == 'formal'), 'paper mode/stage mismatch')
    terminal = read(run / 'logs/terminal_state.json')
    require(terminal['Status'] == 'exited' and terminal['ExitCode'] == 0 and not terminal['OOMKilled'], 'training failed')
    audit = read(run / 'diagnosis_summary.json')
    require(audit['train_steps'] == list(range(spec['steps'])) and audit['all_logged_metrics_finite'], 'training step/finite audit failed')
    require(audit['consumed_split_overlap'] == 0 and len(audit['consumed_rollouts']) == spec['steps'], 'consumption audit failed')
    checkpoint = '/workspace/' + str(run.relative_to(ROOT)) + '/checkpoints/iter_0000499_hf'
    validation = check_eval(run / 'restart_eval/validation.json', ROOT / spec['validation_split'], checkpoint)
    base = check_eval(ROOT / spec['base_validation'], ROOT / spec['validation_split'])
    quality = (validation['accuracy'] >= base['accuracy'] - spec['quality_accuracy_drop_pp'] / 100 - 1e-12
               and validation['truncated_fraction'] <= base['truncated_fraction'] + spec['quality_truncated_increase'] + 1e-12)
    result = validation if stage == 'pilot' else check_eval(run / 'restart_eval/test.json', ROOT / spec['test_split'], checkpoint)
    return {'seed': meta['seed'], 'group': meta['baseline_mode'], 'accuracy_pp': result['accuracy'] * 100,
            'quality_pass': quality, 'run': str(run.relative_to(ROOT))}


def paired_report(records, seeds, spec, pilot=False):
    pairs = {}
    for r in records:
        key = (r['seed'], r['group'])
        require(key not in pairs, 'duplicate seed/group')
        pairs[key] = r
    require(set(pairs) == {(s, g) for s in seeds for g in ['group_rm', 'b6']}, 'missing/unplanned seed/group')
    ref = [pairs[s, 'group_rm']['accuracy_pp'] for s in seeds]
    trt = [pairs[s, 'b6']['accuracy_pp'] for s in seeds]
    require(all(math.isfinite(x) and 0 <= x <= 100 for x in ref + trt), 'invalid accuracies')
    delta = [b - a for a, b in zip(ref, trt)]
    quality = all(r['quality_pass'] for r in records)
    nondegenerate = statistics.stdev(delta) > 1e-12
    result = {'analysis_unit': 'seed', 'effect': 'RewardTxn-Oracle', 'unit': 'pp', 'seeds': seeds,
              'differences_pp': delta, 'quality_pass': quality, 'nondegenerate': nondegenerate,
              'equivalence_claim_enabled': False}
    if pilot:
        if nondegenerate:
            result['planning'] = plan_paired_tost_sample_size(delta, spec['margin_pp'], candidates=[len(spec['formal_seeds'])],
                                                            alpha=spec['alpha'], target_power=spec['target_power'])
            result['equivalence_claim_enabled'] = quality and result['planning']['candidates'][0]['acceptable']
    elif nondegenerate:
        result['tost'] = paired_tost(ref, trt, spec['margin_pp'], spec['alpha'])
    else:
        result['reason'] = 'zero paired variance: no inferential equivalence claim'
    return result


def manifest_report(path, spec, stage):
    check_data(spec)
    m = read(path)
    require(m['stage'] == stage and m['protocol_sha256'] == sha(SPEC), 'manifest stage/protocol mismatch')
    require(len(set(m['runs'])) == len(m['runs']), 'duplicate run paths')
    records = [check_run(ROOT / r, spec, stage) for r in m['runs']]
    result = paired_report(records, spec[stage + '_seeds'], spec, pilot=stage == 'pilot')
    result['records'] = records
    if stage == 'formal':
        verify_frozen(spec)
        pilot = manifest_report(ROOT / spec['pilot_manifest'], spec, 'pilot')
        require(not set(m['runs']).intersection(read(ROOT / spec['pilot_manifest'])['runs']), 'pilot reused in formal analysis')
        result['equivalence_claim_enabled'] = bool(pilot['equivalence_claim_enabled'] and result['quality_pass']
                                                  and result.get('tost', {}).get('equivalent', False))
    return result


def freeze_assets(spec):
    assets = [SPEC, ROOT / spec['pilot_manifest']]
    assets += [ROOT / spec[k] for k in ['train_file', 'train_split', 'validation_split', 'test_split', 'historical_split', 'base_validation']]
    assets += [ROOT / 'models/datasets/gsm8k/dapo-gsm8k-train.jsonl',
               ROOT / ('models/' + spec['model']), ROOT / ('models/' + spec['model'] + '_torch_dist'),
               ROOT / 'prereg/fault_schedules/e7-restart-clean-empty.json']
    for r in read(ROOT / spec['pilot_manifest'])['runs']:
        assets += [ROOT / r / f for f in ['meta.json', 'diagnosis_summary.json', 'logs/terminal_state.json', 'restart_eval/validation.json']]
    return assets


def freeze(spec):
    pilot = manifest_report(ROOT / spec['pilot_manifest'], spec, 'pilot')
    require(pilot['quality_pass'], 'pilot quality failed')
    output = ROOT / spec['freeze_manifest']
    require(not output.exists(), 'freeze output already exists; do not overwrite')
    args = ['python3', str(ROOT / 'scripts/freeze_paper_inputs.py'), 'generate', '--out', str(output),
            '--image', 'slimerl/slime:v0.3.1', '--repo', 'slime=' + str(ROOT / 'third_party/slime')]
    for i, path in enumerate(freeze_assets(spec)):
        args += ['--asset', f'restart_{i}={path}']
    subprocess.run(args, check=True)
    require(read(output)['frozen'] is True, 'inputs not frozen: inspect repository cleanliness and manifest errors')
    return {'frozen': True, 'equivalence_claim_enabled': pilot['equivalence_claim_enabled']}


def verify_frozen(spec):
    args = ['python3', str(ROOT / 'scripts/freeze_paper_inputs.py'), 'verify',
            '--manifest', str(ROOT / spec['freeze_manifest']), '--require-image', 'slimerl/slime:v0.3.1',
            '--require-repo-path', 'slime=' + str(ROOT / 'third_party/slime')]
    for asset in freeze_assets(spec):
        args += ['--require-asset-path', str(asset)]
    subprocess.run(args, check=True)


def preflight(spec, stage, seed):
    check_data(spec)
    require(seed in spec[stage + '_seeds'], 'seed not in planned stage')
    expected = {'RTX_LR': str(spec['lr']), 'RTX_MODEL_DIR': '/root/models/' + spec['model'],
                'RTX_DATA_PATH': '/workspace/' + spec['train_file'], 'RTX_NUM_ROLLOUT': str(spec['steps']),
                'RTX_SAVE_HF': '1', 'RTX_SAVE_INTERVAL': str(spec['save_interval']),
                'RTX_TRAIN_ENTRY': spec['train_entry'], 'RTX_FULLY_ASYNC': '1',
                'RTX_PAPER_MODE': '1' if stage == 'formal' else '0'}
    for key, value in expected.items():
        actual = os.environ.get(key)
        require(float(actual) == spec['lr'] if key == 'RTX_LR' and actual else actual == value, 'configuration mismatch: ' + key)
    require('--use-rollout-logprobs' in os.environ.get('RTX_EXTRA_MODEL_ARGS', '').split(), 'rollout reference disabled')
    require('--load' not in os.environ.get('RTX_EXTRA_MODEL_ARGS', '').split(), 'restart must start from base')
    if stage == 'formal':
        pilot = manifest_report(ROOT / spec['pilot_manifest'], spec, 'pilot')
        require(pilot['quality_pass'], 'pilot quality failed')
        verify_frozen(spec)
        print(json.dumps({'pilot_equivalence_claim_enabled': pilot['equivalence_claim_enabled']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['data', 'preflight', 'run', 'aggregate', 'freeze'])
    p.add_argument('--stage', choices=['pilot', 'formal'], default='formal')
    p.add_argument('--seed', type=int)
    p.add_argument('--run', type=Path)
    p.add_argument('--manifest', type=Path)
    a = p.parse_args()
    spec = read(SPEC)
    if a.command == 'data':
        result = check_data(spec)
    elif a.command == 'freeze':
        result = freeze(spec)
    elif a.command == 'preflight':
        preflight(spec, a.stage, a.seed)
        result = {'preflight': 'PASS', 'stage': a.stage}
    elif a.command == 'run':
        result = check_run(a.run, spec, a.stage)
        require(result['quality_pass'], 'fixed-endpoint validation quality failed')
    else:
        result = manifest_report(a.manifest, spec, a.stage)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print('E7 gate refused: ' + str(exc), file=sys.stderr)
        sys.exit(2)
