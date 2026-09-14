#!/usr/bin/env python3
"""Read-only pilot trajectory audit; stdlib only, no model loading or training.

Checks every saved debug batch against the summaries, not just aggregate gates.
Only plain-data pickle archives are accepted; class loading is forbidden.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class PlainData(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError(f'Non-plain pickle: {module}.{name}')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(values):
    return statistics.mean(values)


def audit_run(run, source, train_ids):
    summary = read(run / 'diagnosis_summary.json')
    assert summary['train_steps'] == list(range(500))
    batches = summary['consumed_rollouts']
    assert [b['rollout_id'] for b in batches] == list(range(500))
    assert {p.stem for p in (run / 'rollout_debug').glob('*.pt')} == {str(i) for i in range(500)}
    hashes, samples, groups = [], {}, set()
    for batch in batches:
        file = run / 'rollout_debug' / f'{batch["rollout_id"]}.pt'
        hashes.append(digest(file))
        with zipfile.ZipFile(file) as archive:
            name, = [n for n in archive.namelist() if n.endswith('/data.pkl')]
            raw = PlainData(io.BytesIO(archive.read(name))).load()
        assert raw['rollout_id'] == batch['rollout_id']
        assert len(raw['samples']) == len(batch['samples']) == 32
        assert Counter(s['group_index'] for s in raw['samples']) == Counter({int(k): v for k, v in batch['groups'].items()})
        assert len(batch['groups']) == 4 and set(batch['groups'].values()) == {8}
        for s, saved in zip(raw['samples'], batch['samples']):
            sid = saved['source_index']
            assert sid in train_ids
            row = source[sid]
            prompt = s['prompt']
            assert all(m['content'] in (prompt if isinstance(prompt, str) else [p['content'] for p in prompt]) for m in row['prompt'])
            assert s['label'] == row.get('label', row.get('answer', ''))
            for key, raw_key in [('sample_index', 'index'), ('group_index', 'group_index'),
                                 ('reward', 'reward'), ('status', 'status'),
                                 ('weight_versions', 'weight_versions'), ('response_length', 'response_length')]:
                assert saved[key] == s[raw_key], (file, key)
            assert saved['response_sha256'] == hashlib.sha256(s['response'].encode()).hexdigest()
            assert saved['token_count'] == len(s['tokens'])
            assert saved['logprob_count'] == s['response_length'] == len(s['rollout_log_probs'])
            assert all(math.isfinite(x) for x in s['rollout_log_probs'])
            assert saved['logprobs_finite']
            assert s['index'] not in samples, ('duplicate consumed index', s['index'])
            samples[s['index']] = saved
            groups.add(s['group_index'])
    assert len(samples) == 16000 and len(groups) == 2000
    metrics = summary['metrics']
    windows = {}
    for lo, hi in [(0, 50), (50, 250), (250, 500)]:
        windows[f'{lo}:{hi}'] = {key: average([m[key] for m in metrics if key in m and lo <= m['step'] < hi])
            for key in ['rollout/raw_reward', 'rollout/response_lengths', 'rollout/truncated',
                        'train/pg_clipfrac', 'train/grad_norm', 'train/train_rollout_logprob_abs_diff']}
    result = {
        'raw_batches_verified': len(batches), 'consumed_samples': len(samples), 'unique_consumed_groups': len(groups),
        'raw_batch_hash_chain_sha256': hashlib.sha256(''.join(hashes).encode()).hexdigest(),
        'summary_sha256': digest(run / 'diagnosis_summary.json'),
        'first_groups': list(batches[0]['groups']),
        'reward_mean': average([s['reward'] for s in samples.values()]),
        'response_length_mean': average([s['response_length'] for s in samples.values()]),
        'statuses': dict(Counter(s['status'] for s in samples.values())),
        'weight_version_count_distribution': dict(Counter(len(s['weight_versions']) for s in samples.values())),
        'windows': windows,
    }
    return result, batches, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'runs/e7_restart_0.5B_20260911_pilot.json')
    args = parser.parse_args()
    manifest = read(args.manifest)
    train = read(ROOT / 'runs/diagnosis-20260910/train_split.json')
    source_path = ROOT / train['source']
    assert digest(source_path) == train['source_sha256']
    with source_path.open() as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    runs, evidence = {}, {}
    for relative in manifest['runs']:
        run = ROOT / relative
        record, batches, samples = audit_run(run, source, set(train['eval_indices']))
        meta = read(run / 'meta.json')
        runs[relative] = record
        evidence[(meta['seed'], meta['baseline_mode'])] = (batches, samples)
    pairs = {}
    for seed in sorted({s for s, g in evidence}):
        a, sa = evidence[seed, 'group_rm']
        b, sb = evidence[seed, 'b6']
        common = sa.keys() & sb.keys()
        assert all(sa[i]['source_index'] == sb[i]['source_index'] for i in common)
        group_overlap = [len(set(x['groups']) & set(y['groups'])) for x, y in zip(a, b)]
        pairs[seed] = {
            'first_divergent_group_batch': next((i for i, n in enumerate(group_overlap) if n != 4), None),
            'identical_group_set_batches': sum(n == 4 for n in group_overlap),
            'same_step_shared_group_fraction': sum(group_overlap) / 2000,
            'overall_shared_groups': len({s['group_index'] for s in sa.values()} & {s['group_index'] for s in sb.values()}),
            'overall_shared_samples': len(common),
            'same_response_on_shared_sample_id': sum(sa[i]['response_sha256'] == sb[i]['response_sha256'] for i in common),
            'first_batch_common_samples': [i for i in [s['sample_index'] for s in a[0]['samples']]
                                          if i in {s['sample_index'] for s in b[0]['samples']}],
        }
    print(json.dumps({'manifest_sha256': digest(args.manifest), 'runs': runs, 'pairs': pairs,
                      'limitations': 'No new generation. No causal estimate separating queue loss, sampling randomness and method overhead.'}, indent=2))


if __name__ == '__main__':
    main()
