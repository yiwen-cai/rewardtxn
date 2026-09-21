"""Offline smoke checks; not a fault-recovery or reward-authority oracle."""
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def natural_f4(events):
    groups = defaultdict(list)
    for event in events:
        if event['event'] in ('generation_complete', 'score_execution_started'):
            groups[(event['source_row_id'], event['task_id'])].append(event)
    candidates = []
    for (source, task), group in groups.items():
        generated = set()
        started = []
        for event in sorted(group, key=lambda e: e['monotonic_ns']):
            if event['event'] == 'generation_complete':
                generated.add(event['sample_idx'])
            else:
                started.append(event['sample_idx'])
                if len(started) == 4 and len(set(started)) == 4 and generated == set(range(8)):
                    candidates.append({'source_row_id': source, 'task_id': task,
                                       'fourth_execution': event})
    return candidates


def verify(root):
    root = Path(root)
    assert (root/'exitcode').read_text().strip() == '0'
    final = read(root/'inspect-final.json')[0]
    cleanup = read(root/'cleanup.json')
    assert final['State']['ExitCode'] == 0 and final['Id'] == cleanup['id']
    assert cleanup['returncode'] == 1 and 'No such object' in cleanup['stderr']
    network = read(root/'network-cleanup.json')
    assert network['returncode'] == 1 and 'not found' in network['stderr']
    journal = rows(root/'events.jsonl')
    assert [e['classification'] for e in journal if e['kind'] == 'result'] == ['execution_complete']
    assert not any(e['kind'] == 'signal_sent' for e in journal)
    events = [e for p in (root/'observer-ft1').glob('*.jsonl') for e in rows(p)]
    configs = [e for e in events if e['event'] == 'configured']
    assert len(configs) == 1
    cfg = configs[0]['config']
    assert cfg['total_train_steps'] == 10 and cfg['recover']['retries'] == 0
    assert cfg['actor']['megatron']['async_save'] is True
    assert len([e for e in events if e['event'] == 'training_returned']) == 1
    pilot = [e for p in (root/'observer-pilot').glob('*.jsonl') for e in rows(p)]
    updates = [e for e in pilot if e['event'] == 'optimizer_end']
    assert len(updates) == 10 and len({e['update_id'] for e in updates}) == 10
    assert all(e['stats']['update_successful'] == 1.0 for e in updates)
    assert set(read(root/'final-native-state.json')) == {'model', 'optimizer', 'lr_scheduler', 'rng_state'}
    jobs = [read(p) for p in (root/'areal').rglob('job-lifecycle/*.json')]
    assert jobs and all(j['cleanup']['empty'] for j in jobs)
    trainers = [j for j in jobs if j['log_path'].endswith('/trainer.log')]
    assert len(trainers) == 1 and trainers[0]['root_exit_code'] == 0
    arm = configs[0]['arm']
    if arm == 'R':
        commits = [e for e in rows(root/'rewardtxn/events.jsonl') if e['event'] == 'committed']
        assert [e['global_step'] for e in commits] == list(range(10))
    else:
        checkpoint = [e for e in events if e['event'] == 'final_checkpoint']
        assert len(checkpoint) == 1 and checkpoint[0]['files'] and checkpoint[0]['recover_files']
    return {'scope': 'no-fault smoke only; retained-chain/reload oracle still required',
            'smoke_passed': True, 'arm': arm, 'optimizer_updates': len(updates),
            'natural_f4_candidates': natural_f4(events), 'fault_validation_passed': False}
