"""Read-only independent verification of a completed three-update GPU probe."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    assert (root / 'exitcode').read_text().strip() == '0', 'container failed'
    exits = {}
    for phase in ('first', 'resume'):
        # Preserve local.py's COMPLETED -> JobException return independently.
        assert read(root / (phase + '-exit.json'))['exit_code'] in (0, 1)
        exits[phase] = read(root / (phase + '-trainer-exit.json'))
        assert exits[phase]['exit_code'] == 0
        assert exits[phase]['supervisor_pid'] != exits[phase]['trainer_pid']
    final = read(root / 'inspect-final.json')[0]
    cleanup = read(root / 'cleanup.json')
    assert final['State']['ExitCode'] == 0 and not final['State']['Running']
    assert final['Id'] == cleanup['id'] and cleanup['returncode'] == 1 and 'No such object' in cleanup['stderr']
    method = root / 'rewardtxn'
    events = [json.loads(line) for line in (method / 'events.jsonl').read_text().splitlines()]
    commits = [e for e in events if e['event'] == 'committed']
    assert [e['global_step'] for e in commits] == [0, 1, 2]
    assert commits[0]['pid'] != commits[1]['pid'] == commits[2]['pid']
    assert commits[0]['pid'] == exits['first']['trainer_pid']
    assert commits[1]['pid'] == exits['resume']['trainer_pid']
    loaded = [e for e in events if e['event'] == 'native_state_loaded']
    assert len(loaded) == 1 and loaded[0]['exact_match'] is True
    assert loaded[0]['generation'] == commits[0]['generation']
    assert loaded[0]['pid'] == exits['resume']['trainer_pid']
    assert commits[0]['monotonic_ns'] < loaded[0]['monotonic_ns'] < commits[1]['monotonic_ns']
    all_samples, total_bytes, generations = set(), 0, []
    head = read(method / 'state/control.json')['head']
    while head is not None:
        directory = method / 'state/generations' / head['generation']
        token, manifest = read(directory / 'token.json'), read(directory / 'manifest.json')
        assert sha(directory / 'token.json') == head['token_sha256']
        assert sha(directory / 'manifest.json') == token['manifest_sha256']
        assert sha(directory / 'intent.json') == token['intent_sha256']
        assert manifest['parent'] == token['parent']
        actual = {str(p.relative_to(directory / 'checkpoint')): p for p in (directory / 'checkpoint').rglob('*') if p.is_file()}
        assert set(actual) == set(manifest['files'])
        for name, details in manifest['files'].items():
            path = actual[name]
            assert not path.is_symlink() and path.stat().st_size == details['size'] and sha(path) == details['sha256']
            total_bytes += details['size']
        generations.append((directory.name, manifest))
        head = token['parent']
    generations.reverse()
    assert [g for g, _ in generations] == [e['generation'] for e in commits]
    receipts_by_sample = {}
    for gid, manifest in generations:
        samples = [s['sample'] for u in manifest['updates'] for g in u['groups'] for s in g['samples']]
        assert len(samples) == len(set(samples)) == 32 and not all_samples.intersection(samples)
        for update in manifest['updates']:
            for group in update['groups']:
                assert group['k'] == 8 and sorted(s['sample_index'] for s in group['samples']) == list(range(8))
                receipts_by_sample.update({s['sample']: s['receipt'] for s in group['samples']})
        all_samples.update(samples)
        assert set(manifest['data']['consumed']) == all_samples
        assert set(manifest['data']['drawn']) == all_samples | {p['sample'] for p in manifest['data']['pending']}
        ordered, matched = [], {}
        for event in ('update_prepared', 'optimizer_applied', 'scheduler_applied', 'async_scheduled', 'async_finalized', 'committed'):
            matches = [e for e in events if e['event'] == event and e.get('generation') == gid]
            assert len(matches) == 1, (gid, event, len(matches))
            ordered.append(matches[0]['monotonic_ns'])
            matched[event] = matches[0]
        assert ordered == sorted(ordered)
        assert len({e['pid'] for e in matched.values()}) == 1
        assert matched['optimizer_applied']['stats']['update_successful'] == 1.0
        call_id = matched['async_scheduled']['call_id']
        assert matched['async_finalized']['call_id'] == call_id
        optimizer = [r for r in manifest['receipts'] if r['kind'] == 'optimizer']
        finalized = [r for r in manifest['receipts'] if r['kind'] == 'finalize']
        assert len(optimizer) == len(finalized) == 1
        assert optimizer[0]['successful'] is True and optimizer[0]['scheduler_applied'] is True
        assert optimizer[0]['physical_updates'] == [u['physical_update_id'] for u in manifest['updates']]
        assert finalized[0]['writer_closed'] is True and finalized[0]['async_call_ids'] == [call_id]
        assert all(r['snapshot_id'] == manifest['data_snapshot_id'] for r in manifest['receipts'])
    adopted_consumed = set()
    blobs = method / 'artifacts/blobs'
    for path in (method / 'artifacts/samples').glob('*/adoption.json'):
        adoption = read(path)
        destination = adoption['destination']; sample = destination['sample']
        if sample not in receipts_by_sample or receipts_by_sample[sample]['attempt'] != destination:
            continue
        assert adoption['origin']['attempt']['epoch'] < destination['epoch']
        receipt = receipts_by_sample[sample]
        pairs = {}
        for name, key in (('response', 'response_sha256'), ('reward', 'reward_sha256'), ('tensor', 'tensor_input_sha256')):
            ref = read(path.parent / (name + '.json'))['blob']
            assert ref['sha256'] == receipt['payload'][key]
            blob = blobs / ref['sha256']
            assert sha(blob) == ref['sha256'] and blob.stat().st_size == ref['size']
            current = read(blob)
            assert current['binding']['attempt'] == destination
            old_sha = adoption['origin']['payload'][key]
            assert sha(blobs / old_sha) == old_sha
            old = read(blobs / old_sha)
            assert old['binding']['attempt'] == adoption['origin']['attempt']
            pairs[name] = current['payload'], old['payload']
        assert pairs['response'][0] == pairs['response'][1]
        assert pairs['reward'][0]['return'] == pairs['reward'][1]['return']
        assert pairs['tensor'][0]['fields'] == pairs['tensor'][1]['fields']
        for field in pairs['tensor'][0]['fields'].values():
            ref = field['raw']
            assert sha(blobs / ref['sha256']) == ref['sha256']
            assert (blobs / ref['sha256']).stat().st_size == ref['size']
        adopted_consumed.add(sample)
    assert adopted_consumed, 'no persisted replay actually reached committed training'
    return {'verified': True, 'updates': 3, 'samples': len(all_samples), 'generations': len(generations),
            'checkpoint_bytes_hashed': total_bytes, 'adopted_samples_committed': len(adopted_consumed),
            'native_full_state_load_exact': True, 'async_saves_finalized': 3,
            'scope': 'single actor engineering save/new-process-load; not injected-fault recovery or formal matrix'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.evidence), indent=2))
