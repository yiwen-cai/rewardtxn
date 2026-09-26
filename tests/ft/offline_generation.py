"""Offline file check for one retained R generation, independent of method state code.

A generation may carry a ``pruned.json`` marker (R_SLOWDOWN_FIX_PLAN v2, L3):
bulk DCP shards listed there were removed after later commits. A file missing
without such a marker is unverifiable and fails the audit.
"""
import hashlib
import json


def _bulk(name):
    return name.startswith('native/') and name.endswith('.distcp')


CHUNKED_SCHEMA = 'sha256-chunked-v1'


def file_matches(path, info, sha):
    """Independent check of either manifest file format (whole-file or chunked)."""
    if 'digest_schema' not in info:
        return set(info) == {'size', 'sha256'} and sha(path) == info['sha256']
    assert info['digest_schema'] == CHUNKED_SCHEMA, 'unknown digest schema'
    size, step, chunks = info['size'], info['chunk_bytes'], info['chunk_sha256']
    assert len(chunks) == -(-size // step), 'chunk count'
    bound = hashlib.sha256(CHUNKED_SCHEMA.encode() + b'\0' + size.to_bytes(8, 'big') + step.to_bytes(8, 'big')
                           + b''.join(bytes.fromhex(c) for c in chunks)).hexdigest()
    assert bound == info['chunked_digest'], 'chunked digest binding'
    with open(path, 'rb') as stream:
        actual = [hashlib.sha256(stream.read(step)).hexdigest() for _ in chunks]
        assert stream.read(1) == b'', 'trailing bytes'
    return actual == chunks


def parent_matches(token, parent, generations, sha):
    """Intent/manifest parent vs committed token parent. The lag-1 pipelined
    form binds the then-uncommitted predecessor by its intent.json SHA-256
    (R_PERF_PHASE2_PLAN section 1.1); the token binds the committed token."""
    if isinstance(parent, dict) and set(parent) == {'generation', 'intent_sha256'}:
        return (token['parent'] is not None and token['parent']['generation'] == parent['generation']
                and sha(generations / parent['generation'] / 'intent.json') == parent['intent_sha256'])
    return token['parent'] == parent


def check_files(directory, token, manifest, sha, *, pinned=()):
    """Return (bytes_hashed, files_hashed, files_pruned); assert on any mismatch."""
    checkpoint = directory / 'checkpoint'
    assert not any(p.is_symlink() for p in checkpoint.rglob('*')), directory.name
    files = {str(p.relative_to(checkpoint)): p for p in checkpoint.rglob('*') if p.is_file()}
    expected = manifest['files']
    removed = set()
    marker_path = directory / 'pruned.json'
    if marker_path.exists():
        marker = json.loads(marker_path.read_text())
        assert marker['schema'] == 1 and marker['generation'] == directory.name, 'bad pruning marker'
        assert marker['manifest_sha256'] == token['manifest_sha256'], 'pruning marker/manifest mismatch'
        for name, info in marker['removed'].items():
            assert _bulk(name) and expected.get(name) == info, f'pruning marker lists {name}'
        assert directory.name not in pinned, f'pinned generation {directory.name} was pruned'
        removed = set(marker['removed'])
    assert set(files) <= set(expected), f'unmanifested files in {directory.name}'
    missing = set(expected) - set(files)
    assert missing <= removed, f'unverifiable: {sorted(missing - removed)} missing without marker'
    size = 0
    for name, path in files.items():
        info = expected[name]
        assert path.stat().st_size == info['size'] and file_matches(path, info, sha), name
        size += info['size']
    return size, len(files), len(missing)


def pinned_generations(state_root):
    root = state_root / 'pins'
    if not root.exists():
        return {}
    result = {}
    for path in root.glob('*.json'):
        record = json.loads(path.read_text())
        assert path.name == record['reason'] + '-' + record['generation'] + '.json', 'bad pin record'
        result.setdefault(record['generation'], []).append(record['reason'])
    return result
