"""Post-acceptance DCP retention for the minimal experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import time


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def shards(root, arm):
    base = root / ('areal/checkpoints' if arm == 'A' else 'rewardtxn/state/generations')
    assert base.is_dir() and not base.is_symlink()
    result = sorted(base.rglob('*.distcp'))
    assert result and all(p.is_file() and not p.is_symlink() for p in result)
    if arm == 'R':
        assert all(p.parent.name == 'native' and p.parent.parent.name == 'checkpoint' for p in result)
    return result


def retain_or_clear(root, *, keep_full=False):
    root = Path(root).resolve()
    case = json.loads((root / 'ft1-case.json').read_text())
    assert case.get('minimal') is True and root.parent.name == 'minimal_evidence'
    status = json.loads((root / 'acceptance-status.json').read_text())
    source = json.loads((root / 'source-verification.json').read_text())
    result = json.loads((root / 'functional-verification.json').read_text())
    assert status['result'] == 'functional_verification_written' and source['verified']
    assert result['classification'] in ('correct_recovered', 'safe_discard', 'no_fault_verified')
    assert result['full_native_reload_verified'] and result['safety_verified_for_retained_chain']
    if case['scenario'] == 'F2':
        assert source['target_rows'] == 32 and source['same_execution_source_rows'] in range(33)
    existing = root / 'storage-cleanup.json'
    if existing.exists():
        raise FileExistsError(existing)
    started = time.monotonic()
    before = shutil.disk_usage(root).free
    records = [{'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
               for path in shards(root, case['arm'])]
    report = {'arm': case['arm'], 'scenario': case['scenario'], 'keep_full': keep_full,
              'shards': records, 'status': 'hashed_before_cleanup', 'free_before': before}
    write(existing, report)
    if not keep_full:
        for record in records:
            path = root / record['path']
            assert path.is_file() and not path.is_symlink() and path.stat().st_size == record['bytes']
            path.unlink()
        assert all(not (root / record['path']).exists() for record in records)
    report['status'] = 'retained_full' if keep_full else 'shards_deleted'
    report['free_after'] = shutil.disk_usage(root).free
    report['freed_bytes_observed'] = report['free_after'] - before
    report['elapsed_seconds'] = time.monotonic() - started
    write(existing, report)
    return report
