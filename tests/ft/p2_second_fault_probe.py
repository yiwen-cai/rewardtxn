"""One isolated CPU two-fault chain; no training or multi-target oracle claim."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ft import run, state, faults
from scripts.ft.descendants import snapshot


def local(name):
    spec = importlib.util.spec_from_file_location('second_' + name, ROOT / 'tests/ft' / (name + '.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


helpers = local('p2_controller_probe')
save, contained = helpers.save, helpers.contained
COMPONENTS = local('test_state').COMPONENTS
GROUPS = ('bootstrap', 'primary', 'second')
SCENARIOS = ('two-faults', 'second-early', 'wrong-target', 'second-incomplete')


def data_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def frozen_inputs():
    groups = {}
    for number, name in enumerate(GROUPS):
        groups[name] = [{'prompt': [{'role': 'user', 'content': f'Return integer {number + 1}.'}],
            'label': str(number + 1), 'completion': str(number + 1), 'tokens': [100 + number, 200 + i],
            'loss_mask': [0, 1], 'logprobs': [0.0, -0.25], 'policy_version': number} for i in range(8)]
    return {'scope': 'CPU full-state fixture; no optimizer backend', 'groups': groups,
            'ranks': ['actor:0', 'actor:1'], 'components': list(COMPONENTS),
            'logical_updates': ['bootstrap-update', 'primary-update', 'second-update'],
            'source_order': [f'{g}:{i}' for g in GROUPS for i in range(8)]}


def component_bytes(frozen, group, rank, component):
    return data_bytes({'component': component, 'rank': rank, 'retained_groups': list(GROUPS[:GROUPS.index(group) + 1]),
        'source_sha256': sha(data_bytes(frozen)), 'cpu_fixture_state': [GROUPS.index(group) + 1] * 128})


def record(directory, kind, **fields):
    row = dict(kind=kind, identity=snapshot(os.getpid()), monotonic_ns=time.monotonic_ns(), **fields)
    with (directory / 'worker-events.jsonl').open('ab') as stream:
        stream.write(data_bytes(row) + b'\n'); stream.flush(); os.fsync(stream.fileno())


def acquire(directory, initial=False):
    root = directory / 'state'
    if initial:
        owner = state.acquire_owner(root, -1, run_nonce='cpu-two-fault',
            config_sha256=sha(data_bytes(read(directory / 'inputs.json'))), verifier_version='fixture-exact-v1')
        prior = []
    else:
        control = read(root / 'control.json')
        prior = control['processes']
        owner = state.acquire_owner(root, control['epoch'], prior)
    record(directory, 'owner_acquired', epoch=owner.epoch, prior_processes=prior)
    return owner


def load_actual(directory, owner, expected_group):
    decision = state.select_recovery(owner)
    record(directory, 'recovery_selected', epoch=owner.epoch, decision=decision)
    if decision['generation'] is None:
        raise RuntimeError('expected a real recovered checkpoint')
    path = owner.root / 'generations' / decision['generation']
    manifest = read(path / 'manifest.json')
    frozen = read(directory / 'inputs.json')
    reads = []
    for component in frozen['components']:
        for rank in frozen['ranks']:
            name = f'{rank.replace(":", "-")}/{component}.bin'
            content = (path / 'checkpoint' / name).read_bytes()
            expected = component_bytes(frozen, expected_group, rank, component)
            if content != expected or sha(content) != manifest['files'][name]['sha256']:
                raise RuntimeError('actual component load differs from frozen fixture state')
            reads.append({'path': name, 'sha256': sha(content), 'size': len(content)})
    record(directory, 'checkpoint_loaded', epoch=owner.epoch, generation=decision['generation'],
        expected_group=expected_group, reads=reads, data=manifest['data'],
        manifest_sha256=sha((path / 'manifest.json').read_bytes()))
    token = read(path / 'token.json')
    return {'generation': decision['generation'], 'token_sha256': sha(data_bytes(token))}, manifest


def build(directory, owner, group, parent=None, previous=None, incomplete=False):
    frozen = read(directory / 'inputs.json')
    entries = []
    for i, raw in enumerate(frozen['groups'][group]):
        key = f'{group}:{i}'
        authorization = state.authorize_attempt(owner, key, None, f'{group}-attempt', expected_policy_version=raw['policy_version'])
        reward = float(raw['completion'].strip() == raw['label'].strip())
        payload = {'response_sha256': sha(data_bytes(raw)), 'reward_sha256': sha(data_bytes(reward)),
            'tensor_input_sha256': sha(data_bytes({'raw': raw, 'reward': reward})),
            'policy_version': raw['policy_version'], 'verifier_version': 'fixture-exact-v1'}
        receipt = state.accept_result(owner, authorization, payload)
        entries.append({'sample_index': i, 'sample': key, 'receipt': receipt})
    index = GROUPS.index(group)
    consumed = ([] if previous is None else list(previous['data']['consumed'])) + [e['sample'] for e in entries]
    pending = []
    if index < 2:
        next_group = GROUPS[index + 1]
        for i, raw in enumerate(frozen['groups'][next_group]):
            prompt = {'messages': raw['prompt'], 'label': raw['label']}
            pending.append({'sample': f'{next_group}:{i}', 'action': 'regenerate', 'prompt': prompt,
                            'prompt_sha256': sha(data_bytes(prompt)), 'k': 8})
    drawn = [] if previous is None else list(previous['data']['drawn'])
    for key in consumed + [p['sample'] for p in pending]:
        if key not in drawn:
            drawn.append(key)
    data = {'source_sha256': sha(data_bytes(frozen)), 'epoch': 0, 'shuffle_state': {'seed': 20260920},
            'drawn': drawn, 'consumed': consumed, 'pending': pending, 'cursor': len(drawn)}
    prompt = {'messages': frozen['groups'][group][0]['prompt'], 'label': frozen['groups'][group][0]['label']}
    intent = {'parent': parent, 'ack_capability': 'none', 'config_sha256': sha(data_bytes(frozen)),
        'expected_ranks': frozen['ranks'], 'data_snapshot_id': group + '-snapshot',
        'components': {component: {rank: [f'{rank.replace(":", "-")}/{component}.bin'] for rank in frozen['ranks']} for component in COMPONENTS},
        'updates': [{'logical_update_id': group + '-update', 'physical_update_id': group + '-physical',
            'train_input_sha256': sha(data_bytes(entries)), 'groups': [{'logical_group_id': group,
                'prompt': prompt, 'prompt_sha256': sha(data_bytes(prompt)), 'k': 8, 'samples': entries}]}]}
    generation = state.prepare_generation(owner, intent, data)
    record(directory, 'prepared', group=group, generation=generation, epoch=owner.epoch, parent=parent, data=data)
    state.record_evidence(owner, generation, {'kind': 'optimizer', 'snapshot_id': group + '-snapshot',
        'physical_updates': [group + '-physical'], 'successful': True, 'scheduler_applied': True})
    checkpoint = owner.root / 'generations' / generation / 'checkpoint'
    for rank in frozen['ranks'][:1] if incomplete else frozen['ranks']:
        rankdir = checkpoint / rank.replace(':', '-'); rankdir.mkdir(parents=True)
        for component in COMPONENTS:
            with (rankdir / (component + '.bin')).open('xb') as stream:
                stream.write(component_bytes(frozen, group, rank, component)); stream.flush(); os.fsync(stream.fileno())
        state.record_evidence(owner, generation, {'kind': 'finalize', 'snapshot_id': group + '-snapshot', 'rank': rank, 'writer_closed': True})
    return generation


def cut_commit(directory, owner, generation, group, scenario):
    original = state._write
    def after_manifest(path, value, immutable=True):
        original(path, value, immutable)
        if path.name == 'manifest.json' and path.parent.name == generation:
            if (path.parent / 'token.json').exists():
                raise RuntimeError('cut already has token')
            manifest = read(path)
            hashes = {name: sha((path.parent / 'checkpoint' / name).read_bytes()) for name in manifest['files']}
            if any(hashes[name] != details['sha256'] for name, details in manifest['files'].items()):
                raise RuntimeError('cut component mismatch')
            event = 'primary-fault' if group == 'primary' else 'second-fault'
            record(directory, 'manifest_cut', event_id=event, group=group, generation=generation, epoch=owner.epoch,
                manifest_sha256=sha(path.read_bytes()), files=manifest['files'], token_absent=True,
                event_nonce=faults.event_nonce(os.environ['FT_RUN_NONCE'], event, int(os.environ['FT_ATTEMPT'])))
            if scenario == 'second-early':
                event = 'second-fault'
            evidence = {'boundary': 'complete_manifest_before_token', 'group': group,
                        'input_sha256': sha((directory / 'inputs.json').read_bytes())}
            if scenario == 'wrong-target':
                original_send = faults.send
                def wrong_role(channel, message):
                    return original_send(channel, dict(message, role='wrong_target') if message.get('kind') == 'ready' else message)
                with patch.object(faults, 'send', wrong_role):
                    faults.barrier(event, evidence)
            else:
                faults.barrier(event, evidence)
            raise RuntimeError('scheduled target unexpectedly released')
    with patch.object(state, '_write', after_manifest):
        state.commit_generation(owner, generation)
    raise RuntimeError('manifest cut not reached')


def worker(directory, scenario):
    contained()
    attempt = int(os.environ['FT_ATTEMPT'])
    with acquire(directory, initial=attempt == 0) as owner:
        if attempt == 0:
            bootstrap = build(directory, owner, 'bootstrap')
            state.commit_generation(owner, bootstrap)
            parent, previous = load_actual(directory, owner, 'bootstrap')
            generation = build(directory, owner, 'primary', parent, previous)
            cut_commit(directory, owner, generation, 'primary', scenario)
        elif attempt == 1:
            parent, previous = load_actual(directory, owner, 'primary')
            generation = build(directory, owner, 'second', parent, previous, incomplete=scenario == 'second-incomplete')
            if scenario == 'second-incomplete':
                try:
                    state.commit_generation(owner, generation)
                except state.StateError as exc:
                    record(directory, 'incomplete_rejected', generation=generation, reason=str(exc))
                    return
                raise RuntimeError('incomplete second candidate committed')
            cut_commit(directory, owner, generation, 'second', scenario)
        elif attempt == 2:
            load_actual(directory, owner, 'second')
            record(directory, 'final_loaded', epoch=owner.epoch)
        else:
            raise RuntimeError('unexpected attempt')


def scenario_run(root, scenario):
    directory = root / scenario; directory.mkdir()
    save(directory / 'inputs.json', frozen_inputs())
    config = {'attempts': [{'target': [sys.executable, str(Path(__file__).resolve()), '--worker', str(directory), '--scenario', scenario]} for _ in range(3)],
        'env': {'PYTHONPATH': str(ROOT)}, 'handshake_timeout': 4, 'run_timeout': 20, 'cleanup_timeout': 2}
    schedule = [{'event_id': event, 'attempt': i, 'target': 'target', 'waiters': ['target'],
        'evidence': {'boundary': 'complete_manifest_before_token', 'group': group,
                     'input_sha256': sha((directory / 'inputs.json').read_bytes())}}
        for i, (event, group) in enumerate((('primary-fault', 'primary'), ('second-fault', 'second')))]
    save(directory / 'config.json', config); save(directory / 'schedule.json', schedule)
    result = run.execute(run.preflight(directory / 'config.json', directory / 'schedule.json'), directory / 'controller')
    save(directory / 'controller-result.json', result)
    expected = 'execution_complete' if scenario == 'two-faults' else 'technical_invalid'
    if result['classification'] != expected:
        raise RuntimeError('controller did not reach expected protocol boundary')
    if scenario == 'second-incomplete':
        command = [sys.executable, str(Path(__file__).resolve()), '--inspect-incomplete', str(directory)]
        child = subprocess.Popen(command)
        fd = os.pidfd_open(child.pid)
        try:
            if child.wait(timeout=8) != 0:
                raise RuntimeError('incomplete candidate recovery inspector failed')
        finally:
            if child.poll() is None:
                signal.pidfd_send_signal(fd, signal.SIGKILL); child.wait(timeout=2)
            os.close(fd)


def old_owner_probe(root):
    directory = root / 'old-owner-live'; directory.mkdir()
    save(directory / 'inputs.json', frozen_inputs())
    owner = acquire(directory, initial=True)
    owner.close()  # Lock released, but its registered process is still truly alive.
    before = (directory / 'state/control.json').read_bytes()
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--challenge', str(directory)])
    fd = os.pidfd_open(child.pid)
    try:
        if child.wait(timeout=8) != 0:
            raise RuntimeError('live prior owner negative failed')
    finally:
        if child.poll() is None:
            signal.pidfd_send_signal(fd, signal.SIGKILL); child.wait(timeout=2)
        os.close(fd)
    if (directory / 'state/control.json').read_bytes() != before:
        raise RuntimeError('live-owner rejection changed persisted epoch')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--make-config', type=Path)
    p.add_argument('--guardian', type=Path)
    p.add_argument('--worker', type=Path)
    p.add_argument('--scenario', choices=SCENARIOS)
    p.add_argument('--inspect-incomplete', type=Path)
    p.add_argument('--challenge', type=Path)
    args = p.parse_args()
    if args.make_config:
        save(args.make_config, {'argv': ['/opt/.venv/bin/python', '/workspace/tests/ft/p2_second_fault_probe.py', '--guardian', '/output/probes'],
            'env': {'PYTHONPATH': '/workspace'}, 'schedule': [], 'timeouts': {'run': 150, 'handshake': 5, 'lease': 10}})
        return
    contained()
    if args.guardian:
        args.guardian.mkdir()
        save(args.guardian / 'source-sha256.json', {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in
            (Path(__file__).resolve(), ROOT / 'tests/ft/p2_controller_probe.py', ROOT / 'scripts/ft/run.py', ROOT / 'scripts/ft/faults.py', ROOT / 'scripts/ft/state.py')})
        def deadline(signum, frame):
            raise RuntimeError('two-fault probe absolute deadline')
        signal.signal(signal.SIGALRM, deadline)
        for scenario in SCENARIOS:
            signal.alarm(30)
            try:
                scenario_run(args.guardian, scenario)
            finally:
                signal.alarm(0)
        signal.alarm(10)
        try:
            old_owner_probe(args.guardian)
        finally:
            signal.alarm(0)
        save(args.guardian / 'summary.json', {'execution_finished': True, 'controller_protocol': 'requires_independent_test_verification',
            'training': 'not_evaluated', 'oracle': 'not_evaluated', 'pending_count': 184})
    elif args.worker:
        worker(args.worker, args.scenario)
    elif args.inspect_incomplete:
        with acquire(args.inspect_incomplete) as owner:
            parent, manifest = load_actual(args.inspect_incomplete, owner, 'primary')
            record(args.inspect_incomplete, 'incomplete_recovery_complete', selected=parent, data=manifest['data'])
    elif args.challenge:
        try:
            acquire(args.challenge)
        except state.StateError as exc:
            if str(exc) != 'previous registered process still alive':
                raise
            save(args.challenge / 'rejection.json', {'reason': str(exc), 'challenger': snapshot(os.getpid())})
        else:
            raise RuntimeError('live prior owner accepted')
    else:
        p.error('choose one action')


if __name__ == '__main__':
    main()
