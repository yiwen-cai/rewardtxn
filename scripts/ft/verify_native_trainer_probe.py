"""Read-only engineering acceptance; never an oracle-valid classifier."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def records(paths):
    return [json.loads(line) for path in paths for line in path.read_text().splitlines() if line]


def manifest(path):
    result = []
    for file in sorted(path.rglob('*')):
        if file.is_file():
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            result.append({'path': str(file.relative_to(path)), 'size': file.stat().st_size, 'sha256': digest.hexdigest()})
    return result


def verify_training_evidence(journal, witness, pilot):
    from .areal_native_trainer_probe import EVENT, EVIDENCE
    checks = {}
    def check(name, passed):
        checks[name] = bool(passed)
    registrations = [e for e in journal if e['kind'] == 'descendant_registered' and e['role'] == 'trainer']
    reg = {e['incarnation']: e for e in registrations}
    check('two_authenticated_incarnations', len(registrations) == len(reg) == 2 and
          len({json.dumps(e['identity'], sort_keys=True) for e in registrations}) == 2 and
          all(e['ancestry'] for e in registrations))
    initial = [e for e in witness if e['event'] == 'registered' and e['assignment']['status'] == 'pending']
    replacement = [e for e in witness if e['event'] == 'registered' and e['assignment']['status'] == 'already_fired']
    check('one_assignment_each', len(initial) == len(replacement) == 1)
    check('all_witnesses_bound_to_authenticated_identity', bool(witness) and all(
        e.get('incarnation') in reg and e.get('identity') == reg[e['incarnation']]['identity']
        and e.get('pid') == e['identity']['pid'] for e in witness))
    if len(initial) != 1 or len(replacement) != 1:
        return checks
    first, second = initial[0], replacement[0]
    assignments = [e for e in journal if e['kind'] == 'injection_assignment' and e['role'] == 'trainer']
    check('assignments_match_controller', len(assignments) == 2 and all(
        any(a['incarnation'] == w['incarnation'] and a['injection'] == w['assignment'] for a in assignments)
        for w in (first, second)) and first['incarnation'] != second['incarnation']
        and all(w['assignment']['event_id'] == EVENT for w in (first, second))
        and first['assignment']['event_nonce'] == second['assignment']['event_nonce'])
    sent = [e for e in journal if e['kind'] == 'signal_sent']
    exits = [e for e in journal if e['kind'] == 'process_exit_observed']
    ready_journal = [e for e in journal if e['kind'] == 'ready']
    check('one_exact_signal_and_exit', len(sent) == len(exits) == len(ready_journal) == 1 and all(
        e.get('incarnation') == first['incarnation'] and e.get('identity') == first['identity']
        and e.get('event_id') == EVENT for e in sent + exits + ready_journal)
        and sent[0]['signal'] == 9 and all(e.get('event_nonce') == first['assignment']['event_nonce'] for e in sent + exits)
        and ready_journal[0]['evidence'] == EVIDENCE)
    def events(event, person):
        return [e for e in witness if e['event'] == event and e['incarnation'] == person['incarnation']]
    ready = events('ready_witness', first)
    initial_updates = events('successful_update', first)
    saved = [e for e in events('complete_save', first) if e['global_step'] == 0]
    fields = ('checkpoint_path', 'checkpoint', 'metadata_path', 'metadata')
    check('second_success_after_complete_step0', len(ready) == len(saved) == 1 and len(initial_updates) == 2
          and [e['ordinal'] for e in initial_updates] == [1, 2]
          and all(e['stats']['update_successful'] == 1 for e in initial_updates)
          and saved[0]['time_ns'] < initial_updates[1]['time_ns'] < ready[0]['time_ns']
          and ready[0]['evidence'] == EVIDENCE
          and ready[0]['predecessor'] == {k: saved[0][k] for k in fields})
    def pilot_matches(e):
        identity = second['identity']
        return (e.get('pid') == identity['pid'] and e.get('starttime') == identity['start_time']
                and e.get('boot_id') == identity['boot_id'] and e.get('cgroup', '').strip() == identity['cgroup']
                and e.get('process_incarnation') == second.get('pilot_incarnation'))
    loads = [e for e in pilot if e['event'] == 'checkpoint_load_returned' and pilot_matches(e)]
    matching = [e for e in loads if ready and e['files'] == ready[0]['predecessor']['checkpoint']
                and e['path'] == ready[0]['predecessor']['checkpoint_path'] and e['with_optim'] and e['weight_format'] == 'dcp']
    check('actual_native_load_matches_predecessor_fullhash', len(ready) == len(matching) == 1)
    recovered = events('recover_info_loaded', second)
    check('actual_step0_recoverinfo_and_full_metadata', len(recovered) == len(ready) == 1
          and recovered[0]['last_step_info']['global_step'] == 0
          and recovered[0]['metadata'] == ready[0]['predecessor']['metadata']
          and recovered[0]['metadata_path'] == ready[0]['predecessor']['metadata_path'])
    updates = events('successful_update', second)
    check('updates_after_actual_load', len(updates) == 2 and len(matching) == len(recovered) == 1
          and [e['ordinal'] for e in updates] == [1, 2]
          and all(e['stats']['update_successful'] == 1 for e in updates)
          and matching[0]['wall_time_ns'] < recovered[0]['time_ns'] < min(e['time_ns'] for e in updates))
    finals = [e for e in events('complete_save', second) if e['global_step'] == 2]
    check('final_step2_save', len(finals) == 1 and len(updates) == 2 and finals[0]['time_ns'] > max(e['time_ns'] for e in updates))
    return checks


def verify(directory):
    root = Path(directory)
    journal = records([root / 'events.jsonl'])
    witness = records(sorted(root.glob('trainer-witness-*.jsonl')))
    pilot = records(sorted((root / 'areal/pilot_events').glob('*.jsonl')))
    checks = verify_training_evidence(journal, witness, pilot)
    def check(name, passed):
        checks[name] = bool(passed)
    replacement = [e for e in witness if e['event'] == 'registered' and e['assignment']['status'] == 'already_fired']
    finals = [e for e in witness if replacement and e['event'] == 'complete_save'
              and e['incarnation'] == replacement[0]['incarnation'] and e['global_step'] == 2]
    if len(finals) == 1:
        final = finals[0]
        def local(path):
            return root / Path(path).relative_to('/output')
        check('final_checkpoint_rehashed', manifest(local(final['checkpoint_path'])) == final['checkpoint'])
        check('final_metadata_rehashed', manifest(local(final['metadata_path'])) == final['metadata'])
    else:
        check('final_checkpoint_rehashed', False)
        check('final_metadata_rehashed', False)
    text = (root / 'launcher.log').read_text()
    runs = re.findall(r'run_id=(\d+), is_recover_run=(True|False)', text)
    check('native_run0_then_run1', runs == [('0', 'False'), ('1', 'True')])
    observations = [e for e in journal if e['kind'] == 'method_observation']
    check('native_exit1_retained', len(observations) == 1 and observations[0].get('launcher_exit_code') == 1)
    check('native_completed_exception_retained', 'COMPLETED' in text and 'JobException' in text)
    supervisor = json.loads((root / 'supervisor_result.json').read_text())
    check('containment_cleanup', supervisor['cleanup_confirmed'] and not supervisor['cleanup_errors'] and supervisor['failure'] is None)
    check('network_removed', json.loads((root / 'network-cleanup.json').read_text())['removed'])
    preflight = json.loads((root / 'gpu-preflight-exit.json').read_text())
    check('preflight_subprocess_exited', preflight['exit_code'] == 0)
    return {'passed': all(checks.values()), 'checks': checks, 'method_observation': observations,
            'native_runs': runs, 'scope': 'one trainer process kill/native retry; not R, replay, F4 or pending recovery'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    try:
        result = verify(args.run_dir)
    except (OSError, ValueError, KeyError) as exc:
        result = {'passed': False, 'incomplete_evidence': str(exc)}
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
