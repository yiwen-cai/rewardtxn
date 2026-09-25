"""Evaluate the fixed R_SLOWDOWN_FIX_PLAN v2 gate criteria on one 30-step F2 pair.

Read-only over evidence. Usage: check_ft1_gate.py <R evidence> <A evidence> <disk samples jsonl> <output json>
"""
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def slope(values):
    n = len(values)
    if n < 3:
        return None
    mean_x, mean_y = (n - 1) / 2, sum(values) / n
    num = sum((i - mean_x) * (v - mean_y) for i, v in enumerate(values))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den


def main(r_root, a_root, disk_path, output):
    r_root, a_root = Path(r_root), Path(a_root)
    case = read(r_root / 'ft1-case.json')
    criteria = case['gate_criteria']
    assert case['arm'] == 'R' and read(a_root / 'ft1-case.json')['arm'] == 'A'
    events = rows(r_root / 'rewardtxn/events.jsonl')
    commits = [e for e in events if e['event'] == 'committed']
    # Step intervals within one trainer process; the restart gap is reported separately.
    intervals, restart = [], []
    for previous, current in zip(commits, commits[1:]):
        gap = (current['monotonic_ns'] - previous['monotonic_ns']) / 1e9
        (intervals if current['pid'] == previous['pid'] else restart).append(gap)
    recoveries = [e for e in events if e['event'] == 'recovery_selected']
    loaded = [e for e in recoveries if e['generation'] is not None]
    disk = rows(disk_path) if Path(disk_path).exists() else []
    peak = max((d['r_bytes'] for d in disk), default=None)
    acceptance = {arm: read(root / 'acceptance-status.json').get('result')
                  for arm, root in (('R', r_root), ('A', a_root))}
    wall = {arm: read(root / 'cost.json')['wall_seconds'] for arm, root in (('R', r_root), ('A', a_root))}
    fitted = slope(intervals)
    checks = {
        'r_wall_seconds': {'value': wall['R'], 'limit': criteria['r_wall_seconds_max'],
                           'pass': wall['R'] <= criteria['r_wall_seconds_max']},
        'r_step_interval_slope': {'value': fitted, 'limit': criteria['r_step_interval_slope_max'],
                                  'pass': fitted is not None and fitted < criteria['r_step_interval_slope_max']},
        'r_recovery_full_hash_generations': {
            'value': [len(e['content_hashed']) for e in loaded], 'limit': criteria['r_recovery_full_hash_generations'],
            'pass': bool(loaded) and all(len(e['content_hashed']) == criteria['r_recovery_full_hash_generations']
                                         and e['content_hashed'] == [e['generation']] for e in loaded)},
        'acceptance': {'value': acceptance, 'limit': criteria['acceptance_result'],
                       'pass': all(v == criteria['acceptance_result'] for v in acceptance.values())},
        'r_disk_peak_bytes': {'value': peak, 'limit': criteria['r_disk_peak_bytes_max'],
                              'pass': peak is not None and peak <= criteria['r_disk_peak_bytes_max']},
    }
    report = {'gate': case['gate'], 'formal_sample': False, 'passed': all(c['pass'] for c in checks.values()),
              'checks': checks, 'wall_seconds': wall, 'r_committed_steps': len(commits),
              'r_step_intervals_seconds': intervals, 'r_restart_gap_seconds': restart,
              'r_recovery_events': recoveries, 'disk_samples': len(disk)}
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps({'passed': report['passed'], **{k: v['pass'] for k, v in checks.items()}}))
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
