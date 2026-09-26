"""Offline estimate: if the R trainer were SIGKILLed at time t of a completed
no-fault run, how many accepted-but-unretained receipts would the restarted
process be allowed to reuse (TrainingBridge.authorize conditions)?

Recovery head at t = latest generation whose DCP finalize happened before t
(lag-1 promotion). Reuse requires an accepted receipt not consumed by the
recovered chain and 0 <= V - v <= max_lag for every output version v, where V
is the recovered policy version (policy.json). Accept time ~ tensor.json mtime.
"""
import glob
import json
import os
import statistics
import sys
from pathlib import Path

MAX_LAG = 2


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def estimate(run):
    run = Path(run)
    method = run / 'rewardtxn'
    pilot = [e for f in glob.glob(str(run / 'observer-pilot/*.jsonl')) for e in rows(f)]
    offset = pilot[0]['wall_time_ns'] - pilot[0]['monotonic_ns']
    events = rows(method / 'events.jsonl')
    gens = [e['generation'] for e in events if e['event'] == 'update_prepared']
    samples = {e['generation']: set(e['samples']) for e in events if e['event'] == 'update_prepared'}
    finalized = {e['generation']: int(e['monotonic_ns']) for e in events if e['event'] == 'async_finalized'}
    version = {g: json.load(open(method / 'state/generations' / g / 'checkpoint/policy.json'))['version'] for g in gens}
    blobs = method / 'artifacts/blobs'
    accepted = []
    for index in glob.glob(str(method / 'artifacts/samples/*/tensor.json')):
        record = json.load(open(blobs / json.load(open(index))['blob']['sha256']))
        sample = record['binding']['attempt']['sample']
        response = json.load(open(Path(index).with_name('response.json')))
        payload = json.load(open(blobs / response['blob']['sha256']))['payload']
        accepted.append((os.stat(index).st_mtime_ns - offset, sample, payload['output_versions']))
    start = min(t for t, _, _ in accepted)
    end = max(finalized.values())
    results = []
    for t in range(start, end, 500_000_000):  # every 0.5 s
        done = [g for g in gens if finalized.get(g, 1 << 62) < t]
        head = done[-1] if done else None
        V = version[head] if head else 0
        consumed = set().union(*(samples[g] for g in done)) if done else set()
        inflight = [(s, v) for a, s, v in accepted if a < t and s not in consumed]
        ok = [s for s, v in inflight if v and all(0 <= V - x <= MAX_LAG for x in v)]
        ahead = [s for s, v in inflight if any(V - x < 0 for x in v)]
        results.append((len(inflight), len(ok), len(ahead)))
    n = len(results)
    return {'run': run.name, 'cut_points': n,
            'inflight_accepted_median': statistics.median(r[0] for r in results),
            'reusable_median': statistics.median(r[1] for r in results),
            'reusable_mean': round(statistics.mean(r[1] for r in results), 2),
            'share_cuts_with_any_reusable': round(sum(r[1] > 0 for r in results) / n, 3),
            'rejected_version_ahead_median': statistics.median(r[2] for r in results),
            'reusable_fraction_of_inflight': round(sum(r[1] for r in results) / max(1, sum(r[0] for r in results)), 3)}


if __name__ == '__main__':
    out = [estimate(p) for p in sys.argv[1:]]
    print(json.dumps(out, indent=1))
