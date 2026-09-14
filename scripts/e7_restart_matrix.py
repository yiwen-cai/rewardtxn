#!/usr/bin/env python3
"""Versioned clean matrix. --check is read-only; training requires invoking without it."""
import argparse
import datetime
import json
import subprocess
import sys

from e7_restart_checks import ROOT, SPEC, read, require, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['pilot', 'formal'], default='formal')
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    spec = read(SPEC)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    runs = [(s, g, f'e7restart-{a.stage}-{g}-s{s}-{stamp}') for s in spec[a.stage + '_seeds'] for g in spec['implemented_clean_groups']]
    launcher = ROOT / 'scripts/e7_restart_run.sh'
    for seed, group, name in runs:
        subprocess.run(['bash', str(launcher), a.stage, str(seed), group, name, '--check'], check=True, cwd=ROOT)
    if a.check:
        print(f'Preflight passed for {len(runs)} planned runs; no training launched.')
        return
    manifest = ROOT / spec['pilot_manifest'] if a.stage == 'pilot' else ROOT / f'runs/e7_restart_formal_{stamp}.json'
    require(not manifest.exists(), 'manifest already exists; do not overwrite or silently select replacement pilot')
    content = {'stage': a.stage, 'protocol_sha256': sha(SPEC), 'runs': ['runs/' + n for _, _, n in runs], 'outcomes': []}
    manifest.write_text(json.dumps(content, indent=2) + '\n')
    failed = False
    for seed, group, name in runs:
        result = subprocess.run(['bash', str(launcher), a.stage, str(seed), group, name], cwd=ROOT)
        content['outcomes'].append({'run': 'runs/' + name, 'exit_code': result.returncode})
        manifest.write_text(json.dumps(content, indent=2) + '\n')
        if result.returncode:
            failed = True
            # Preserve the fixed matrix; failed quality must not silently select a replacement seed.
    require(not failed, 'one or more runs failed; inspect manifest; no success or equivalence claim')
    subprocess.run(['python3', str(ROOT / 'scripts/e7_restart_checks.py'), 'aggregate', '--stage', a.stage, '--manifest', str(manifest)], check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        print('E7 matrix stopped: ' + str(exc), file=sys.stderr)
        sys.exit(2)
