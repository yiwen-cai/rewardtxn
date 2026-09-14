#!/usr/bin/env python3
"""Regrade fixed pilot payloads through both real RM paths, including audit I/O."""
import asyncio
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path('/workspace')
OUT = ROOT / 'runs/pilot-causal-audit-20260911'


async def main():
    manifest = json.loads((ROOT / 'runs/e7_restart_0.5B_20260911_pilot.json').read_text())
    records = []
    fixed = {}
    for run in manifest['runs']:
        for step in [0, 249, 499]:
            path = ROOT / run / f'rollout_debug/{step}.pt'
            raw = torch.load(path, map_location='cpu', weights_only=False)
            outputs = {}
            for group, module in [('group_rm', 'day2_custom_rm'), ('b6', 'phase2_seal_rm')]:
                directory = OUT / 'reward_replay' / Path(run).name / str(step) / group
                directory.mkdir(parents=True, exist_ok=False)
                os.environ.update(RTX_FAULT='none', RTX_FAULT_START='-1', RTX_FAULT_END='-1',
                                  RTX_RUN_DIR=str(directory), RTX_CAS_INDEX_DIR=str(directory),
                                  RTX_SEAL=str(int(group == 'b6')), RTX_GROUP_RM='1',
                                  RTX_SEAL_AUTO_FIX=str(int(group == 'b6')), RTX_FAULT_WINDOWS='')
                rm = importlib.reload(importlib.import_module(module))
                samples = [SimpleNamespace(**copy.deepcopy(s)) for s in raw['samples']]
                rm._lock = asyncio.Lock()
                values = await rm.rm_function(SimpleNamespace(), samples)
                outputs[group] = values
                if hasattr(rm, '_reset_cas_connection'):
                    rm._reset_cas_connection()
                # Reverse-order check uses a separate persistent audit directory.
                reverse_dir = directory / 'reversed'
                reverse_dir.mkdir()
                os.environ.update(RTX_RUN_DIR=str(reverse_dir), RTX_CAS_INDEX_DIR=str(reverse_dir))
                rm = importlib.reload(rm)
                rm._lock = asyncio.Lock()
                reverse = await rm.rm_function(SimpleNamespace(), samples[::-1])
                assert reverse[::-1] == values, (run, step, group, 'ordering')
                if hasattr(rm, '_reset_cas_connection'):
                    rm._reset_cas_connection()
                if run == manifest['runs'][0] and step == 0:
                    prepared = copy.deepcopy(raw)
                    for s, reward in zip(prepared['samples'], values):
                        s['reward'] = reward
                    target = OUT / f'fixed_{group}.pt'
                    assert not target.exists()
                    torch.save(prepared, target)
                    fixed[group] = str(target)
            expected = [s['reward'] for s in raw['samples']]
            rec = dict(run=run, step=step, samples=len(expected),
                       input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       between_rm_mismatches=sum(a != b for a, b in zip(*outputs.values())),
                       original_reward_mismatches={g: sum(a != b for a, b in zip(v, expected))
                                                   for g, v in outputs.items()},
                       reverse_order_pass=True)
            records.append(rec)
            print(json.dumps(rec), flush=True)
    report = dict(records=records, fixed_inputs=fixed, real_audit_io=True,
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                 ['scripts/day2_custom_rm.py', 'scripts/phase2_seal_rm.py',
                                  'third_party/slime/slime/rollout/rm_hub/math_utils.py']})
    (OUT / 'reward_parity.json').write_text(json.dumps(report, indent=2) + '\n')
    assert all(r['between_rm_mismatches'] == 0 and not any(r['original_reward_mismatches'].values())
               for r in records)


if __name__ == '__main__':
    asyncio.run(main())
