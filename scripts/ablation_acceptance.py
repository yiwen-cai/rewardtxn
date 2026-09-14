#!/usr/bin/env python3
"""Validate all seven technical smoke outputs without selecting on accuracy."""
import argparse
import json
from pathlib import Path

from ablation_run import digest, verify, write

ARMS=('O','R','DBM','LOGM','BOTHM','PAYLOAD','LITE')


def validate(batch):
    frozen=verify(batch)
    rows=[]
    for arm in ARMS:
        matches=list((batch/'runs').glob(f'smoke-*-{arm}-s11'))
        if len(matches)!=1: raise RuntimeError(('expected exactly one smoke per arm',arm,matches))
        run=matches[0]
        success=json.loads((run/'run_success.json').read_text())
        assert success['steps']==10 and success['arm']==arm and success['status']=='technical_success'
        assert not (run/'failure.json').exists()
        meta=json.loads((run/'meta.json').read_text())
        assert meta['freeze_sha256']==digest(batch/'freeze.json')
        decision=success['decision']; assert decision['fatal'] is False
        assert decision['nonce']==meta['environment']['ABLATION_RUN_NONCE']
        assert json.loads((run/'control/shutdown.json').read_text())==success['shutdown']
        for component,barrier in decision['barriers'].items():
            manifest_path={'rm-worker':'diagnostic_export/export_manifest.json','rm-host':'manager_closed_manifest.json',
                           'driver':'driver_closed_manifest.json','evaluator':'evaluation_manifest.json'}[component]
            path=run/manifest_path
            assert digest(path)==barrier['manifest_sha256']
            manifest=json.loads(path.read_text()); assert manifest['fatal'] is False
            for item in manifest['files']: assert digest(item['path'])==item['sha256']
        audit=json.loads((run/'consumption_audit.json').read_text())
        assert audit['consumed']==320 and audit['consumed_payload_coverage']==1
        assert audit['consumed_reward_mismatches']==audit['consumed_source_label_and_rescore_mismatches']==audit['unknown_attempt_terminal_count']==0
        assert audit['scheduler']['gpu_uuid']==frozen['gpu_map'][0]['uuid']
        engine_files=list(run.glob('engine_rank*.json')); assert len(engine_files)==3
        devices=[]
        for path in engine_files:
            engine=json.loads(path.read_text())
            visible=engine['cuda_visible_devices']
            mapping=[int(x) for x in visible.split(',')] if visible else list(range(4))
            devices.append(mapping[engine['server_args']['base_gpu_id']])
        assert sorted(devices)==[1,2,3],devices
        export=json.loads((run/'diagnostic_export/export_manifest.json').read_text())
        assert export['variant']==arm and export['sealed'] and export['fatal'] is False
        assert export['counters']['rss_bytes']<=frozen['limits']['rss_bytes']
        rows.append(dict(arm=arm,run=str(run),consumed=audit['consumed'],tail=audit['tail_count'],
                         scheduler=audit['scheduler'],rollout_device_indices=devices,counters=export['counters'],
                         evidence_sha256={n:digest(run/n) for n in ['run_success.json','consumption_audit.json','evaluation_manifest.json']}))
    return dict(pass_=True,source=frozen['source'],freeze_sha256=digest(batch/'freeze.json'),arms=rows,
                scope='seven 10-step technical smokes; 500-step optimizer schedule; no accuracy selection')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',required=True,type=Path);a=p.parse_args()
    result=validate(a.batch);result['pass']=result.pop('pass_')
    path=a.batch/'smoke_acceptance.json'
    if path.exists(): raise FileExistsError(path)
    write(path,result)
    print('seven-arm smoke acceptance PASS')
