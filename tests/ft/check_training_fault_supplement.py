"""Post-run coverage: all job receipts, exact network, and frozen source files."""
import collections
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def verify(root):
    repo=Path(__file__).resolve().parents[2]
    records=[read(p) for p in (root/'areal').rglob('job-lifecycle/*.json')]
    roles=collections.Counter(Path(r['log_path']).name for r in records)
    assert roles=={'trainer.log':2,'llm_server.log':2}
    assert all(r['cleanup']['empty'] for r in records)
    created=read(root/'inspect-created.json')[0]
    final=read(root/'inspect-final.json')[0]
    network=(root/'network.id').read_text().strip()
    assert read(root/'network-cleanup.json')['id']==network
    assert created['HostConfig']['NetworkMode']==network
    assert {v['NetworkID'] for v in final['NetworkSettings']['Networks'].values()}=={network}
    assert created['Id']==final['Id']==(root/'container.id').read_text().strip()
    for name,digest in read(root/'source-sha256.json').items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
    frozen=read(root/'frozen.json')
    assert frozen['config_sha256']==hashlib.sha256((root/'controller-config.json').read_bytes()).hexdigest()
    identity=read(root/'identity.json')
    first=json.loads((root/'events.jsonl').read_text().splitlines()[0])
    assert first['identity']['pid']==1 and first['identity']['pid_namespace']!=identity['host_pidns']
    return {'verified':True,'job_receipts':dict(roles),'all_jobs_cleaned':True,
        'network_id_bound_to_actual_container':True,'frozen_sources_unchanged':True,
        'raw_controller_config_sha256_verified':True,'private_pid_namespace_verified':True}


if __name__=='__main__':
    print(json.dumps(verify(Path(sys.argv[1])),indent=2))
