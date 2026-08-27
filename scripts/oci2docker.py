#!/usr/bin/env python3
"""OCI layout -> docker save tar 转换器"""
import gzip
import io
import json
import os
import shutil
import tarfile

OCI_DIR = '/tmp/areal-oci'
OUT_TAR = '/tmp/areal-docker-save.tar'

index = json.load(open(f'{OCI_DIR}/index.json'))
mdigest = index['manifests'][0]['digest'].split(':')[1]
manifest = json.load(open(f'{OCI_DIR}/blobs/sha256/{mdigest}'))
config_digest = manifest['config']['digest'].split(':')[1]
layers = [l['digest'].split(':')[1] for l in manifest['layers']]
print(f'config={config_digest[:16]} layers={len(layers)}')

with tarfile.open(OUT_TAR, 'w') as tf:
    cfg_name = f'{config_digest}.json'
    tf.add(f'{OCI_DIR}/blobs/sha256/{config_digest}', arcname=cfg_name)
    layer_names = []
    for i, ld in enumerate(layers):
        src = f'{OCI_DIR}/blobs/sha256/{ld}'
        lname = f'{ld}/layer.tar'
        layer_names.append(lname)
        tmp = f'/tmp/layer_{i}.tar'
        with gzip.open(src, 'rb') as fin, open(tmp, 'wb') as fout:
            shutil.copyfileobj(fin, fout, length=1 << 20)
        tf.add(tmp, arcname=lname)
        os.remove(tmp)
        print(f'layer {i} done ({ld[:12]})', flush=True)
    mj = json.dumps([{
        'Config': cfg_name,
        'RepoTags': ['areal-project/areal-runtime:v2.0.0-sglang'],
        'Layers': layer_names,
    }]).encode()
    ti = tarfile.TarInfo('manifest.json')
    ti.size = len(mj)
    tf.addfile(ti, io.BytesIO(mj))
print('docker-save tar written:', OUT_TAR)
