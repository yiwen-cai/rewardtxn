"""Closed engineering profile. Functions here never select GPUs automatically."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PROFILE = 'native-trainer-4gpu'
GPU_ENV = {'HOME', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'HF_HUB_OFFLINE', 'WANDB_MODE',
           'PYTHONDONTWRITEBYTECODE', 'AREAL_CACHE_DIR'}


def validate_uuids(values):
    if len(values) != 4 or len(set(values)) != 4 or not all(
            re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', value) for value in values):
        raise ValueError('exactly four distinct full GPU UUIDs required')
    return list(values)


def match_cuda_uuids(actual, expected):
    """Compare full physical GPU UUIDs; only the literal GPU- prefix is optional."""
    def canonical(values):
        result = []
        for value in values:
            match = re.fullmatch(r'(?:GPU-)?([0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12})', value) if isinstance(value, str) else None
            if match is None:
                raise ValueError('malformed full CUDA UUID')
            result.append('GPU-' + match.group(1).lower())
        if len(result) != 4 or len(set(result)) != 4:
            raise ValueError('exactly four distinct canonical CUDA UUIDs required')
        return result
    normalized = canonical(actual)
    if set(normalized) != set(canonical(expected)):
        raise RuntimeError(f'CUDA UUID mismatch: {actual}')
    return normalized


def idle_snapshot(values, output):
    validate_uuids(values)
    queries = ['--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu',
               '--query-compute-apps=gpu_uuid,pid,process_name']
    raw = [subprocess.run(['nvidia-smi', query, '--format=csv,noheader,nounits'],
                         check=True, capture_output=True, text=True, timeout=10).stdout for query in queries]
    Path(output).write_text(json.dumps({'gpu_raw': raw[0], 'compute_raw': raw[1]}, indent=2))
    rows = {row[0].strip(): [v.strip() for v in row] for row in csv.reader(raw[0].splitlines())}
    if not set(values) <= rows.keys() or len({rows[v][1] for v in values}) != 1:
        raise RuntimeError('requested GPUs missing or different model')
    if any(float(rows[v][3]) > 100 or float(rows[v][4]) != 0 for v in values):
        raise RuntimeError('selected GPUs are not idle (100 MiB/0% threshold)')
    if any(row and row[0].strip() in values for row in csv.reader(raw[1].splitlines())):
        raise RuntimeError('selected GPU has a compute process')


def provenance(source):
    files = [source / 'scripts/ft' / name for name in ('__init__.py', 'run.py', 'faults.py', 'descendants.py', 'namespace_run.py', 'container_run.py', 'native_gpu.py', 'areal_native_trainer_probe.py', 'areal_pilot.py', 'areal_pilot_hooks.py')] + [
        source / 'third_party/areal/areal/infra/launcher/local.py',
        source / 'third_party/areal/areal/trainer/rl_trainer.py',
        source / 'third_party/areal/areal/engine/megatron_engine.py',
        source / 'third_party/areal/areal/utils/recover.py',
        source / 'docs/experiments/rewardtxn-ft-20260916/native-trainer.yaml']
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_config(config):
    from .areal_native_trainer_probe import EVENT, EVIDENCE
    expected = [{'event_id': EVENT, 'target': 'trainer', 'waiters': ['trainer'], 'evidence': EVIDENCE}]
    if config['schedule'] != expected or config['timeouts'] != {'run': 900, 'handshake': 10, 'lease': 20} or config.get('observer_roles'):
        raise ValueError('native probe requires frozen one-shot schedule and timeouts')
    required = {'HOME': '/tmp', 'AREAL_CACHE_DIR': '/tmp/areal-native-probe',
                'CUDA_VISIBLE_DEVICES': '0,1,2,3', 'CUDA_HOME': '/usr/local/cuda',
                'HF_HUB_OFFLINE': '1', 'WANDB_MODE': 'disabled', 'PYTHONDONTWRITEBYTECODE': '1',
                'PYTHONPATH': '/workspace:/workspace/third_party/areal', 'OMP_NUM_THREADS': '4',
                'LD_LIBRARY_PATH': '/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64'}
    if not set(config['env']) <= set(required) | {'PATH', 'USER', 'LOGNAME'} or any(config['env'].get(k) != v for k, v in required.items()):
        raise ValueError('native probe environment differs from explicit contract')
    if not config['env'].get('PATH', '').startswith('/opt/.venv/bin:') or not all(config['env'].get(k) for k in ('USER', 'LOGNAME')):
        raise ValueError('native interpreter PATH and USER/LOGNAME required')


def preflight():
    """Run in the final child environment; this is CUDA functionality, not timing."""
    import socket
    import torch
    from areal.infra.utils.launcher import BASE_ENVIRONS
    os.environ.update(BASE_ENVIRONS)
    expected = validate_uuids(json.loads(Path('/output/gpu-uuids.json').read_text()))
    if torch.cuda.device_count() != 4:
        raise RuntimeError('CUDA must see exactly four devices')
    actual = []
    from torch.cuda import jiterator
    tiny_jit = jiterator._create_jit_fn("template <typename T> T ft_identity(T x) { return x + T(1); }")
    for i in range(4):
        props = torch.cuda.get_device_properties(i)
        actual.append(str(props.uuid))
        with torch.cuda.device(i):
            if (torch.ones(1, device=f'cuda:{i}') + 1).item() != 2:
                raise RuntimeError('CUDA computation failed')
            if tiny_jit(torch.ones(1, device=f'cuda:{i}')).item() != 2:
                raise RuntimeError('CUDA JIT failed')
            torch.cuda.synchronize()
    Path('/output/gpu-uuid-observation.json').write_text(json.dumps({'actual_raw': actual, 'expected_raw': expected}))
    normalized = match_cuda_uuids(actual, expected)
    for key in ('HOME', 'AREAL_CACHE_DIR', 'PYTORCH_KERNEL_CACHE_PATH', 'TRITON_CACHE_DIR', 'VLLM_CACHE_ROOT'):
        path = Path(os.environ[key]); path.mkdir(parents=True, exist_ok=True)
        test = path / 'ft-write-check'; test.write_text('ok'); test.unlink()
    from areal.utils.network import gethostip
    address = gethostip()
    with socket.socket() as s:
        s.bind((address, 0))
    status = Path('/proc/self/status').read_text()
    if 'CapEff:\t0000000000000000' not in status or 'NoNewPrivs:\t1' not in status:
        raise RuntimeError('restricted process capabilities not preserved')
    fd = os.pidfd_open(os.getpid()); os.close(fd)
    mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    if not any(line.split()[4] == '/workspace' and 'ro' in line.split()[5].split(',') for line in mounts):
        raise RuntimeError('workspace mount is not read-only')
    for value in ('/workspace/models/Qwen2.5-0.5B-Instruct/config.json', '/workspace/runs/diagnosis-20260910/train.jsonl'):
        with open(value, 'rb') as source:
            source.read(1)
    Path('/output/gpu-preflight.json').write_text(json.dumps({'uuids': normalized, 'uuids_raw': actual, 'ip': address,
        'torch': torch.__version__, 'status': status, 'effective_env': {k: os.environ[k] for k in sorted(GPU_ENV | set(BASE_ENVIRONS) | {'PATH', 'PYTHONPATH', 'CUDA_VISIBLE_DEVICES', 'USER', 'LOGNAME', 'OMP_NUM_THREADS'}) if k in os.environ}, 'scope': 'context, small tensor, cache writability; no performance claim'}))


def main():
    frozen = json.loads(Path('/output/source-sha256.json').read_text())
    if provenance(Path('/workspace')) != frozen:
        raise RuntimeError('source changed after launch snapshot')
    # The controller starts this method root once; exec preserves its PID and ownership.
    with Path('/output/gpu-preflight.log').open('w') as log:
        child = subprocess.Popen(['/opt/.venv/bin/python', '-m', 'scripts.ft.native_gpu', '--preflight'], stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5)
            raise
    Path('/output/gpu-preflight-exit.json').write_text(json.dumps({'pid': child.pid, 'exit_code': code}))
    if code != 0:
        raise RuntimeError('GPU preflight failed; native launcher not started')
    executable = '/opt/.venv/bin/python'
    argv = [executable, '-m', 'areal.infra.launcher.local',
            '/workspace/scripts/ft/areal_native_trainer_probe.py',
            '--config', '/output/native-trainer.yaml']
    Path('/output/native-exec.json').write_text(json.dumps({'argv': argv}))
    os.execv(executable, argv)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        preflight()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('unexpected arguments')
