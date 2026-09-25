"""The diagnostic wrapper records the failing tensor without replacing MCore's copy."""
from functools import partial
import json
from types import SimpleNamespace

import pytest
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

from areal.engine.megatron_utils.checkpointer import _diagnose_async_preload, _minimal_blocking_d2h


def test_dcp_trace_names_original_failing_tensor(tmp_path):
    class FailingTensor:
        shape = (8, 16)
        dtype = 'torch.float32'
        device = 'cuda:0'

        def stride(self):
            return (16, 1)

        def storage_offset(self):
            return 0

        def data_ptr(self):
            return 1234

        def to(self, *_args, **_kwargs):
            raise RuntimeError('synthetic copy failure')

    item = SimpleNamespace(index=SimpleNamespace(fqn='optimizer.exp_avg', offset=(0,)))
    buckets = [('file', 'key', ([], [(item, FailingTensor())]))]
    original = partial(FileSystemWriterAsync.preload_tensors, buckets, False)
    request = AsyncRequest(None, (0, buckets, None), [], preload_fn=original)
    path = tmp_path / 'dcp-diag.jsonl'
    wrapped = _diagnose_async_preload(request, str(path), False)
    assert wrapped.async_fn_args is request.async_fn_args
    with pytest.raises(RuntimeError, match='synthetic copy failure'):
        wrapped.preload_fn()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    failure = next(record for record in records if record['event'] == 'preload_exception')
    assert failure['key'] == 'optimizer.exp_avg'
    assert failure['shape'] == [8, 16]
    assert failure['stride'] == [16, 1]
    assert failure['device'] == 'cuda:0'
    assert records[-1]['event'] == 'preload_failed'


def test_blocking_copy_keeps_async_request_and_payload(tmp_path, monkeypatch):
    calls = []

    class Tensor:
        def to(self, device, *, non_blocking):
            calls.append((device, non_blocking))
            return 'same-cpu-value'

    monkeypatch.setattr('torch.cuda.synchronize', lambda: None)
    item = SimpleNamespace(index=SimpleNamespace(fqn='optimizer.exp_avg', offset=(0,)))
    buckets = [('file', 'key', (['metadata'], [(item, Tensor())]))]
    original = partial(FileSystemWriterAsync.preload_tensors, buckets, True)
    request = AsyncRequest(None, (0, buckets, None), [], preload_fn=original)
    path = tmp_path / 'blocking-copy.jsonl'
    wrapped = _diagnose_async_preload(request, str(path), False, blocking_copy=True)
    common = _minimal_blocking_d2h(request)
    assert wrapped.async_fn_args is request.async_fn_args
    assert wrapped._replace(preload_fn=request.preload_fn) == request
    assert common._replace(preload_fn=request.preload_fn) == request
    assert wrapped.preload_fn() == common.preload_fn() == request.preload_fn()
    assert calls == [('cpu', False), ('cpu', False), ('cpu', True)]
    assert json.loads(path.read_text().splitlines()[0])['blocking_copy'] is True
