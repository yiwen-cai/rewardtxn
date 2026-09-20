"""CPU official-call-return artifacts, with synthetic generation in tests.

Legitimate 0/1 are preserved. An envelope proves only that the official scorer
returned, INCLUDING its internal timeout/error fallback zeros, not verification
success. No formal/GPU verifier semantics, consumed authority or adoption.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import copy
import dataclasses
import hashlib
import math
import re
from pathlib import Path

import torch
from areal import workflow_context
from areal.api import AsyncRewardWrapper, ModelResponse
from areal.workflow.rlvr import RLVRWorkflow

from .reward_return import RewardReturn, ReturningReward, input_digest
from . import state
from .replay import BlobStore, ReplayError, _decode, _directory, _encode, _publish, _read


class ArtifactError(ReplayError):
    pass


class UncertainReward(ArtifactError):
    pass


DTYPES = {'input_ids': torch.int32, 'loss_mask': torch.int32, 'logprobs': torch.float32,
          'versions': torch.int32, 'turn_ids': torch.int32, 'attention_mask': torch.bool,
          'rewards': torch.float32}


def digest(value):
    return hashlib.sha256(_encode(value)).hexdigest()


class CallReturnRLVR(RLVRWorkflow):
    """Same live CPU owner/attempt; preserve official returned zero and one."""
    def __init__(self, *, owner, root, attempts, tokenizer_sha256, reward_fn,
                 gconfig, tokenizer, timeout_seconds=15, **kwargs):
        if not callable(reward_fn):
            raise ArtifactError('explicit frozen scoring callable required')
        if re.fullmatch(r'[0-9a-f]{64}', tokenizer_sha256) is None:
            raise ArtifactError('explicit tokenizer asset hash required')
        super().__init__(reward_fn=reward_fn, gconfig=gconfig, tokenizer=tokenizer, **kwargs)
        # Public constructor only; the super-created default executor is lazy,
        # and receives no jobs. Official retry implementation remains unchanged.
        self.async_reward_fn = AsyncRewardWrapper(ReturningReward(reward_fn), max_workers=1, max_retries=0,
                                                  timeout_seconds=timeout_seconds)
        self.owner = owner
        self.attempts = copy.deepcopy(attempts)
        self.root = Path(root).absolute()
        _directory(self.root)
        self.blobs = BlobStore(self.root / 'blobs')
        self.tokenizer_sha256 = tokenizer_sha256
        self._io_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='r-cpu-artifacts')
        self._context = ContextVar('r_cpu_sample')
        self._active = set()

    def close(self):
        if self._active:
            raise ArtifactError('cannot close active workflow')
        self._io_pool.shutdown(wait=True)

    def _check(self, context):
        # Explicit project-private coupling: state has no public read-authority API.
        with state._locked(self.owner) as control:
            attempt = context['binding']['attempt']
            if (control['scope'] != 'cpu_contract' or control['attempts'].get(attempt['sample']) != attempt
                    or attempt['epoch'] != self.owner.epoch or attempt['owner_nonce'] != self.owner.nonce
                    or attempt['versions']['verifier_version'] != control['verifier_version']):
                raise ArtifactError('inactive CPU authorization; adoption is unsupported')

    async def _io(self, context, operation):
        self._check(context)
        def guarded():
            self._check(context)
            result = operation()
            self._check(context)
            return result
        result = await asyncio.get_running_loop().run_in_executor(self._io_pool, guarded)
        self._check(context)
        return result

    def _stage(self, context, name):
        path = context['directory'] / (name + '.json')
        if not path.exists() and not path.is_symlink():
            return None
        index = _decode(_read(path))
        if set(index) != {'blob'}:
            raise ArtifactError('invalid artifact index')
        record = _decode(self.blobs.get(index['blob']))
        if (set(record) != {'binding', 'stage', 'payload'} or record['binding'] != context['binding']
                or record['stage'] != name):
            raise ArtifactError('artifact binding mismatch')
        return record['payload'], index['blob']['sha256']

    def _put(self, context, name, payload):
        self._check(context)
        ref = self.blobs.put(_encode({'binding': context['binding'], 'stage': name, 'payload': payload}))
        _directory(context['directory'])
        _publish(context['directory'] / (name + '.json'), _encode({'blob': ref}), 'rlvr_' + name)
        return payload, ref['sha256']

    def _response(self, context, payload):
        required = {'origin_rid', 'input_tokens', 'output_tokens', 'output_logprobs', 'output_versions', 'stop_reason'}
        if set(payload) != required or payload['input_tokens'] != context['input_ids']:
            raise ArtifactError('response input/schema mismatch')
        n = len(payload['output_tokens'])
        version = context['binding']['attempt']['versions']['policy_version']
        if (n < 1 or len(payload['output_logprobs']) != n or len(payload['output_versions']) != n
                or any(type(t) is not int or not 0 <= t <= 2147483647 for t in payload['input_tokens'] + payload['output_tokens'])
                or any(type(v) is not int or v != version for v in payload['output_versions'])
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in payload['output_logprobs'])
                or payload['stop_reason'] not in ('stop', 'length') or not payload['origin_rid']):
            raise ArtifactError('invalid response or unsupported mixed policy versions')
        return ModelResponse(**{k: copy.deepcopy(v) for k, v in payload.items() if k != 'origin_rid'}, tokenizer=self.tokenizer)

    def _scoring_input(self, context, response):
        resp = response[0]
        return input_digest(self.tokenizer.decode(resp['input_tokens']),
                            self.tokenizer.decode(resp['output_tokens']),
                            resp['input_tokens'], resp['output_tokens'], context['data'])

    def _invocation_nonce(self, context, response):
        # Stable logical scoring invocation, NOT an official retry sequence.
        return digest({'schema': 2, 'attempt': context['binding']['attempt'],
                       'input_sha256': self._scoring_input(context, response)})

    def _validate_return(self, context, response, envelope):
        if (type(envelope) is not RewardReturn or type(envelope.schema) is not int or envelope.schema != 1
                or envelope.status != 'official_call_returned'
                or not isinstance(envelope.invocation_nonce, str)
                or re.fullmatch('[0-9a-f]{64}', envelope.invocation_nonce) is None
                or envelope.invocation_nonce != self._invocation_nonce(context, response)
                or envelope.input_sha256 != self._scoring_input(context, response)
                or envelope.verifier_sha256 != context['binding']['attempt']['versions']['verifier_version']
                or type(envelope.score) not in (int, float) or envelope.score not in (0, 1)):
            raise UncertainReward('missing or mismatched official-call-return envelope')
        return float(envelope.score)

    def _reward(self, context, response, reward):
        payload = reward[0]
        if (set(payload) != {'response_sha256', 'return'}
                or payload['response_sha256'] != response[1]
                or not isinstance(payload['return'], dict)
                or set(payload['return']) != {f.name for f in dataclasses.fields(RewardReturn)}):
            raise ArtifactError('reward/response binding or schema mismatch')
        return self._validate_return(context, response, RewardReturn(**payload['return']))

    def _validate_tensors(self, context, tensors, response, reward):
        resp = self._response(context, response[0])
        value = self._reward(context, response, reward)
        n, p = resp.input_len + resp.output_len, resp.input_len
        if set(tensors) != set(DTYPES):
            raise ArtifactError('tensor fields differ from native RLVR contract')
        for key, tensor in tensors.items():
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.layout != torch.strided
                    or tensor.dtype != DTYPES[key] or list(tensor.shape) != ([1] if key == 'rewards' else [1, n])
                    or (tensor.is_floating_point() and not torch.isfinite(tensor).all().item())):
                raise ArtifactError('invalid CPU tensor dtype/shape/value')
        # Validation only; misses always use the original RLVR tensor builder.
        expected = {'input_ids': resp.input_tokens + resp.output_tokens,
                    'logprobs': [0.0] * p + resp.output_logprobs,
                    'versions': [-1] * p + resp.output_versions,
                    'loss_mask': [0] * p + [1] * resp.output_len,
                    'turn_ids': [-1] * p + [0] * resp.output_len,
                    'attention_mask': [True] * n}
        if any(not torch.equal(tensors[k], torch.tensor(v, dtype=DTYPES[k]).unsqueeze(0)) for k, v in expected.items()) or tensors['rewards'].item() != value:
            raise ArtifactError('tensor bytes do not represent response/reward')

    def _tensor_payload(self, tensors, response, reward):
        return {'response_sha256': response[1], 'reward_sha256': reward[1],
                'fields': {k: {'dtype': str(t.dtype), 'shape': list(t.shape),
                               'raw': self.blobs.put(t.contiguous().numpy().tobytes())} for k, t in tensors.items()}}

    def _load_tensors(self, context, stage, response, reward):
        payload = stage[0]
        if (set(payload) != {'response_sha256', 'reward_sha256', 'fields'}
                or payload['response_sha256'] != response[1] or payload['reward_sha256'] != reward[1]
                or set(payload['fields']) != set(DTYPES)):
            raise ArtifactError('tensor upstream binding mismatch')
        result = {}
        for name, field in payload['fields'].items():
            if set(field) != {'dtype', 'shape', 'raw'} or field['dtype'] != str(DTYPES[name]):
                raise ArtifactError('invalid tensor descriptor')
            shape = field['shape']
            if not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape):
                raise ArtifactError('invalid tensor shape')
            raw = self.blobs.get(field['raw'])
            if len(raw) != math.prod(shape) * torch.empty((), dtype=DTYPES[name]).element_size():
                raise ArtifactError('tensor byte length mismatch')
            result[name] = torch.frombuffer(bytearray(raw), dtype=DTYPES[name]).reshape(shape).clone()
        self._validate_tensors(context, result, response, reward)
        return result

    async def arun_episode(self, engine, data):
        data = copy.deepcopy(data)
        if '_r_reward_request' in data:
            raise ArtifactError('reserved scoring control field in task data')
        draw = data['_r_draw']
        index = workflow_context.get().sample_idx
        if type(index) is not int or not 0 <= index < draw['k']:
            raise ArtifactError('native sample index outside persisted group')
        sample = f"{draw['group_id']}:{index}"
        if sample in self._active:
            raise ArtifactError('concurrent duplicate sample invocation')
        attempt = copy.deepcopy(self.attempts[sample])
        if attempt['sample'] != sample:
            raise ArtifactError('sample/authorization mismatch')
        input_ids = self.get_input_ids_fn(self.data_extract_prompt_fn(data), self.tokenizer, self.enable_thinking)
        with state._locked(self.owner) as control:
            run = {k: control[k] for k in ('run_nonce', 'config_sha256', 'scope')}
        binding = {'schema': 2, 'run': run, 'attempt': attempt, 'draw': draw, 'data_sha256': digest(data),
                   'tokenizer_sha256': self.tokenizer_sha256,
                   'request_gconfig_sha256': digest(dataclasses.asdict(self.gconfig.new(n_samples=1))),
                   'input_ids_sha256': digest(input_ids)}
        context = {'binding': binding, 'input_ids': input_ids, 'data': data,
                   'directory': self.root / 'samples' / digest([sample, attempt['epoch'], attempt['owner_nonce'], attempt['attempt']])}
        self._check(context)
        self._active.add(sample)
        token = self._context.set(context)
        try:
            response, reward, tensor = await self._io(context, lambda: tuple(self._stage(context, s) for s in ('response', 'reward', 'tensor')))
            if (reward is not None or tensor is not None) and response is None or tensor is not None and reward is None:
                raise ArtifactError('upstream artifact missing')
            if tensor is not None:
                result = await self._io(context, lambda: self._load_tensors(context, tensor, response, reward))
            else:
                result = await super().arun_episode(engine, data)
                self._check(context)
                response, reward = await self._io(context, lambda: (self._stage(context, 'response'), self._stage(context, 'reward')))
                def persist():
                    self._validate_tensors(context, result, response, reward)
                    return self._put(context, 'tensor', self._tensor_payload(result, response, reward))
                tensor = await self._io(context, persist)
            payload = {'response_sha256': response[1], 'reward_sha256': reward[1], 'tensor_input_sha256': tensor[1],
                       **attempt['versions']}
            await self._io(context, lambda: state.accept_result(self.owner, attempt, payload))
            return result
        finally:
            self._context.reset(token)
            self._active.remove(sample)

    async def _collect_samples(self, engine, req, prompt_str, task_data):
        context = self._context.get()
        outer = self
        class ResponseSource:
            async def agenerate(self, request):
                if request.input_ids != context['input_ids']:
                    raise ArtifactError('native request input changed')
                cached = await outer._io(context, lambda: outer._stage(context, 'response'))
                if cached is not None:
                    return outer._response(context, cached[0])
                resp = await engine.agenerate(request)
                outer._check(context)
                if resp.input_images or resp.processor is not None or resp.routed_experts is not None:
                    raise ArtifactError('only plain text responses supported')
                payload = {k: copy.deepcopy(getattr(resp, k)) for k in
                           ('input_tokens', 'output_tokens', 'output_logprobs', 'output_versions', 'stop_reason')}
                payload['origin_rid'] = request.rid
                outer._response(context, payload)
                await outer._io(context, lambda: outer._put(context, 'response', payload))
                return resp
        result = await super()._collect_samples(ResponseSource(), req, prompt_str, task_data)
        self._check(context)
        return result

    async def _compute_rewards(self, resp, prompt_str, task_data):
        context = self._context.get()
        response, reward = await self._io(context, lambda: (self._stage(context, 'response'), self._stage(context, 'reward')))
        if reward is not None:
            return self._reward(context, response, reward)
        nonce = self._invocation_nonce(context, response)
        request = {'invocation_nonce': nonce, 'input_sha256': self._scoring_input(context, response),
                   'verifier_sha256': context['binding']['attempt']['versions']['verifier_version']}
        envelope = await super()._compute_rewards(resp, prompt_str, dict(task_data, _r_reward_request=request))
        self._check(context)
        self._validate_return(context, response, envelope)
        payload = {'response_sha256': response[1], 'return': dataclasses.asdict(envelope)}
        reward = await self._io(context, lambda: self._put(context, 'reward', payload))
        return self._reward(context, response, reward)
