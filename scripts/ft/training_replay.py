"""Single actor adapter for retained consumption and cross-execution artifacts.

The storage owner remains a CPU ledger owner. Backend writer exclusion is a
separate contract in training_adapter, not a renamed storage scope.
"""
import copy
import uuid

from areal.api import RolloutWorkflow

from . import state
from .batch_identity import IdentityBridge
from .replay import ReplayError, _decode, _encode, _read, _publish
from .rlvr_replay import CallReturnRLVR, ArtifactError, digest


class RetainedLoader:
    """Repack whole unconsumed groups; only a verified retained chain can skip."""
    def __init__(self, draw, owner, authorize):
        self.draw, self.owner, self.authorize = draw, owner, authorize
        self.batch_size, self.sampler = draw.batch_size, draw.sampler
        with state._locked(owner) as control:
            head, tokens = state._head(owner, control)
            self.consumed = set() if head is None else set(tokens[head['generation']][1]['data']['consumed'])
        # Validate all consumed slots against the complete durable draw prefix.
        available = {s['group_id'] + ':' + str(i) for r in draw._records for s in r['slots'] for i in range(s['k'])}
        if not self.consumed <= available:
            raise ReplayError('retained consumption absent from draw journal')
        for record in draw._records:
            for slot in record['slots']:
                group = {slot['group_id'] + ':' + str(i) for i in range(slot['k'])}
                if self.consumed & group and not group <= self.consumed:
                    raise ReplayError('partial retained group')

    def __len__(self):
        return len(self.draw)

    def __iter__(self):
        batch = []
        for original in self.draw:
            for item in original:
                if item['_r_draw']['group_id'] + ':0' in self.consumed:
                    continue
                self.authorize(item)
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if batch:
            raise ReplayError('unconsumed tail cannot form a complete training batch')

    def state_dict(self):
        return self.draw.state_dict()

    def load_state_dict(self, descriptor):
        self.draw.load_state_dict(descriptor)

    def snapshot(self, additional):
        with self.draw._lock:
            self.draw._check()
            drawn, pending = [], []
            consumed = self.consumed | set(additional)
            for record in self.draw._records:
                for item in self.draw._return(record):
                    slot = item['_r_draw']
                    prompt = {k: v for k, v in item.items() if k != '_r_draw'}
                    for i in range(slot['k']):
                        sample = slot['group_id'] + ':' + str(i)
                        drawn.append(sample)
                        if sample not in consumed:
                            pending.append({'sample': sample, 'action': 'regenerate', 'prompt': prompt,
                                            'prompt_sha256': state._hash(state._bytes(prompt)), 'k': slot['k']})
            return {'source_sha256': self.draw._contract['source_sha256'], 'epoch': 0,
                    'shuffle_state': self.draw.state_dict(), 'drawn': drawn,
                    'consumed': [s for s in drawn if s in consumed], 'pending': pending, 'cursor': len(drawn)}


class TrainingRLVR(CallReturnRLVR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.origins = {}
        self.current_version = None
        self.max_lag = 0

    def valid_versions(self, context, versions):
        admission = context['binding']['attempt']['versions']['policy_version']
        current = self.current_version()
        return all(type(v) is int and admission <= v <= current and current - v <= self.max_lag
                   for v in versions)

    def bind_context(self, context):
        origin = self.origins.get(context['binding']['attempt']['sample'])
        if origin is not None:
            original = _decode(_read(self.blobs.root / origin['payload']['response_sha256']))
            context['binding']['origin_attempt'] = original['binding'].get('origin_attempt', origin['attempt'])
        return context

    def _invocation_nonce(self, context, response):
        # Adoption preserves the ORIGINAL call-return nonce; it never invents
        # an observation of the scorer being invoked in this execution.
        binding = context['binding']
        original = binding.get('origin_attempt', binding['attempt'])
        return digest({'schema': 2, 'attempt': original,
                       'input_sha256': self._scoring_input(context, response)})

    async def restore_artifacts(self, context):
        origin = self.origins.get(context['binding']['attempt']['sample'])
        if origin is None:
            return
        def adopt():
            # Only accept complete, previously accepted artifacts. Partial or
            # too-old work is regenerated under a fresh authorization.
            stages = {}
            for name, key in (('response', 'response_sha256'), ('reward', 'reward_sha256'),
                              ('tensor', 'tensor_input_sha256')):
                sha = origin['payload'][key]
                record = _decode(_read(self.blobs.root / sha))
                if digest(record) != sha or record['stage'] != name:
                    raise ArtifactError('adoption artifact hash/schema mismatch')
                stages[name] = record
            binding = stages['response']['binding']
            if (any(s['binding'] != binding for s in stages.values())
                    or binding['attempt'] != origin['attempt']
                    or binding['data_sha256'] != digest(context['data'])
                    or binding['tokenizer_sha256'] != self.tokenizer_sha256
                    or binding['request_gconfig_sha256'] != context['binding']['request_gconfig_sha256']
                    or binding['run'] != context['binding']['run']):
                raise ArtifactError('adoption lineage mismatch')
            old_context = dict(context, binding=binding)
            pairs = {name: (r['payload'], origin['payload'][key]) for name, r, key in (
                ('response', stages['response'], 'response_sha256'),
                ('reward', stages['reward'], 'reward_sha256'),
                ('tensor', stages['tensor'], 'tensor_input_sha256'))}
            tensors = self._load_tensors(old_context, pairs['tensor'], pairs['response'], pairs['reward'])
            # Repeat adoptions retain the first scorer invocation's authority.
            if binding.get('origin_attempt') != context['binding'].get('origin_attempt') and 'origin_attempt' in binding:
                raise ArtifactError('nested adoption origin mismatch')
            response = self._put(context, 'response', pairs['response'][0])
            reward = self._put(context, 'reward', dict(pairs['reward'][0], response_sha256=response[1]))
            self._put(context, 'tensor', self._tensor_payload(tensors, response, reward))
            _publish(context['directory'] / 'adoption.json', _encode({'origin': origin,
                'destination': context['binding']['attempt']}), 'adoption')
        await self._io(context, adopt)


class TrainingBridge(IdentityBridge, RolloutWorkflow):
    def __init__(self, *args, version, max_lag, **kwargs):
        super().__init__(*args, **kwargs)
        self.version, self.max_lag = version, max_lag

    def authorize(self, data):
        current_version = self.version()
        for index in range(data['_r_draw']['k']):
            sample = data['_r_draw']['group_id'] + ':' + str(index)
            with state._locked(self.owner) as control:
                old = copy.deepcopy(control['attempts'].get(sample))
                receipt = copy.deepcopy(control['accepted'].get(sample))
            if old is not None and old['epoch'] == self.owner.epoch:
                self.workflow.attempts[sample] = old
                continue
            reusable = receipt is not None and receipt['attempt'] == old
            if reusable:
                sha = receipt['payload']['response_sha256']
                response = _decode(_read(self.workflow.blobs.root / sha))
                if digest(response) != sha or response['binding']['attempt'] != old:
                    raise ArtifactError('corrupt historical response')
                versions = response['payload']['output_versions']
                reusable = bool(versions) and all(type(v) is int and 0 <= current_version - v <= self.max_lag
                                                 for v in versions)
            version = old['versions']['policy_version'] if reusable else current_version
            attempt = state.authorize_attempt(self.owner, sample, None if old is None else old['attempt'],
                                              uuid.uuid4().hex, expected_policy_version=version)
            if reusable:
                self.workflow.origins[sample] = receipt
            self.workflow.attempts[sample] = attempt

    def validate_rows(self, data):
        # Native PPO moves trajectories to the actor device before train_batch.
        # Validation explicitly stages a copy; identity carriers never enter
        # the model. This verification overhead belongs to the method.
        import torch
        return super().validate_rows({k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                      for k, v in data.items()})
