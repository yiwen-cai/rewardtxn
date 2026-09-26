"""CPU method-owned batch identity and prepared-not-applied intents.

No optimizer success, consumption, GPU ownership, adoption or commit authority.
"""
import asyncio
import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import struct
import re
import uuid

import torch
from areal import workflow_context

from . import state
from .replay import BlobStore, _decode, _encode, _read, _publish, _directory
from .rlvr_replay import ArtifactError, digest

KEYS = tuple('_r_receipt_' + str(i) for i in range(4))


class PreparedBoundary(RuntimeError):
    """Explicit CPU stop, never a successful optimizer result."""


def strip_identity(data):
    """Independent container; tensors are not mutated or copied unnecessarily."""
    if isinstance(data, list):
        return [strip_identity(item) for item in data]
    return {key: value for key, value in data.items() if key not in KEYS}


def tensor_hashes(data):
    return {key: {'dtype': str(value.dtype), 'shape': list(value.shape),
                  'sha256': hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
            for key, value in data.items() if isinstance(value, torch.Tensor)}


class IdentityBridge:
    def __init__(self, workflow, root, *, policy_version):
        if type(policy_version) is not int or policy_version < 0:
            raise ArtifactError("explicit CPU policy version required")
        self.policy_version = policy_version
        self.workflow, self.owner = workflow, workflow.owner
        self.root = Path(root).absolute()
        _directory(self.root)
        self.blobs = BlobStore(self.root / 'blobs')
        self.io = ThreadPoolExecutor(max_workers=1, thread_name_prefix='r-identity')
        self.intent = None
        # Per-update memo of fully verified identity records (training thread
        # only). Authority is still re-checked on every use via _current().
        self._resolved = {}

    def close(self):
        self.io.shutdown(wait=True)

    def _blob(self, store, sha):
        if not isinstance(sha, str) or re.fullmatch('[0-9a-f]{64}', sha) is None:
            raise ArtifactError('invalid identity blob reference')
        raw = _read(store.root / sha)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ArtifactError('identity blob hash mismatch')
        return _decode(raw)

    def authorize(self, data):
        """Called by the one loader producer before delivering a draw item."""
        for index in range(data['_r_draw']['k']):
            sample = data['_r_draw']['group_id'] + ':' + str(index)
            with state._locked(self.owner) as control:
                attempt = copy.deepcopy(control['attempts'].get(sample))
            if attempt is None:
                attempt = state.authorize_attempt(self.owner, sample, None, uuid.uuid4().hex,
                                                  expected_policy_version=self.policy_version)
            if (attempt['epoch'] != self.owner.epoch or attempt['owner_nonce'] != self.owner.nonce
                    or attempt['versions']['policy_version'] != self.policy_version):
                raise ArtifactError('old owner authorization')
            self.workflow.attempts[sample] = attempt

    def _current(self, record):
        receipt = record['receipt']; attempt = receipt['attempt']
        with state._locked(self.owner) as control:
            if (control['scope'] != 'cpu_contract' or control['attempts'].get(attempt['sample']) != attempt
                    or control['accepted'].get(attempt['sample']) != receipt
                    or attempt['epoch'] != self.owner.epoch or attempt['owner_nonce'] != self.owner.nonce):
                raise ArtifactError('inactive receipt authorization')

    def _resolve(self, sha):
        record = self._blob(self.blobs, sha)
        if set(record) != {'schema', 'receipt', 'data', 'stages'} or record['schema'] != 1:
            raise ArtifactError('invalid identity record')
        self._current(record)
        stages = {name: (self._blob(self.workflow.blobs, value), value)
                  for name, value in record['stages'].items()}
        if set(stages) != {'response', 'reward', 'tensor'}:
            raise ArtifactError('missing original artifacts')
        binding = stages['tensor'][0]['binding']
        if (binding['schema'] != 2 or binding['attempt'] != record['receipt']['attempt']
                or binding['data_sha256'] != digest(record['data'])):
            raise ArtifactError('receipt/input binding mismatch')
        for name, (stage, stage_sha) in stages.items():
            payload_key = {'response': 'response_sha256', 'reward': 'reward_sha256', 'tensor': 'tensor_input_sha256'}[name]
            if (stage['stage'] != name or stage['binding'] != binding
                    or record['receipt']['payload'][payload_key] != stage_sha):
                raise ArtifactError('receipt/artifact binding mismatch')
        response = (stages['response'][0]['payload'], stages['response'][1])
        reward = (stages['reward'][0]['payload'], stages['reward'][1])
        tensor = (stages['tensor'][0]['payload'], stages['tensor'][1])
        context = {'binding': binding, 'data': record['data'], 'input_ids': response[0]['input_tokens']}
        tensors = self.workflow._load_tensors(context, tensor, response, reward)
        self._current(record)
        return record, tensors

    def resolve_verified(self, sha):
        """Training-side _resolve memoized within one update."""
        found = self._resolved.get(sha)
        if found is None:
            found = self._resolved[sha] = self._resolve(sha)
        else:
            self._current(found[0])
        return found

    def _attach(self, data, index, tensors):
        sample = data['_r_draw']['group_id'] + ':' + str(index)
        with state._locked(self.owner) as control:
            receipt = copy.deepcopy(control['accepted'][sample])
        p = receipt['payload']
        record = {'schema': 1, 'receipt': receipt, 'data': copy.deepcopy(data),
                  'stages': {'response': p['response_sha256'], 'reward': p['reward_sha256'],
                             'tensor': p['tensor_input_sha256']}}
        sha = self.blobs.put(_encode(record))['sha256']
        _, original = self._resolve(sha)
        if tensor_hashes(tensors) != tensor_hashes(original):
            raise ArtifactError('workflow tensor differs from accepted artifact')
        words = struct.unpack('>4q', bytes.fromhex(sha))
        result = dict(tensors)
        for key, word in zip(KEYS, words):
            result[key] = torch.where(tensors['attention_mask'],
                                      torch.full_like(tensors['input_ids'], word, dtype=torch.int64), 0)
        return result, record

    async def arun_episode(self, engine, data):
        data = copy.deepcopy(data)
        index = workflow_context.get().sample_idx
        tensors = await self.workflow.arun_episode(engine, data)
        result, record = await asyncio.get_running_loop().run_in_executor(self.io, self._attach, data, index, tensors)
        # Current-owner/CAS re-read after await, on the I/O worker so the event
        # loop never blocks on mutation.lock.
        await asyncio.get_running_loop().run_in_executor(self.io, self._current, record)
        return result

    def validate_rows(self, data):
        mask = data['attention_mask']
        if any(isinstance(value, torch.Tensor) and value.device.type != 'cpu' for value in data.values()):
            raise ArtifactError('CPU bridge rejects device tensors')
        if mask.dtype != torch.bool or mask.ndim != 2 or not mask.any(dim=1).all():
            raise ArtifactError('invalid identity mask')
        for key in KEYS:
            if key not in data or data[key].dtype != torch.int64 or data[key].shape != mask.shape:
                raise ArtifactError('missing/malformed identity carrier')
            if torch.any(data[key][~mask] != 0):
                raise ArtifactError('identity in padding')
        rows = []
        for row in range(mask.shape[0]):
            words = []
            for key in KEYS:
                values = data[key][row][mask[row]]
                if not torch.all(values == values[0]):
                    raise ArtifactError('nonconstant row identity')
                words.append(values[0].item())
            sha = struct.pack('>4q', *words).hex()
            record, original = self.resolve_verified(sha)
            # These structural inputs stay unchanged through PPO advantage transforms.
            for key in ('input_ids', 'versions'):
                actual = data[key][row][mask[row]].detach().cpu()
                expected = original[key][0][original['attention_mask'][0]]
                if not torch.equal(actual, expected):
                    raise ArtifactError('identity does not match actual row tokens/version')
            rows.append({'identity': sha, 'sample': record['receipt']['attempt']['sample'],
                         'attempt': record['receipt']['attempt'], 'draw': record['data']['_r_draw']})
        return rows

    def _row_fingerprints(self, data):
        # PPO removes these three diagnostics immediately before train_batch.
        ignored = set(KEYS) | {'rewards', 'tot_rewards', 'kl_rewards'}
        mask = data['attention_mask']
        fingerprints = []
        for row in range(mask.shape[0]):
            values = {}
            for key, value in data.items():
                if key in ignored:
                    continue
                if not isinstance(value, torch.Tensor) or value.shape[0] != mask.shape[0]:
                    raise ArtifactError('unsupported PPO batch field')
                if value.ndim == 2 and value.shape == mask.shape:
                    values[key] = value[row][mask[row]]
                elif value.ndim == 1:
                    values[key] = value[row:row+1]
                else:
                    raise ArtifactError('unsupported PPO tensor shape')
            fingerprints.append(tensor_hashes(values))
        return fingerprints

    def begin_update(self, trajectories, config):
        """Validate complete K once, before native PPO splits microbatches."""
        self._resolved = {}
        rows = [row for trajectory in trajectories for row in self.validate_rows(trajectory)]
        samples = [row['sample'] for row in rows]
        if len(samples) != len(set(samples)):
            raise ArtifactError('duplicate update sample')
        groups = {}
        for row in rows:
            draw = row['draw']; groups.setdefault(draw['group_id'], []).append(row)
        for group, members in groups.items():
            k = members[0]['draw']['k']
            if any(r['draw']['k'] != k for r in members):
                raise ArtifactError('inconsistent group K')
            if {r['sample'] for r in members} != {group + ':' + str(i) for i in range(k)}:
                raise ArtifactError('incomplete update group')
        # Logical ID excludes physical attempt and row order; metadata records both.
        logical = digest({'samples': sorted(samples), 'config': config})
        self.intent = {'schema': 1, 'status': 'prepared_not_applied', 'parent_generation': None,
                       'logical_update_id': logical, 'physical_invocation_id': uuid.uuid4().hex,
                       'config': copy.deepcopy(config), 'rows': rows,
                       'input_tensors': [tensor_hashes(strip_identity(t)) for t in trajectories],
                       'row_tensors': {row['identity']: fp for row, fp in zip(rows,
                           [fp for t in trajectories for fp in self._row_fingerprints(t)])}}
        return copy.deepcopy(self.intent)

    def at_train_batch(self, data):
        path, _ = self.prepare_train_batch(data)
        raise PreparedBoundary(str(path))

    def prepare_train_batch(self, data):
        """Persist the checked intent; the caller must bind actual backend events."""
        if self.intent is None:
            raise ArtifactError('update boundary not prepared')
        rows = self.validate_rows(data)
        expected = {r['identity'] for r in self.intent['rows']}
        ids = [r['identity'] for r in rows]
        if len(ids) != len(set(ids)) or not set(ids) <= expected:
            raise ArtifactError('microbatch is not an authorized update subset')
        if self.intent['config'].get('ppo_n_minibatches') == 1 and set(ids) != expected:
            raise ArtifactError('single minibatch omitted update members')
        for row, fingerprint in zip(rows, self._row_fingerprints(data)):
            if self.intent['row_tensors'][row['identity']] != fingerprint:
                raise ArtifactError('PPO input changed after update boundary')
        clean = strip_identity(data)
        record = dict(self.intent, microbatch_rows=rows, actual_train_tensors=tensor_hashes(clean))
        path = self.root / ('intent-' + self.intent['physical_invocation_id'] + '.json')
        with state._locked(self.owner) as control:
            if any(control['attempts'].get(row['sample']) != row['attempt'] for row in self.intent['rows']):
                raise ArtifactError('update authorization changed before intent publication')
            _publish(path, _encode(record), 'update_intent')
        return path, clean
