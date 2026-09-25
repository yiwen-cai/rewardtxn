"""C1a: R's parallel native_snapshot_r must equal the serial native_snapshot byte for byte.

CPU tensors only; the GPU path differs only in the .cpu() copy inside ``normalized``.
"""
import json
import unittest

import numpy as np
import torch

from scripts.ft import training_adapter


def fixture_state(seed):
    from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject
    generator = torch.Generator().manual_seed(seed)
    tensors = {f'layer{i}.weight': torch.randn(64, 32, generator=generator) for i in range(40)}
    tensors['embedding'] = torch.randn(1000, 16, generator=generator).to(torch.bfloat16)
    return {
        'model': {**tensors, 'extra': LocalNonpersistentObject(object())},
        'optimizer': {'param_groups': [{'lr': 1e-6, 'step': 3}],
                      'state': {i: {'param': torch.randn(8, generator=generator),
                                    'exp_avg': torch.zeros(8), 'exp_avg_sq': torch.ones(8), 'step': torch.tensor(3)}
                                for i in range(10)}},
        'lr_scheduler': {'last_epoch': 3, 'base_lrs': (1e-6,), 'np': np.float64(0.5)},
        'rng_state': ShardedObject('rng_state', [{
            'random_rng_state': (3, tuple(range(5)), None), 'np_rng_state': ('MT19937', np.arange(6, dtype=np.uint32), 1, 0, 0.0),
            'torch_rng_state': torch.arange(16, dtype=torch.uint8), 'cuda_rng_state': torch.arange(8, dtype=torch.uint8),
            'rng_tracker_states': {'model-parallel-rng': torch.arange(4, dtype=torch.uint8)}}], (1,), (0,)),
    }


class Engine:
    def __init__(self, seed):
        self.seed = seed
        self.checkpointer = self

    def generate_state_dict(self, with_optimizer, with_rng):
        assert with_optimizer and with_rng
        return fixture_state(self.seed)


class NativeSnapshotRTests(unittest.TestCase):
    def test_parallel_equals_serial_bytes(self):
        for seed in (0, 1, 2):
            serial = training_adapter.native_snapshot(Engine(seed))
            parallel = training_adapter.native_snapshot_r(Engine(seed))
            self.assertEqual(serial, parallel)
            encode = lambda value: json.dumps(value, sort_keys=True).encode()
            self.assertEqual(encode(serial), encode(parallel))

    def test_different_state_differs(self):
        self.assertNotEqual(training_adapter.native_snapshot_r(Engine(0)),
                            training_adapter.native_snapshot_r(Engine(1)))


if __name__ == '__main__':
    unittest.main()
