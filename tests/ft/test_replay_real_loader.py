"""Opt-in existing-image CPU torchdata/official AReaL loader API round trip."""
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest

from scripts.ft.replay import DrawLoader, ReplayError


@unittest.skipUnless(os.environ.get('FT_REPLAY_REAL_LOADER') == '1', 'requires existing AReaL CPU image')
class RealLoaderContracts(unittest.TestCase):
    def test_official_loader_cycle_and_resume_preserve_durable_tail(self):
        from datasets import Dataset
        from torchdata.stateful_dataloader import StatefulDataLoader
        from areal.api.cli_args import TrainDatasetConfig
        from areal.utils.dataloader import create_dataloader
        from areal.utils.data import cycle_dataloader
        rows = [{'source_row_id': i, 'messages': [{'role': 'user', 'content': f'CPU prompt {i // 2}'}]}
                for i in range(32)]
        source_hash = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        config = TrainDatasetConfig(path='in-memory-cpu-fixture', type='rl', batch_size=4,
                                    shuffle=True, drop_last=True, num_workers=0)
        def base(): return create_dataloader(Dataset.from_list(rows), 0, 1, config)
        reference = list(cycle_dataloader(base(), num_cycles=1))
        self.assertEqual(len(reference), 8)
        evidence = os.environ.get('FT_REPLAY_EVIDENCE_DIR')
        if evidence: Path(evidence).mkdir(parents=True, exist_ok=True)
        context = contextlib.nullcontext(tempfile.mkdtemp(prefix='loader-', dir=evidence)) if evidence else tempfile.TemporaryDirectory()
        with context as directory:
            root = Path(directory)
            provenance = {'scope': 'real CPU loader only, no workflow/training/GPU',
                          'versions': {name: importlib.metadata.version(name) for name in ('torch', 'torchdata', 'datasets')},
                          'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                              (Path(inspect.getfile(StatefulDataLoader)), Path(inspect.getfile(create_dataloader)),
                               Path(inspect.getfile(cycle_dataloader)))}}
            (root / 'provenance.json').write_text(json.dumps(provenance, indent=2))
            def proxy(): return DrawLoader(base(), root / 'draw', 'real-cpu-run', source_hash, 'c'*64, k=8)
            def clean(batch): return [{k: v for k, v in item.items() if k != '_r_draw'} for item in batch]
            with proxy() as loader:
                self.assertEqual((len(loader), loader.batch_size), (8, 4))
                generator = cycle_dataloader(loader, num_cycles=1)
                batches = [next(generator)]
                snapshot = loader.state_dict()
                batches.extend([next(generator), next(generator)])
                self.assertEqual([clean(b) for b in batches], reference[:3])
            with proxy() as recovered:
                recovered.load_state_dict(snapshot)
                generator = cycle_dataloader(recovered, num_cycles=1)
                replayed = [next(generator) for _ in range(3)]
                self.assertEqual(replayed, batches)
                remaining = list(generator)
                self.assertEqual([clean(b) for b in remaining], reference[3:])
                self.assertEqual(remaining[0][0]['_r_draw']['occurrence'], 12)
                self.assertEqual(recovered.state_dict()['wal_sequence'], 8)
                with self.assertRaises(ReplayError): recovered.sampler.set_epoch(1)
            (root / 'result.json').write_text(json.dumps({'passed': True, 'draw_batches': 8,
                'snapshot_prefix': 1, 'recovered_durable_prefix': 3, 'replayed_groups': 12,
                'fresh_groups': 20, 'consumed_authority': False, 'gpu': False}))


if __name__ == '__main__': unittest.main()
