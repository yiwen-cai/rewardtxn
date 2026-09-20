"""Deterministic CPU loader/crash fixture; no training or commit authority."""
import json
import os
from pathlib import Path
import signal
import sys

from scripts.ft import replay


class Sampler:
    num_replicas, rank, shuffle, seed = 1, 0, False, 211

    def set_epoch(self, epoch):
        if epoch != 0:
            raise ValueError('fixture epoch')


class Loader:
    num_workers, batch_size, drop_last = 0, 4, True

    def __init__(self):
        self.sampler = Sampler()
        self.offset, self.calls = 0, 0

    def __len__(self): return 8
    def __iter__(self): return self

    def __next__(self):
        if self.offset == 32: raise StopIteration
        self.calls += 1
        batch = [{'source_row_id': i, 'messages': [{'role': 'user', 'content': 'same prompt'}]}
                 for i in range(self.offset, self.offset + 4)]
        self.offset += 4
        return batch

    def state_dict(self): return {'offset': self.offset}
    def load_state_dict(self, state): self.offset = state['offset']


def opened(root, base=None):
    return replay.DrawLoader(base or Loader(), root, 'cpu-run', 'a' * 64, 'b' * 64, k=8)


def crash(root, point):
    with opened(root) as loader:
        def fault(phase, path=None):
            if phase == point:
                if phase.endswith('_temp_written'):
                    raw = path.read_bytes()
                    path.write_bytes(raw[:max(1, len(raw) // 2)])
                Path(root).parent.joinpath('cut.json').write_text(json.dumps({'point': phase}))
                os.kill(os.getpid(), signal.SIGKILL)
        if point == 'after_genesis': fault(point)
        replay._cut = fault
        batch = next(loader)
        if point == 'after_delivery':
            Path(root).parent.joinpath('first-item.json').write_text(json.dumps(batch[0]))
            fault(point)
        raise RuntimeError('fault point not reached')


if __name__ == '__main__':
    # Keep class identity identical when parent tests reopen the fixture journal.
    from replay_fixture import crash
    crash(sys.argv[1], sys.argv[2])
