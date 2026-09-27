import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ft1_chain import killed_update


def ev(event, ns, pid, uid=None):
    e = {'event': event, 'monotonic_ns': ns, 'pid': pid, 'process_incarnation': str(pid)}
    if uid:
        e['update_id'] = uid
    return e


class KilledUpdate(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / 'events.jsonl').write_text(json.dumps({'kind': 'signal_sent', 'controller_monotonic_ns': 50}) + '\n')
        self.pilot = [ev('batch_taken', 10, 1), ev('train_batch', 11, 1, 'u1'), ev('optimizer_end', 12, 1, 'u1'),
                      ev('batch_taken', 20, 1), ev('train_batch', 21, 1, 'u2'),
                      ev('batch_taken', 60, 2), ev('train_batch', 61, 2, 'u3'), ev('optimizer_end', 62, 2, 'u3')]

    def split(self, pilot):
        return ([e for e in pilot if e['event'] == 'batch_taken'], [e for e in pilot if e['event'] == 'train_batch'],
                [e for e in pilot if e['event'] == 'optimizer_end'])

    def test_accepts_single_kill_during_train(self):
        b, t, a = self.split(self.pilot)
        got = killed_update(self.root, {'scenario': 'F4T'}, self.pilot, b, t, a)
        self.assertEqual(got[1]['update_id'], 'u2')
        self.assertIs(got[0], self.pilot[3])

    def test_no_open_update(self):
        pilot = self.pilot[:3]
        self.assertIsNone(killed_update(self.root, {'scenario': 'F4T'}, pilot, *self.split(pilot)))

    def test_rejects_other_scenarios(self):
        with self.assertRaises(AssertionError):
            killed_update(self.root, {'scenario': 'F2'}, self.pilot, *self.split(self.pilot))

    def test_rejects_open_update_not_last_in_process(self):
        pilot = self.pilot[:5] + [ev('batch_taken', 30, 1)] + self.pilot[5:]
        with self.assertRaises(AssertionError):
            killed_update(self.root, {'scenario': 'F4T'}, pilot, *self.split(pilot))

    def test_rejects_signal_before_open_update(self):
        (self.root / 'events.jsonl').write_text(json.dumps({'kind': 'signal_sent', 'controller_monotonic_ns': 15}) + '\n')
        with self.assertRaises(AssertionError):
            killed_update(self.root, {'scenario': 'F4T'}, self.pilot, *self.split(self.pilot))

    def test_rejects_two_open_updates(self):
        pilot = self.pilot[:-1]
        with self.assertRaises(AssertionError):
            killed_update(self.root, {'scenario': 'F4T'}, pilot, *self.split(pilot))


if __name__ == '__main__':
    unittest.main()
