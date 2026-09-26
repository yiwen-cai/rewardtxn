"""F4' cut: trainer ready on the 4th returned target score with all 8 entered (CPU, container)."""
import asyncio
import types
import unittest
from unittest.mock import patch


class Killed(Exception):
    pass


class FakeClient:
    instances = []

    def __init__(self, role, event_id=None, status='pending'):
        self.role, self.event_id, self.incarnation = role, event_id, 'inc'
        self.injection = {'event_id': event_id, 'status': FakeClient.status}
        self.readies = []
        FakeClient.instances.append(self)

    def ready(self, event_id, evidence):
        self.readies.append((event_id, evidence))

    def wait_release(self, event_id):
        raise Killed('killed')  # stands in for SIGKILL


class F4THook(unittest.TestCase):
    def run_group(self, status='pending', entered_before_fourth=8, row=2602):
        from areal.workflow.rlvr import RLVRWorkflow
        import areal
        from scripts.ft import ft1_fault_hooks, descendants
        FakeClient.instances, FakeClient.status = [], status
        events = []
        context = {}
        gates = [asyncio.Event() for _ in range(8)]

        async def original(self, resp, prompt_str, task_data):
            await gates[task_data['idx']].wait()
            return task_data['idx']

        with patch.object(RLVRWorkflow, '_compute_rewards', original), \
                patch.object(descendants, 'Client', FakeClient), \
                patch.object(areal.workflow_context, 'get', lambda: context['ctx']):
            frozen = ft1_fault_hooks.contract('F4T')
            ft1_fault_hooks.install_f4t(frozen, lambda event, **fields: events.append((event, fields)))
            hooked = RLVRWorkflow._compute_rewards

            async def one(idx):
                context['ctx'] = types.SimpleNamespace(task_id=7, sample_idx=idx)
                return await hooked(None, None, None, {'source_row_id': row, 'idx': idx})

            async def main():
                tasks = []
                for idx in range(entered_before_fourth):
                    tasks.append(asyncio.ensure_future(one(idx)))
                    await asyncio.sleep(0)
                outcomes = []
                for idx in range(4):
                    gates[idx].set()
                    try:
                        outcomes.append(await tasks[idx])
                    except Killed:
                        outcomes.append('killed')
                for task in tasks[4:]:
                    task.cancel()
                return outcomes

            outcomes = asyncio.run(main())
        return outcomes, events, FakeClient.instances[0]

    def test_fires_on_fourth_return_with_all_entered(self):
        outcomes, events, client = self.run_group()
        self.assertEqual(outcomes, [0, 1, 2, 'killed'])
        self.assertEqual(client.readies, [('ft1-f4t-trainer', client.readies[0][1])])
        self.assertEqual(client.readies[0][1]['source_row_id'], 2602)
        ready = [f for e, f in events if e == 'fault_ready']
        self.assertEqual(ready[0]['entered_samples'], list(range(8)))

    def test_missed_when_not_all_generated(self):
        outcomes, events, client = self.run_group(entered_before_fourth=5)
        self.assertEqual(outcomes, [0, 1, 2, 3])
        self.assertEqual(client.readies, [])
        self.assertIn('fault_cut_missed', [e for e, _ in events])

    def test_already_fired_restart_is_disarmed(self):
        outcomes, events, client = self.run_group(status='already_fired')
        self.assertEqual(outcomes, [0, 1, 2, 3])
        self.assertEqual(client.readies, [])

    def test_other_rows_untouched(self):
        outcomes, events, client = self.run_group(row=913)
        self.assertEqual(outcomes, [0, 1, 2, 3])
        self.assertEqual(client.readies, [])


if __name__ == '__main__':
    unittest.main()
