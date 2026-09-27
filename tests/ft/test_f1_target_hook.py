"""F1 trainer marker follows the frozen source row and nonzero task."""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch


class F1TargetHook(unittest.TestCase):
    def test_late_target_and_wrong_task(self):
        from areal.workflow.rlvr import RLVRWorkflow
        import areal
        from scripts.ft.ft1_f1_trainer import install

        frozen = json.loads(Path('docs/experiments/rewardtxn-ft-20260916/ft1-f1-target-2602.json').read_text())

        async def original(self, engine, req, prompt_str, task_data):
            return 'returned'

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'f1-target.json').write_text(json.dumps(frozen))
            observer = types.SimpleNamespace(observe=lambda *args, **kwargs: None)
            context = types.SimpleNamespace(task_id=20)
            request = types.SimpleNamespace(input_ids=frozen['input_tokens'])
            with patch.dict(os.environ, {'FT_CONTROL_SOCKET': str(root / 'control.sock'), 'FT_RUN_NONCE': 'test'}), \
                    patch.object(RLVRWorkflow, '_collect_samples', original), \
                    patch.object(areal.workflow_context, 'get', lambda: context):
                install(observer)
                result = asyncio.run(RLVRWorkflow._collect_samples(
                    None, None, request, '', {'source_row_id': 2602}))
                self.assertEqual(result, 'returned')
                observer.observe('generation_complete', source_row_id=2602, task_id=20,
                                 input_tokens=frozen['input_tokens'], sample_idx=0)
                active = json.loads((root / 'f1-active.json').read_text())
                complete = json.loads((root / 'f1-complete.json').read_text())
                self.assertEqual((active['source_row_id'], active['task_id']), (2602, 20))
                self.assertEqual(complete['sample_idx'], 0)
                context.task_id = 0
                with self.assertRaisesRegex(AssertionError, 'target task drift'):
                    asyncio.run(RLVRWorkflow._collect_samples(
                        None, None, request, '', {'source_row_id': 2602}))


if __name__ == '__main__':
    unittest.main()
