#!/usr/bin/env python3
"""Build a diagnostic-only Slime overlay; reject drift, never edit baseline files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(('overlay anchor drift', old, text.count(old)))
    return text.replace(old, new, 1)


def build(destination):
    spec = json.loads((ROOT/'docs/experiments/rewardtxn-ablation-20260911/design.json').read_text())
    for name, expected in spec['source_sha256_at_design'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    shutil.copytree(ROOT/'third_party/slime', destination,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc', '.pytest_cache'))
    edits = {}
    p = destination/'train_async.py'
    s = p.read_text()
    s = replace_once(s, 'import ray\n', 'import ray\nimport ablation_runtime as diagnostic\n')
    s = replace_once(s, '    configure_logger()\n', '    diagnostic.driver_start(args)\n    configure_logger()\n')
    s = replace_once(s, 'range(args.start_rollout_id, args.num_rollout)',
                     'range(args.start_rollout_id, diagnostic.training_steps(args))')
    s = replace_once(s, '        # Sync the last generation\n',
                     '        diagnostic.driver_check(rollout_id)\n        # Sync the last generation\n')
    s = replace_once(s, 'rollout_id + 1 < args.num_rollout', 'rollout_id + 1 < diagnostic.training_steps(args)')
    # Only shorten endpoint/saves; args.num_rollout remains 500 for optimizer schedule.
    s = s.replace('args.save_interval, num_rollout_per_epoch, args.num_rollout',
                  'args.save_interval, num_rollout_per_epoch, diagnostic.training_steps(args)')
    s = s.replace('rollout_id == args.num_rollout - 1', 'rollout_id == diagnostic.training_steps(args) - 1')
    s = replace_once(s, '    ray.get(rollout_manager.dispose.remote())',
                     '    diagnostic.driver_finalize(rollout_manager)')
    s = replace_once(s, '    train(args)\n',
                     '    try:\n        train(args)\n        diagnostic.driver_close()\n    except BaseException as exc:\n        diagnostic.driver_failed(exc)\n        raise\n')
    p.write_text(s); edits[str(p.relative_to(destination))] = hashlib.sha256(p.read_bytes()).hexdigest()
    p = destination/'slime/ray/rollout.py'; s=p.read_text()
    s=replace_once(s, '        self.args = args\n',
                  '        self.args = args\n        import ablation_runtime\n        ablation_runtime.manager_start()\n')
    s=replace_once(s, '    def dispose(self):\n', '''    def diagnostic_finalize(self):
        import ablation_runtime
        return ablation_runtime.manager_finalize()

    def diagnostic_exit(self, permit):
        import ablation_runtime
        ablation_runtime.manager_exit(permit)

    def diagnostic_close(self):
        self.dispose()
        import ablation_runtime
        return ablation_runtime.manager_close()

    def dispose(self):
''')
    s=replace_once(s, '        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)',
                  '        self._save_debug_rollout_data(data, rollout_id=rollout_id, evaluation=False)\n        import ablation_runtime\n        ablation_runtime.consumed(data, rollout_id)')
    p.write_text(s); edits[str(p.relative_to(destination))] = hashlib.sha256(p.read_bytes()).hexdigest()
    p=destination/'slime/rollout/fully_async_rollout.py'; s=p.read_text()
    s=replace_once(s, 'import asyncio\n', 'import asyncio\nimport ablation_runtime as diagnostic\n')
    s=replace_once(s, '        active_tasks: set[asyncio.Task] = set()\n',
                  '        active_tasks: set[asyncio.Task] = set()\n        diagnostic.worker_start(self, active_tasks)\n')
    s=replace_once(s, '                            generate_and_rm_group(\n',
                  '                            diagnostic.generate_group(generate_and_rm_group,\n')
    s=replace_once(s, '                logger.exception("fully-async loop iteration error: %s", e)',
                  '                diagnostic.worker_failed(e)\n                logger.exception("fully-async loop iteration error: %s", e)')
    start=s.index('        if active_tasks:\n            logger.info(')
    end=s.index('\n    def _make_done_cb',start)
    s=s[:start]+'        await diagnostic.worker_finalize(self, active_tasks)\n'+s[end:]
    s=replace_once(s, '                result = done_task.result()\n            except Exception:  # noqa: BLE001\n', '                result = done_task.result()\n            except asyncio.CancelledError:\n                return\n            except Exception:  # noqa: BLE001\n')
    s=replace_once(s, '            if not isinstance(result, list):\n',
                  '            if not isinstance(result, list):\n                diagnostic.worker_failed(RuntimeError("unexpected group result"))\n')
    s=replace_once(s, '                    self.data_buffer.add_samples([result])',
                  '                    if self.running:\n                        self.data_buffer.add_samples([result])')
    s=replace_once(s, '            self.output_queue.put((gid, result))',
                  '            diagnostic.queued(gid, result, self.output_queue.qsize())\n            self.output_queue.put((gid, result))')
    p.write_text(s); edits[str(p.relative_to(destination))] = hashlib.sha256(p.read_bytes()).hexdigest()
    p=destination/'slime/backends/megatron_utils/model.py'; s=p.read_text()
    s=replace_once(s, '    return opt_param_scheduler\n',
                  '    import ablation_runtime\n    ablation_runtime.record_scheduler(args, opt_param_scheduler)\n    return opt_param_scheduler\n')
    p.write_text(s); edits[str(p.relative_to(destination))] = hashlib.sha256(p.read_bytes()).hexdigest()
    p=destination/'slime/backends/sglang_utils/sglang_engine.py'; s=p.read_text()
    s=replace_once(s, '        self.node_rank = server_args_dict["node_rank"]',
                  '        import ablation_runtime\n        ablation_runtime.record_engine(self.rank, server_args_dict)\n        self.node_rank = server_args_dict["node_rank"]')
    p.write_text(s); edits[str(p.relative_to(destination))] = hashlib.sha256(p.read_bytes()).hexdigest()
    # Preserve the existing launch recipe; record the Ray job wait result before EXIT cleanup.
    shell=(ROOT/'scripts/day2_slime_train.sh').read_text()
    shell=replace_once(shell, 'ray job submit --address=', 'set +e\nray job submit --address=')
    shell += '\nABLATION_JOB_RC=$?\nexport ABLATION_JOB_RC\npython3 -c \'import os; from ablation_runtime import write_json,run_dir; write_json(run_dir()/"driver_job_exit.json", {"exitcode":int(os.environ["ABLATION_JOB_RC"]),"waited":True,"kind":"ray_job_submit_wait"})\'\nexit "$ABLATION_JOB_RC"\n'
    p=destination/'diagnostic_day2.sh'; p.write_text(shell)
    edits[str(p.relative_to(destination))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (destination/'diagnostic_overlay.json').write_text(json.dumps({'baseline':spec['source_sha256_at_design'], 'overlay':edits},indent=2)+'\n')
    return edits


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('destination',type=Path)
    print(json.dumps(build(parser.parse_args().destination),indent=2))
