"""Bounded CPU-only real-Ray exit evidence probe; run inside the frozen image.

python tests/ablation/test_ray_exit.py --output /tmp/ray-exit-evidence.json
No GPU, training, or connection to an existing Ray cluster is used.
"""
import argparse
import json
import os
from pathlib import Path
import time


def run():
    import ray
    from google.protobuf.json_format import MessageToDict
    from ray.core.generated import gcs_pb2

    ray.init(num_cpus=2, num_gpus=0, include_dashboard=False,
             object_store_memory=100 * 1024 * 1024, log_to_driver=False)

    @ray.remote(num_cpus=0, max_restarts=0)
    class Probe:
        def identity(self):
            context = ray.get_runtime_context()
            return dict(actor_id=context.get_actor_id(),
                        worker_id=context.get_worker_id(), pid=os.getpid(),
                        starttime=Path('/proc/self/stat').read_text().split()[21])

        def finish(self, mode):
            if mode == 'normal':
                ray.actor.exit_actor()
            os._exit(1)

    results = []
    try:
        accessor = ray._private.state.state._connect_and_get_accessor()
        for mode in ['normal', 'crash', 'kill']:
            actor = Probe.remote()
            identity = ray.get(actor.identity.remote(), timeout=30)
            if mode == 'kill':
                ray.kill(actor, no_restart=True)
            else:
                actor.finish.remote(mode)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                # Private API is deliberately pinned to this image/Ray version.
                raw_actor = accessor.get_actor_info(ray.ActorID.from_hex(identity['actor_id']))
                raw_worker = accessor.get_worker_info(ray.WorkerID.from_hex(identity['worker_id']))
                a = gcs_pb2.ActorTableData.FromString(raw_actor)
                w = gcs_pb2.WorkerTableData.FromString(raw_worker)
                if a.state == gcs_pb2.ActorTableData.DEAD and not w.is_alive:
                    break
                time.sleep(.1)
            else:
                raise AssertionError(f'{mode}: GCS did not confirm death')
            row = dict(mode=mode, identity=identity,
                       actor=MessageToDict(a, preserving_proto_field_name=True),
                       worker=MessageToDict(w, preserving_proto_field_name=True))
            worker = row['worker']
            cause = row['actor'].get('death_cause', {})
            reason = cause.get('actor_died_error_context', {}).get('reason')
            accepted = (worker.get('exit_type') == 'INTENDED_USER_EXIT'
                        and 'exit_actor() is called.' in worker.get('exit_detail', '')
                        and not cause.get('oom_context') and reason == 'WORKER_DIED'
                        and a.num_restarts == 0 and w.pid == identity['pid'])
            row['accepted_as_intentional_exit'] = accepted
            assert accepted == (mode == 'normal'), row
            results.append(row)
        return dict(ray_version=ray.__version__, passed=True, cases=results)
    finally:
        ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = run()
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
