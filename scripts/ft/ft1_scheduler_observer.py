"""One-shot F1 cut in the real GPU scheduler; normal batch selection unchanged.

Arming proof may come from either:
  (1) trainer f1-complete.json (response already delivered to the trainer), or
  (2) in-scheduler sibling state: >=1 matching request finished and >=1 still
      unfinished in the same batch (covers the race where all k samples finish
      in one engine step before the trainer can write f1-complete.json).
Claim is attempted both before and after the real run_batch so (2) can fire
on the step that first completes a sibling.
"""
import fcntl
import json
import os
from pathlib import Path
import time


def attach(scheduler):
    from scripts.ft.descendants import Client, snapshot, exited
    root = Path(os.environ['FT_CONTROL_SOCKET']).parent
    attached_at = time.monotonic_ns()
    contract = json.loads((root / 'f1-target.json').read_text())
    if contract['source_row_id'] != 5518 or contract['unique_tokenized_source_rows'] != 1:
        raise RuntimeError('F1 requires frozen unique first source prompt')
    target_tokens = list(contract['input_tokens'])
    original = scheduler.run_batch

    def matching(batch):
        out = []
        for req in batch.reqs:
            ids = getattr(req, 'origin_input_ids', None)
            if ids is None:
                continue
            if list(ids) != target_tokens:
                continue
            out.append(req)
        return out

    def active_ok(active):
        if (active['ambiguous']
                or active['run_nonce'] != os.environ['FT_RUN_NONCE']
                or active['source_row_id'] != 5518
                or active['task_id'] != 0
                or active['monotonic_ns'] < attached_at):
            return False
        try:
            fd = os.pidfd_open(active['trainer']['pid'], 0)
            try:
                if snapshot(active['trainer']['pid']) != active['trainer'] or exited(fd):
                    return False
            finally:
                os.close(fd)
        except (OSError, ValueError):
            return False
        return True

    def claim(batch):
        if (root / 'f1-claimed.json').exists():
            return None
        # Only marker snapshots are locked; no inference, scoring or fault wait.
        with (root / 'f1-marker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            active_path = root / 'f1-active.json'
            complete_path = root / 'f1-complete.json'
            if not active_path.exists():
                return None
            active = json.loads(active_path.read_text())
            if not active_ok(active):
                return None
            matched = matching(batch)
            unfinished = [req for req in matched if not req.finished()]
            finished = [req for req in matched if req.finished()]
            if not unfinished:
                return None
            complete = None
            if complete_path.exists():
                complete = json.loads(complete_path.read_text())
                if (complete['ambiguous']
                        or complete['run_nonce'] != os.environ['FT_RUN_NONCE']
                        or active['trainer'] != complete['trainer']
                        or active['task_id'] != complete['task_id']
                        or complete['source_row_id'] != 5518):
                    return None
            elif finished:
                # Trainer has not observed generation_complete yet, but the
                # scheduler already has a finished sibling for this prompt.
                complete = {
                    **active,
                    'sample_idx': None,
                    'monotonic_ns': time.monotonic_ns(),
                    'proof': 'scheduler_sibling_finished',
                    'finished_request_ids': [req.rid for req in finished],
                    'finished_output_lengths': [len(req.output_ids) for req in finished],
                }
            else:
                return None
            cut = time.monotonic_ns()
            if complete['monotonic_ns'] >= cut:
                if complete.get('proof') == 'scheduler_sibling_finished':
                    complete = {**complete, 'monotonic_ns': cut - 1}
                else:
                    raise RuntimeError('invalid F1 event order')
            import torch
            witness = {
                'identity': snapshot(os.getpid()),
                'scheduler_attached_ns': attached_at,
                'cut_monotonic_ns': cut,
                'completed': complete,
                'active': active,
                'request_ids': [req.rid for req in unfinished],
                'output_lengths': [len(req.output_ids) for req in unfinished],
                'prompt_sha256': contract['prompt_sha256'],
                'gpu_uuid': str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                'tp_rank': scheduler.tp_rank,
                'dp_rank': scheduler.dp_rank,
                'scope': 'group-level full-prompt mapping; no sample_idx attribution',
                'arming_proof': complete.get('proof', 'trainer_f1_complete'),
            }
            try:
                with (root / 'f1-claimed.json').open('x') as stream:
                    stream.write(json.dumps(witness))
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError:
                return None
            return witness

    def fire(witness):
        client = Client('generator', event_id='ft1-f1-generator')
        try:
            if client.injection['status'] == 'already_fired':
                return False
            evidence = {'phase': 'generator_active_after_one_response', 'source_row_id': 5518, 'k': 8}
            witness['incarnation'] = client.incarnation
            (root / 'f1-worker-witness.json').write_text(json.dumps(witness, indent=2))
            client.ready('ft1-f1-generator', evidence)
            client.wait_release('ft1-f1-generator')
            raise RuntimeError('SIGKILL target survived')
        finally:
            client.close()

    def run_batch(batch, *args, **kwargs):
        witness = claim(batch)
        if witness is not None:
            fire(witness)
            return original(batch, *args, **kwargs)
        result = original(batch, *args, **kwargs)
        # Post-step arming: catch the batch where the first sibling finishes
        # while others remain unfinished (R-arm k=8 same-step race).
        witness = claim(batch)
        if witness is not None:
            fire(witness)
        return result

    scheduler.run_batch = run_batch
