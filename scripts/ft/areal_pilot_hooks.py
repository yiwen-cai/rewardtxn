"""Append-only observation for the synchronous, no-fault AReaL pilot.

No observation is read back into training or recovery. Not an oracle or R.
"""
from __future__ import annotations

import atexit
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

from areal import workflow_context
from areal.workflow.rlvr import RLVRWorkflow

IDENTITY_KEYS = ("pilot_source_row_id", "pilot_task_id", "pilot_sample_idx")
_writer = None
_installed = False


class EventWriter:
    """One writer thread and exclusive file per process incarnation; fail closed."""

    def __init__(self, root):
        self.pid = os.getpid()
        self.root = Path(root)
        self.incarnation = uuid.uuid4().hex
        self.queue = queue.Queue(maxsize=8192)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            identity = {
                "pid": self.pid,
                "process_incarnation": self.incarnation,
                "starttime": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "cgroup": Path("/proc/self/cgroup").read_text(),
                "rank": os.environ.get("RANK"),
                "source_role": "trainer",
            }
            path = self.root / f"events-{self.pid}-{self.incarnation}.jsonl"
            with path.open("x", buffering=1) as stream:
                while True:
                    item = self.queue.get()
                    try:
                        if item is None:
                            return
                        stream.write(json.dumps({**identity, **item}, ensure_ascii=False) + "\n")
                    finally:
                        self.queue.task_done()
        except BaseException as exc:
            self.error = exc

    def emit(self, event, **fields):
        if self.error is not None:
            raise RuntimeError("pilot observation writer failed") from self.error
        self.queue.put_nowait({"event": event, "event_nonce": uuid.uuid4().hex,
                              "monotonic_ns": time.monotonic_ns(), "wall_time_ns": time.time_ns(),
                              **fields})

    def flush(self, timeout=20):
        deadline = time.monotonic() + timeout
        while self.queue.unfinished_tasks and self.error is None:
            if time.monotonic() >= deadline:
                raise TimeoutError("pilot observation flush exceeded deadline")
            time.sleep(0.01)
        if self.error is not None:
            raise RuntimeError("pilot observation writer failed") from self.error

    def close(self):
        self.flush()
        self.queue.put_nowait(None)
        self.thread.join(timeout=20)
        if self.thread.is_alive() or self.error is not None:
            raise RuntimeError("pilot observation writer did not close cleanly")


def writer():
    global _writer
    if _writer is None or _writer.pid != os.getpid():
        _writer = EventWriter(os.environ["AREAL_PILOT_EVENTS"])
    return _writer


def emit(event, **fields):
    writer().emit(event, **fields)


def close_events():
    global _writer
    if _writer is not None and _writer.pid == os.getpid():
        current, _writer = _writer, None
        current.close()


atexit.register(close_events)


def observed_gsm8k_reward(prompt, completions, prompt_ids, completion_ids, answer,
                          pilot_observation, **kwargs):
    """Pickleable callable executed by the unmodified AsyncRewardWrapper pool."""
    from areal.reward.gsm8k import gsm8k_reward_fn

    invocation = uuid.uuid4().hex
    fields = {**pilot_observation, "invocation": invocation, "source_role": "reward",
              "verifier": "areal.reward.gsm8k.gsm8k_reward_fn:math_verify"}
    emit("reward_start", **fields)
    try:
        result = gsm8k_reward_fn(prompt, completions, prompt_ids, completion_ids,
                                 answer=answer, **kwargs)
        emit("reward_done", reward=result, **fields)
        return result
    except BaseException as exc:
        emit("reward_error", error=repr(exc), **fields)
        raise
    finally:
        # Pool workers do not necessarily run Python atexit handlers. Bound the
        # drain in this CPU subprocess, never on the rollout event loop.
        writer().flush()


class ObservedRLVRWorkflow(RLVRWorkflow):
    async def _compute_rewards(self, resp, prompt_str, task_data):
        ctx = workflow_context.get()
        observation = {"source_row_id": task_data["source_row_id"],
                       "task_id": ctx.task_id, "sample_idx": ctx.sample_idx,
                       "sample_attempt": uuid.uuid4().hex}
        emit("generation_done", **observation, source_role="rollout", messages=task_data["messages"],
             answer=task_data["answer"], input_tokens=resp.input_tokens,
             output_tokens=resp.output_tokens, output_logprobs=resp.output_logprobs,
             output_versions=resp.output_versions, stop_reason=resp.stop_reason)
        return await super()._compute_rewards(
            resp, prompt_str, {**task_data, "pilot_observation": observation})


def batch_evidence(data):
    """Capture actual training tensors and identity, including padding and dtype."""
    batches = data if isinstance(data, list) else [data]
    evidence = []
    for batch in batches:
        keys = (*IDENTITY_KEYS, "input_ids", "attention_mask", "loss_mask", "rewards", "versions")
        record = {key: batch[key].detach().cpu().tolist() for key in keys if key in batch}
        record["batch_size"] = batch["input_ids"].shape[0]
        evidence.append(record)
    return evidence


def file_manifest(path):
    result = []
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result.append({"path": str(file.relative_to(path)), "size": file.stat().st_size,
                           "sha256": digest.hexdigest()})
    return result


def install_hooks():
    """Install before PPOTrainer construction, in each SPMD training process."""
    global _installed
    if _installed:
        return
    import torch
    from areal.engine.megatron_engine import MegatronPPOActor
    from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
    from areal.utils.recover import RecoverHandler, RecoverInfo

    original_group = GroupedRolloutWorkflow.arun_episode

    @functools.wraps(original_group)
    async def group(self, engine, data):
        result = await original_group(self, engine, data)
        if isinstance(self.workflow, ObservedRLVRWorkflow) and result is not None:
            ctx = workflow_context.get()
            if not isinstance(ctx.task_id, int):
                raise TypeError("pilot requires native integer WorkflowContext.task_id")
            if result["input_ids"].shape[0] != self.group_size:
                raise RuntimeError("pilot observed incomplete group")
            for key, values in zip(IDENTITY_KEYS, (
                [data["source_row_id"]] * self.group_size,
                [ctx.task_id] * self.group_size, list(range(self.group_size)),
            )):
                result[key] = torch.tensor(values, dtype=torch.int64)
            emit("group_admitted", source_role="rollout", source_row_id=data["source_row_id"], task_id=ctx.task_id,
                 sample_indices=list(range(self.group_size)))
        return result

    GroupedRolloutWorkflow.arun_episode = group

    original_prepare = MegatronPPOActor.prepare_batch

    @functools.wraps(original_prepare)
    def prepare(self, *args, **kwargs):
        result = original_prepare(self, *args, **kwargs)
        emit("batch_taken", batches=batch_evidence(result), policy_version=self.get_version())
        return result

    MegatronPPOActor.prepare_batch = prepare
    original_logp = MegatronPPOActor.compute_logp

    @functools.wraps(original_logp)
    def compute_logp(self, data, *args, **kwargs):
        clean = [{k: v for k, v in batch.items() if k not in IDENTITY_KEYS} for batch in data]
        return original_logp(self, clean, *args, **kwargs)

    MegatronPPOActor.compute_logp = compute_logp
    original_train = MegatronPPOActor.train_batch

    @functools.wraps(original_train)
    def train_batch(self, input_, *args, **kwargs):
        self._pilot_update = uuid.uuid4().hex
        emit("train_batch", update_id=self._pilot_update, batches=batch_evidence(input_),
             policy_version=self.get_version())
        # This extra key is never an input to model/loss or checkpoint state.
        input_ = {k: v for k, v in input_.items() if k not in IDENTITY_KEYS}
        return original_train(self, input_, *args, **kwargs)

    MegatronPPOActor.train_batch = train_batch
    original_step = MegatronPPOActor.optimizer_step

    @functools.wraps(original_step)
    def optimizer_step(self):
        emit("optimizer_start", update_id=self._pilot_update)
        result = original_step(self)
        emit("optimizer_end", update_id=self._pilot_update, stats=result)
        return result

    MegatronPPOActor.optimizer_step = optimizer_step

    def wrap_io(name):
        original = getattr(MegatronPPOActor, name)

        @functools.wraps(original)
        def operation(self, meta):
            fields = {"path": meta.path, "with_optim": meta.with_optim,
                      "weight_format": meta.weight_format,
                      "update_id": getattr(self, "_pilot_update", None),
                      "async_save": self.config.megatron.async_save}
            emit(f"checkpoint_{name}_start", **fields)
            result = original(self, meta)
            emit(f"checkpoint_{name}_returned", **fields, files=file_manifest(meta.path))
            return result
        setattr(MegatronPPOActor, name, operation)

    wrap_io("save")
    wrap_io("load")
    original_dump = RecoverInfo.dump

    @functools.wraps(original_dump)
    def dump(self, dump_dir):
        result = original_dump(self, dump_dir)
        emit("metadata_written", path=dump_dir,
             last_step_info=dataclasses.asdict(self.last_step_info), files=file_manifest(dump_dir))
        return result

    RecoverInfo.dump = dump
    for name in ("dump", "load"):
        original = getattr(RecoverHandler, name)

        def make_handler(original, name):
            @functools.wraps(original)
            def handler(self, *args, **kwargs):
                emit(f"recover_handler_{name}_start")
                result = original(self, *args, **kwargs)
                emit(f"recover_handler_{name}_returned")
                return result
            return handler

        setattr(RecoverHandler, name, make_handler(original, name))
    _installed = True
