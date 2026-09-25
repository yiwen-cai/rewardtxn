"""CPU contracts against the real installed AReaL; no GPU recovery claims."""
import asyncio
import copy
import json
import logging
from pathlib import Path
import pickle
import tempfile

import torch

from areal.api.alloc_mode import _AllocationMode
from areal.api.cli_args import GRPOConfig, MicroBatchSpec, load_expr_config
from areal.infra import workflow_context
from areal.infra.remote_inf_engine import GroupedRolloutWorkflow
from areal.infra.workflow_context import WorkflowContext
from areal.trainer.ppo.actor import PPOActor
from areal.utils.data import concat_padded_tensors, split_padded_tensor_dict_into_mb_list
from scripts.ft.areal_pilot import load_pilot_dataset
from scripts.ft import areal_pilot_hooks as hooks

CONFIG = Path(__file__).resolve().parents[2] / "docs/experiments/rewardtxn-ft-20260916/pilot.yaml"


def config():
    with tempfile.TemporaryDirectory(prefix="areal-pilot-contract-") as root:
        return load_expr_config(
            ["--config", str(CONFIG), f"cluster.fileroot={root}/artifacts",
             f"cluster.name_resolve.nfs_record_root={root}/names"], GRPOConfig
        )[0]


def test_real_config_and_legacy_allocation_parse():
    cfg = config()
    mode = _AllocationMode.from_str(cfg.allocation_mode)
    assert len(mode.allocations) == 2
    assert cfg.scheduler.type is None
    assert cfg.rollout._version == cfg.actor._version == "v1"
    assert cfg.gconfig.n_samples == 8
    assert cfg.train_dataset.batch_size == cfg.rollout.consumer_batch_size == 4
    assert cfg.actor.ppo_n_minibatches == 1
    assert cfg.actor.megatron.async_save is False
    assert cfg.actor.megatron.use_checkpoint_opt_param_scheduler is True
    assert cfg.recover.no_save_optim is cfg.recover.no_load_optim is False
    assert cfg.recover.freq_steps == 1 and cfg.total_train_steps == 3
    assert cfg.recover.retries == 0
    assert cfg.valid_dataset is None


def test_source_row_identity_and_explicit_answer(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"prompt": [{"role": "user", "content": "one"}], "label": "1"},
            {"prompt": [{"role": "user", "content": "two"}], "label": "2"}]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = load_pilot_dataset(path)
    assert dataset[1] == {"messages": rows[1]["prompt"], "answer": "2", "source_row_id": 1}


class SyntheticRLVR(hooks.ObservedRLVRWorkflow):
    """Synthetic generation only; real grouping and PPO math below."""
    def __init__(self):
        pass

    async def arun_episode(self, engine, data):
        index = workflow_context.get().sample_idx
        await asyncio.sleep((7 - index) * 0.001)
        return {
            "input_ids": torch.tensor([[1, 2, 3, index + 4, 9, 0]], dtype=torch.int32),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool),
            "loss_mask": torch.tensor([[0, 0, 0, 1, 1, 0]], dtype=torch.int32),
            "logprobs": torch.tensor([[0., 0., 0., -.5, -.5, 0.]]),
            "versions": torch.tensor([[-1, -1, -1, 0, 0, -1]]),
            "rewards": torch.tensor([float(index % 2)]),
            "pilot_sample_attempt": torch.tensor([index + 100], dtype=torch.int64),
        }


def test_group_identity_survives_real_ppo_and_minibatching(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_PILOT_EVENTS", str(tmp_path))
    hooks.install_hooks()

    async def collect():
        workflow_context.set(WorkflowContext(task_id=17))
        grouped = GroupedRolloutWorkflow(SyntheticRLVR(), 8, logging.getLogger("test"))
        return await grouped.arun_episode(None, {"source_row_id": 42})

    try:
        grouped = asyncio.run(collect())
        assert grouped["pilot_sample_idx"].tolist() == list(range(8))
        assert grouped["pilot_source_row_id"].tolist() == [42] * 8
        assert grouped["pilot_sample_attempt"].tolist() == list(range(100, 108))
        assert grouped["input_ids"][:, 3].tolist() == list(range(4, 12))
        batches = [copy.deepcopy(grouped) for _ in range(4)]
        plain = [{k: v.clone() for k, v in b.items() if k not in hooks.IDENTITY_KEYS}
                 for b in batches]
        actor = PPOActor(config().actor, engine=None)
        observed_adv = actor.compute_advantages(batches)
        plain_adv = actor.compute_advantages(plain)
        for observed, expected in zip(observed_adv, plain_adv):
            for key in expected:
                torch.testing.assert_close(observed[key], expected[key], rtol=0, atol=0)
        combined = concat_padded_tensors(observed_adv)
        assert combined["input_ids"].shape[0] == 32
        minibatches = split_padded_tensor_dict_into_mb_list(combined, MicroBatchSpec(n_mbs=1))
        assert len(minibatches.mbs) == 1
        for key in hooks.IDENTITY_KEYS:
            torch.testing.assert_close(minibatches.mbs[0][key], combined[key])
    finally:
        hooks.close_events()


def test_picklable_reward_calls_official_verifier_and_preserves_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("AREAL_PILOT_EVENTS", str(tmp_path))
    reward = pickle.loads(pickle.dumps(hooks.observed_gsm8k_reward))
    try:
        result = reward("one plus one", r"Answer: \boxed{2}", [1], [2], answer="2",
                        pilot_observation={"source_row_id": 0, "task_id": 1,
                                           "sample_idx": 0, "sample_attempt": "fixture"})
        assert result == 1.0
    finally:
        hooks.close_events()
    events = [json.loads(line) for p in tmp_path.glob("events-*.jsonl")
              for line in p.read_text().splitlines()]
    assert [e["event"] for e in events] == ["reward_start", "reward_done"]
    assert events[0]["invocation"] == events[1]["invocation"]
    assert events[1]["starttime"] and events[1]["boot_id"] and events[1]["cgroup"]
