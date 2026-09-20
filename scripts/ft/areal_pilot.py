"""Three-step, native SPMD RLVR pilot. No injected faults or recovery protocol."""
import json
import os
from pathlib import Path
import sys


def load_pilot_dataset(path):
    from datasets import Dataset

    rows = []
    with Path(path).open() as stream:
        for row_id, line in enumerate(stream):
            row = json.loads(line)
            if not isinstance(row["prompt"], list) or not isinstance(row["label"], str):
                raise ValueError(f"unexpected source schema at line {row_id + 1}")
            rows.append({"messages": row["prompt"], "answer": row["label"],
                         "source_row_id": row_id})
    return Dataset.from_list(rows)


def main(args):
    from areal import PPOTrainer
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from scripts.ft.areal_pilot_hooks import close_events, emit, install_hooks

    config, _ = load_expr_config(args, GRPOConfig)
    os.environ["AREAL_PILOT_EVENTS"] = str(Path(config.cluster.fileroot) / "pilot_events")
    install_hooks()
    try:
        dataset = load_pilot_dataset(config.train_dataset.path)
        emit("pilot_start", dataset_rows=len(dataset), config_path=args)
        with PPOTrainer(config, train_dataset=dataset, valid_dataset=None) as trainer:
            trainer.train(
                workflow="scripts.ft.areal_pilot_hooks.ObservedRLVRWorkflow",
                workflow_kwargs={
                    "reward_fn": "scripts.ft.areal_pilot_hooks.observed_gsm8k_reward",
                    "gconfig": config.gconfig,
                    "tokenizer": config.tokenizer_path,
                },
            )
        emit("pilot_training_returned")
    except BaseException as exc:
        emit("pilot_error", error=repr(exc))
        raise
    finally:
        close_events()


if __name__ == "__main__":
    main(sys.argv[1:])
