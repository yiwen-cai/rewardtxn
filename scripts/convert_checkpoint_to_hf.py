#!/usr/bin/env python3
"""Convert a Megatron checkpoint to HuggingFace (safetensors) format.

Implementation note: ``tools/convert_to_hf.py`` in this Slime tree is stale
(imports removed modules), so this converter uses the current conversion
machinery directly: build the Megatron model from the checkpoint's config,
load the checkpoint, then call ``save_hf_model_to_path`` (the same path the
actor uses for ``--save-hf`` during training).

Must run inside the Slime container with a GPU, and the model architecture
args must match the checkpoint (the training config
``scripts/models/qwen2.5-1.5B.sh`` provides them).  Launch under ``torchrun``
so distributed env vars are set:

  torchrun --nproc_per_node=1 scripts/convert_checkpoint_to_hf.py \\
      --checkpoint /workspace/runs/<run>/checkpoints/iter_0000029 \\
      --hf-checkpoint /root/models/Qwen2.5-1.5B-Instruct \\
      --output /workspace/runs/<run>/checkpoints/iter_0000029_hf \\
      <model args from scripts/models/qwen2.5-1.5B.sh> \\
      --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 ...

Output: a complete HF directory (model-*.safetensors + model.safetensors.index.json,
config.json, tokenizer files) loadable by ``transformers`` and by
``scripts/e7_eval_checkpoint_incontainer.py``.
"""
from __future__ import annotations

import sys

import torch
import torch.distributed as dist

from slime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
from slime.backends.megatron_utils.initialize import init as megatron_init
from slime.backends.megatron_utils.model import initialize_model_and_optimizer
from slime.utils.arguments import parse_args
from slime.utils.distributed_utils import init_gloo_group


def add_converter_args(parser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Megatron checkpoint directory to convert (equivalent to --load).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for the converted HF checkpoint.",
    )
    return parser


def main() -> int:
    args = parse_args(add_custom_arguments=add_converter_args)
    if not args.checkpoint or not args.output:
        print("error: --checkpoint and --output are required", file=sys.stderr)
        return 2

    args.load = args.checkpoint
    # The checkpoint may or may not contain optimizer state; the converted HF
    # weights only need the model parameters.
    args.no_load_optim = True
    args.no_load_rng = True
    args.save = None

    local_rank = int(torch.cuda.current_device())
    torch.cuda.set_device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://")
    init_gloo_group()

    megatron_init(args)
    model, _optimizer, _sched, iteration = initialize_model_and_optimizer(args)
    print(f"Loaded Megatron checkpoint (iteration {iteration}); converting to HF...", file=sys.stderr)

    save_hf_model_to_path(args, args.output, model, progress_desc="Save HF checkpoint (standalone conversion)")

    if dist.get_rank() == 0:
        print(f"HF checkpoint written to {args.output}", file=sys.stderr)
    dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
