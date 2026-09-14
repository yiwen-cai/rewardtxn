#!/usr/bin/env bash
# Replay precisely the same pre-update batch; no SGLang or new generation.
set -euo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
CONDITION=${1:?usage: e7_diagnosis_fixed.sh high|low|clipped}
LOSS_ARGS=""
case "$CONDITION" in
 high) LR=5e-5; GPU=1 ;;
 low) LR=1e-5; GPU=2 ;;
 clipped) LR=1e-5; GPU=0; LOSS_ARGS="--use-rollout-logprobs" ;;
 *) exit 2 ;;
esac
export RTX_BASE="$BASE" RTX_GPUS="${RTX_GPUS:-device=$GPU}" RTX_NUM_GPUS=1 RTX_ACTOR_GPUS=1 RTX_ROLLOUT_GPUS=0
export RTX_SEED=29 RTX_BASELINE_MODE=group_rm RTX_PAPER_MODE=0
export RTX_MODEL_DIR=/root/models/Qwen2.5-0.5B-Instruct RTX_MODEL_CONFIG=qwen2.5-0.5B.sh
export RTX_DATA_PATH=/workspace/runs/diagnosis-20260910/train.jsonl
export RTX_NO_SAVE_OPTIM=1 RTX_CKPT_KEEP=0 RTX_SAVE_HF=1 RTX_SAVE_INTERVAL=1
export RTX_MAX_TOKENS_PER_GPU=3072 RTX_SGLANG_MEM_FRACTION_STATIC=0.45
export RTX_SGLANG_CONCURRENCY=24 RTX_NUM_ROLLOUT=10 RTX_FULLY_ASYNC=0
export RTX_TRAIN_ENTRY=train.py RTX_LR="$LR"
export RTX_EXP_ID=${RTX_EXP_ID:-"diagnosis-E7-fixed-${CONDITION}-s29-20260910"}
export RTX_RAY_TMP_DIR=/tmp/rtx-fixed
export RTX_EXTRA_MODEL_ARGS="${LOSS_ARGS} --lr-decay-iters 50 --load-debug-rollout-data /workspace/runs/diagnosis-E7-A-s29-20260910-r1/rollout_debug/0.pt --save-debug-rollout-data /workspace/runs/${RTX_EXP_ID}/rollout_debug/{rollout_id}.pt --save-debug-train-data /workspace/runs/${RTX_EXP_ID}/train_debug/{rollout_id}_{rank}.pt"
bash "$BASE/scripts/phase2_run.sh" none 0 0 10 "$RTX_EXP_ID"
