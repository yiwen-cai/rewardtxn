#!/usr/bin/env bash
# Test HF checkpoint saving: convert a Megatron checkpoint to HuggingFace format
# inside the Slime container, mirroring the training container's environment.
#
# Usage:
#   bash scripts/test_hf_checkpoint_save.sh <megatron_checkpoint_dir> [gpu_id]
#
# The output is written to <checkpoint_dir>_hf/ and is loadable by
# transformers / scripts/e7_eval_checkpoint_incontainer.py.
#
# This validates the same conversion path the formal E7 runs will use
# (--save-hf in day2_slime_train.sh), on a real pilot checkpoint.
set -euo pipefail

CHECKPOINT_DIR=${1:?Usage: $0 <megatron_checkpoint_dir> [gpu_id]}
GPU=${2:-4}
BASE=$(cd "$(dirname "$0")/.." && pwd)
SLIME_SOURCE_DIR=${RTX_SLIME_SOURCE_DIR:-$BASE/third_party/slime}
IMAGE=${RTX_IMAGE:-slimerl/slime:v0.3.1}
MASTER_PORT=${RTX_CONVERT_MASTER_PORT:-29617}

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Error: checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi
if [ ! -d "$SLIME_SOURCE_DIR/tools" ]; then
  echo "Error: slime source not found at $SLIME_SOURCE_DIR (set RTX_SLIME_SOURCE_DIR)" >&2
  exit 1
fi
if [ ! -d "$BASE/models/Qwen2.5-1.5B-Instruct" ]; then
  echo "Error: HF model assets not found at $BASE/models/Qwen2.5-1.5B-Instruct" >&2
  exit 1
fi

# Absolute paths, then map to container paths
CHECKPOINT_ABS=$(cd "$CHECKPOINT_DIR" && pwd)
OUTPUT_DIR="${CHECKPOINT_ABS}_hf"
CHECKPOINT_REL=${CHECKPOINT_ABS#$BASE/}
OUTPUT_REL=${OUTPUT_DIR#$BASE/}

echo "=== HF checkpoint save test ==="
echo "Checkpoint : $CHECKPOINT_ABS"
echo "Output     : $OUTPUT_DIR"
echo "GPU        : $GPU"
echo "Slime src  : $SLIME_SOURCE_DIR"
echo ""

docker run --rm \
  --gpus "device=$GPU" \
  --shm-size 32g \
  -v "$BASE":/workspace \
  -v "$SLIME_SOURCE_DIR":/root/slime \
  -v "$BASE/models":/root/models \
  -w /workspace \
  -e SLIME_SOURCE_DIR=/root/slime \
  -e PYTHONPATH=/root/Megatron-LM/:/root/slime:/workspace/scripts \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    git config --global --add safe.directory /root/slime 2>/dev/null || true
    cd /root/slime
    pip install -e . --no-deps -q
    source /root/slime/scripts/models/qwen2.5-1.5B.sh
    export MASTER_ADDR=127.0.0.1
    torchrun --nproc_per_node=1 --master_port='"$MASTER_PORT"' \
      /workspace/scripts/convert_checkpoint_to_hf.py \
      --checkpoint "/workspace/'"$CHECKPOINT_REL"'" \
      --hf-checkpoint /root/models/Qwen2.5-1.5B-Instruct \
      --output "/workspace/'"$OUTPUT_REL"'" \
      "${MODEL_ARGS[@]}" \
      --num-rollout 8 \
      --rollout-batch-size 4 \
      --n-samples-per-prompt 8 \
      --global-batch-size 32 \
      --tensor-model-parallel-size 1 \
      --pipeline-model-parallel-size 1 \
      --context-parallel-size 1 \
      --expert-model-parallel-size 1 \
      --expert-tensor-parallel-size 1 \
      --seed 42
  '

echo ""
echo "=== Conversion complete ==="
echo "Output: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR" | head -20
