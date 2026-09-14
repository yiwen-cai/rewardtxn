#!/usr/bin/env bash
# E7 checkpoint evaluation wrapper: runs evaluation inside Slime Docker container
# Usage: bash scripts/e7_eval_checkpoint_docker.sh <checkpoint_dir> <output_json> [verifier_version]

set -euo pipefail

CHECKPOINT_DIR=${1:?Usage: $0 <checkpoint_dir> <output_json> [verifier_version]}
OUTPUT_JSON=${2:?Usage: $0 <checkpoint_dir> <output_json> [verifier_version]}
VERIFIER=${3:-v1}
BASE=$(cd "$(dirname "$0")/.." && pwd)

if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Error: checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

EVAL_SPLIT="$BASE/prereg/eval_splits/gsm8k_eval500_seed42.json"
if [ ! -f "$EVAL_SPLIT" ]; then
  echo "Error: eval split not found: $EVAL_SPLIT" >&2
  exit 1
fi

# Resolve absolute paths
CHECKPOINT_ABS=$(cd "$CHECKPOINT_DIR" && pwd)
OUTPUT_ABS=$(cd "$(dirname "$OUTPUT_JSON")" && pwd)/$(basename "$OUTPUT_JSON")

echo "=== E7 Checkpoint Evaluation ==="
echo "Checkpoint: $CHECKPOINT_ABS"
echo "Output: $OUTPUT_ABS"
echo "Verifier: $VERIFIER"
echo ""

# Map paths to container
CHECKPOINT_REL=${CHECKPOINT_ABS#$BASE/}
OUTPUT_REL=${OUTPUT_ABS#$BASE/}

# Run evaluation in Docker container
docker run --rm \
  --gpus '"device=4,5,6,7"' \
  --shm-size 32g \
  -v "$BASE":/workspace \
  -v /tmp/rewardtxn:/rtx-scratch \
  -w /workspace \
  slimerl/slime:v0.3.1 \
  python3 /workspace/scripts/e7_eval_checkpoint_incontainer.py \
    --checkpoint "/workspace/$CHECKPOINT_REL" \
    --eval-split /workspace/prereg/eval_splits/gsm8k_eval500_seed42.json \
    --verifier-version "$VERIFIER" \
    --output "/workspace/$OUTPUT_REL" \
    --gpus 4

echo ""
echo "Evaluation complete. Results written to: $OUTPUT_JSON"
