#!/usr/bin/env bash
# Launch one named, isolated diagnosis. Resource gate remains enabled.
set -euo pipefail
BASE=$(cd "$(dirname "$0")/.." && pwd)
CONDITION=${1:?usage: e7_diagnosis_launch.sh A|B|C|D|E}
LOSS_ARGS=""
export RTX_BASE="$BASE"
export RTX_GPUS=${RTX_GPUS:-device=1,2,3,4}
export RTX_SEED=29 RTX_BASELINE_MODE=group_rm RTX_PAPER_MODE=0
export RTX_MODEL_DIR=/root/models/Qwen2.5-0.5B-Instruct
export RTX_MODEL_CONFIG=qwen2.5-0.5B.sh
export RTX_DATA_PATH=/workspace/runs/diagnosis-20260910/train.jsonl
export RTX_NO_SAVE_OPTIM=1 RTX_CKPT_KEEP=0 RTX_SAVE_HF=1 RTX_SAVE_INTERVAL=5
export RTX_MAX_TOKENS_PER_GPU=3072 RTX_SGLANG_MEM_FRACTION_STATIC=0.45
export RTX_SGLANG_CONCURRENCY=24 RTX_NUM_ROLLOUT=50
case "$CONDITION" in
  A) export RTX_FULLY_ASYNC=1 RTX_TRAIN_ENTRY=train_async.py RTX_LR=5e-5 ;;
  B) export RTX_FULLY_ASYNC=0 RTX_TRAIN_ENTRY=train.py RTX_LR=5e-5 ;;
  C) export RTX_FULLY_ASYNC=1 RTX_TRAIN_ENTRY=train_async.py RTX_LR=1e-5 ;;
  D) export RTX_FULLY_ASYNC=1 RTX_TRAIN_ENTRY=train_async.py RTX_LR=1e-5; LOSS_ARGS="--use-rollout-logprobs" ;;
  E) export RTX_FULLY_ASYNC=1 RTX_TRAIN_ENTRY=train_async.py RTX_LR=1e-6; LOSS_ARGS="--use-rollout-logprobs" ;;
  *) echo 'condition must be A, B, C, D, or E' >&2; exit 2 ;;
esac
export RTX_EXP_ID=${RTX_EXP_ID:-"diagnosis-E7-${CONDITION}-s29-20260910"}
export RTX_EXTRA_MODEL_ARGS="${LOSS_ARGS} --save-debug-rollout-data /workspace/runs/${RTX_EXP_ID}/rollout_debug/{rollout_id}.pt --check-weight-update-equal"
python3 - "$BASE" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])
comparison = json.loads((base/'runs/diagnosis-20260910/roundtrip_comparison.json').read_text())
assert comparison['roundtrip']['pass'], 'roundtrip gate failed'
assert (base/'runs/diagnosis-20260910/base_validation.json').exists(), 'base validation incomplete'
PY
bash "$BASE/scripts/phase2_run.sh" none 0 0 50 "$RTX_EXP_ID"
