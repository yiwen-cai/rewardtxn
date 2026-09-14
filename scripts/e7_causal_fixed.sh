#!/usr/bin/env bash
# Two fixed-batch steps: warmup step zero, then one nonzero optimizer update.
set -euo pipefail
BASE=/public/home/caiyiwen/rewardtxn
GROUP=${1:?group_rm|b6}
EXP_ID=${2:?independent run ID}
case "$GROUP" in group_rm|b6) ;; *) exit 2 ;; esac
export RTX_BASE="$BASE" RTX_GPUS=device=1 RTX_NUM_GPUS=1 RTX_ACTOR_GPUS=1 RTX_ROLLOUT_GPUS=0
export RTX_SEED=11 RTX_BASELINE_MODE="$GROUP" RTX_PAPER_MODE=0 RTX_ALLOW_REUSE=0 RTX_SKIP_GATE=0
export RTX_MODEL_DIR=/root/models/Qwen2.5-0.5B-Instruct RTX_MODEL_CONFIG=qwen2.5-0.5B.sh
export RTX_DATA_PATH=/workspace/runs/diagnosis-20260910/train.jsonl
export RTX_NO_SAVE_OPTIM=1 RTX_CKPT_KEEP=0 RTX_SAVE_HF=1 RTX_SAVE_INTERVAL=1
export RTX_MAX_TOKENS_PER_GPU=3072 RTX_NUM_ROLLOUT=2 RTX_FULLY_ASYNC=0
export RTX_TRAIN_ENTRY=train.py RTX_LR=1e-6 RTX_EXP_ID="$EXP_ID"
export RTX_EXTRA_MODEL_ARGS="--use-rollout-logprobs --lr-decay-iters 500 --load-debug-rollout-data /workspace/runs/pilot-causal-audit-20260911/fixed_${GROUP}.pt --save-debug-rollout-data /workspace/runs/${EXP_ID}/rollout_debug/{rollout_id}.pt --save-debug-train-data /workspace/runs/${EXP_ID}/train_debug/{rollout_id}_{rank}.pt"
bash "$BASE/scripts/phase2_run.sh" none -1 -1 2 "$EXP_ID"
docker wait "rtx-p2-$EXP_ID" > "$BASE/runs/$EXP_ID/logs/wait_exit_code.txt"
docker inspect --format '{{json .State}}' "rtx-p2-$EXP_ID" > "$BASE/runs/$EXP_ID/logs/terminal_state.json"
python3 - "$BASE/runs/$EXP_ID/logs/terminal_state.json" <<'PY'
import json, sys
s=json.load(open(sys.argv[1]))
assert s['ExitCode'] == 0 and not s['OOMKilled'], s
PY
docker run --rm --pull=never --network=none -v "$BASE/runs/$EXP_ID:/run-output" \
  slimerl/slime:v0.3.1 chown -R "$(id -u):$(id -g)" /run-output
