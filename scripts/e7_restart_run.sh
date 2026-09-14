#!/usr/bin/env bash
# One versioned 0.5B clean run. Preflight runs before any training mutation.
set -euo pipefail
BASE=${RTX_BASE:-/public/home/caiyiwen/rewardtxn}
STAGE=${1:?pilot|formal}
SEED=${2:?seed}
GROUP=${3:?group_rm|b6}
EXP_ID=${4:?new e7restart- run ID}
case "$STAGE" in pilot) PAPER_MODE=0 ;; formal) PAPER_MODE=1 ;; *) exit 2 ;; esac
case "$GROUP" in group_rm|b6) ;; *) exit 2 ;; esac
if [[ -n "${RTX_PROFILE:-}" ]]; then
  echo 'RTX_PROFILE must be unset for the versioned E7 recipe.' >&2
  exit 2
fi
export RTX_LR=${RTX_LR:-1e-6} RTX_TRAIN_ENTRY=train_async.py RTX_FULLY_ASYNC=1
export RTX_MODEL_DIR=/root/models/Qwen2.5-0.5B-Instruct RTX_MODEL_CONFIG=qwen2.5-0.5B.sh
export RTX_DATA_PATH=/workspace/runs/diagnosis-20260910/train.jsonl
export RTX_SAVE_HF=1 RTX_SAVE_INTERVAL=50 RTX_NUM_ROLLOUT=500 RTX_CKPT_KEEP=0 RTX_NO_SAVE_OPTIM=1
export RTX_GPUS=${RTX_FORMAL_GPUS:-device=0,1,2,3} RTX_NUM_GPUS=4 RTX_SEED="$SEED" RTX_BASELINE_MODE="$GROUP"
export RTX_MAX_TOKENS_PER_GPU=3072 RTX_SGLANG_MEM_FRACTION_STATIC=0.45 RTX_SGLANG_CONCURRENCY=24
export RTX_PAPER_MODE="$PAPER_MODE" RTX_ALLOW_REUSE=0 RTX_SKIP_GATE=0 RTX_EXP_ID="$EXP_ID"
export RTX_FREEZE_MANIFEST="$BASE/prereg/restarts/0.5B-20260911/inputs.json"
export RTX_SCHEDULE="$BASE/prereg/fault_schedules/e7-restart-clean-empty.json"
export RTX_RESTART_PROTOCOL_SHA256=$(sha256sum "$BASE/prereg/restarts/0.5B-20260911/protocol.json" | cut -d ' ' -f 1)
# Fix the approved recipe; arbitrary extra training flags are not accepted by this versioned entry.
if [[ -n "${RTX_EXTRA_MODEL_ARGS:-}" ]]; then
  echo 'Unset RTX_EXTRA_MODEL_ARGS; this versioned recipe fixes its training arguments.' >&2
  exit 2
fi
export RTX_EXTRA_MODEL_ARGS="--use-rollout-logprobs --save-debug-rollout-data /workspace/runs/$EXP_ID/rollout_debug/{rollout_id}.pt --check-weight-update-equal"
python3 "$BASE/scripts/e7_restart_checks.py" preflight --stage "$STAGE" --seed "$SEED"
if [[ "${5:-}" == --check ]]; then exit 0; fi
bash "$BASE/scripts/phase2_run.sh" none -1 -1 500 "$EXP_ID"
GPU=${RTX_GPUS#device=}
python3 "$BASE/scripts/e7_restart_finish.py" "$EXP_ID" --stage "$STAGE" --gpu "${GPU%%,*}"
