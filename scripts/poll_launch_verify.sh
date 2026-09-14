#!/usr/bin/env bash
# Poll for free GPUs 4-7, then launch the E7 patch verification run.
# Usage: bash scripts/poll_launch_verify.sh <exp_id> <n_rollout>
set -uo pipefail

BASE=/public/home/caiyiwen/rewardtxn
EXP_ID=${1:-verify-e7-patch2-0.5B-4gpu-s29-$(date +%Y%m%d-%H%M%S)}
N_ROLLOUT=${2:-15}
STATUS="$BASE/runs/${EXP_ID}/poll_status.json"
mkdir -p "$BASE/runs/${EXP_ID}"
echo "{\"phase\":\"waiting\",\"started_at\":\"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)\"}" > "$STATUS"

gpus_free() {
  local idx
  for idx in 4 5 6 7; do
    local mem free_gib
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx")
    free_gib=$(( (79102 - mem) / 1024 ))
    if [ "$free_gib" -lt 55 ]; then
      return 1
    fi
  done
  # also require no compute processes on those GPUs
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q . && {
    local uuids cnt
    uuids=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F, '$1 ~ /^[4-7]$/ {print $2}' | tr -d ' ')
    for u in $uuids; do
      cnt=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$u")
      [ "$cnt" -gt 0 ] && return 1
    done
  }
  return 0
}

while true; do
  if gpus_free; then
    echo "{\"phase\":\"launching\",\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)\"}" > "$STATUS"
    break
  fi
  sleep 60
done

cd "$BASE"
RTX_GPUS="device=4,5,6,7" \
RTX_SEED=29 \
RTX_BASELINE_MODE="group_rm" \
RTX_PAPER_MODE=0 \
RTX_SKIP_GATE=1 \
RTX_MODEL_DIR="/root/models/Qwen2.5-0.5B-Instruct" \
RTX_MODEL_CONFIG="qwen2.5-0.5B.sh" \
RTX_DATA_PATH="/root/datasets/gsm8k/dapo-gsm8k-train.jsonl" \
RTX_EXTRA_MODEL_ARGS="--rotary-base 1000000" \
RTX_MAX_TOKENS_PER_GPU=3072 \
RTX_SGLANG_MEM_FRACTION_STATIC=0.45 \
RTX_SGLANG_CONCURRENCY=24 \
RTX_SAVE_INTERVAL=10 \
RTX_SCHEDULE="$BASE/prereg/fault_schedules/e7-s29-faulted.json" \
bash scripts/phase2_run.sh none -1 -1 "$N_ROLLOUT" "$EXP_ID"
RC=$?
echo "{\"phase\":\"done\",\"exit\":$RC,\"at\":\"$(date -u +%Y-%m-%dT%H:%M:%S+00:00)\"}" > "$STATUS"
exit $RC
