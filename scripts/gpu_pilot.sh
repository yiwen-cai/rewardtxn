#!/usr/bin/env bash
# P0 GPU pilot: E3/E6/E7 core cells, three paired short runs per experiment.
# Usage: bash scripts/gpu_pilot.sh [--dry-run]
# Override the four-GPU slot with RTX_PILOT_GPUS=device=4,5,6,7.
set -euo pipefail

BASE=${RTX_BASE:-/public/home/caiyiwen/rewardtxn}
PILOT_GPUS=${RTX_PILOT_GPUS:-device=4,5,6,7}
PILOT_STEPS=${RTX_PILOT_STEPS:-30}
PILOT_LOG="$BASE/runs/pilot_results_$(date +%Y%m%d-%H%M%S).json"
DRY_RUN=0

case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  *) echo "usage: bash scripts/gpu_pilot.sh [--dry-run]" >&2; exit 2 ;;
esac
case "${PILOT_STEPS}" in
  ''|*[!0-9]*|0) echo "RTX_PILOT_STEPS must be a positive integer" >&2; exit 2 ;;
esac

GPU_CSV=${PILOT_GPUS#device=}
IFS=',' read -r -a GPU_IDS <<< "$GPU_CSV"
[ "${#GPU_IDS[@]}" -eq 4 ] \
  || { echo "RTX_PILOT_GPUS must select exactly four GPUs" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  case "$gpu" in ''|*[!0-9]*) echo "invalid GPU index in RTX_PILOT_GPUS: $gpu" >&2; exit 2 ;; esac
done

SEEDS=(17 29 42)
MODEL_DIR=/root/models/Qwen2.5-1.5B-Instruct
MODEL_CONFIG=qwen2.5-1.5B.sh
EXTRA_MODEL_ARGS="--rotary-base 1000000"

if [ "$DRY_RUN" = "0" ]; then
  python3 - "$PILOT_LOG" "$PILOT_GPUS" "$PILOT_STEPS" "$(git -C "$BASE" rev-parse HEAD)" <<'PY'
import datetime
import json
import sys

path, gpus, steps, commit = sys.argv[1:]
payload = {
    "pilot_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "pilot_commit": commit,
    "gpus": gpus,
    "steps_per_run": int(steps),
    "paired_seeds": [17, 29, 42],
    "e7_power_precision": {
        "status": "pending_final_eval_accuracy",
        "analysis_unit": "paired_seed",
        "pilot_pairs_required": 3,
    },
    "runs": [],
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
PY
fi

record_run() {
  python3 - "$PILOT_LOG" "$@" <<'PY'
import json
import sys

(path, experiment, variant, baseline, exp_id, seed, status, wall_time,
 container_exit, concurrency, checkpoint_mode) = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
payload["runs"].append({
    "experiment": experiment,
    "variant": variant,
    "baseline_mode": baseline,
    "exp_id": exp_id,
    "seed": int(seed),
    "status": status,
    "wall_time_seconds": int(wall_time),
    "container_exit_code": int(container_exit) if container_exit.isdigit() else None,
    "reward_concurrency": int(concurrency),
    "checkpoint_mode": checkpoint_mode,
    "run_dir": "runs/" + exp_id,
})
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, ensure_ascii=False)
PY
}

run_pilot() {
  local experiment=$1
  local variant=$2
  local baseline=$3
  local seed=$4
  local concurrency=$5
  local checkpoint_mode=$6
  local seal group_rm auto_fix custom_rm save_interval

  case "$baseline" in
    b0)
      seal=0; group_rm=0; auto_fix=0; custom_rm=day2_custom_rm.rm_function
      ;;
    group_rm)
      seal=0; group_rm=1; auto_fix=0; custom_rm=day2_custom_rm.rm_function
      ;;
    b6)
      seal=1; group_rm=1; auto_fix=1; custom_rm=phase2_seal_rm.rm_function
      ;;
    *) echo "unsupported pilot baseline: $baseline" >&2; return 2 ;;
  esac
  if [ "$checkpoint_mode" = "off" ]; then
    save_interval=1000000
  else
    save_interval=10
  fi

  local exp_id="pilot-${experiment}-${variant}-1.5B-4gpu-s${seed}-$(date +%H%M%S)"
  local container_name="rtx-p2-${exp_id}"
  local launch_log="$BASE/runs/${exp_id}.launch.log"

  printf '%-3s %-18s seed=%-3s baseline=%-8s gpus=%s steps=%s\n' \
    "$experiment" "$variant" "$seed" "$baseline" "$PILOT_GPUS" "$PILOT_STEPS"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  local start_time end_time wall_time launch_exit wait_exit container_exit status
  start_time=$(date +%s)
  set +e
  RTX_GPUS="$PILOT_GPUS" \
  RTX_NUM_GPUS=4 \
  RTX_SEED="$seed" \
  RTX_MODEL_DIR="$MODEL_DIR" \
  RTX_MODEL_CONFIG="$MODEL_CONFIG" \
  RTX_EXTRA_MODEL_ARGS="$EXTRA_MODEL_ARGS" \
  RTX_NUM_ROLLOUT="$PILOT_STEPS" \
  RTX_BASELINE_MODE="$baseline" \
  RTX_SEAL="$seal" \
  RTX_GROUP_RM="$group_rm" \
  RTX_SEAL_AUTO_FIX="$auto_fix" \
  RTX_CUSTOM_RM="$custom_rm" \
  RTX_SGLANG_CONCURRENCY="$concurrency" \
  RTX_SAVE_INTERVAL="$save_interval" \
  RTX_PAPER_MODE=0 \
  RTX_ALLOW_REUSE=0 \
  RTX_SKIP_GATE=0 \
  RTX_EXP_ID="$exp_id" \
    bash "$BASE/scripts/phase2_run.sh" none -1 -1 "$PILOT_STEPS" "$exp_id" \
      2>&1 | tee "$launch_log"
  launch_exit=${PIPESTATUS[0]}
  set -e

  if [ "$launch_exit" -ne 0 ]; then
    end_time=$(date +%s)
    wall_time=$((end_time - start_time))
    record_run "$experiment" "$variant" "$baseline" "$exp_id" "$seed" \
      launch_failed "$wall_time" "$launch_exit" "$concurrency" "$checkpoint_mode"
    return "$launch_exit"
  fi

  set +e
  container_exit=$(docker wait "$container_name" 2>>"$launch_log")
  wait_exit=$?
  set -e
  end_time=$(date +%s)
  wall_time=$((end_time - start_time))

  if [ "$wait_exit" -ne 0 ]; then
    status=wait_failed
    container_exit=unknown
  elif [ "$container_exit" = "0" ]; then
    status=success
  else
    status=failed
  fi
  record_run "$experiment" "$variant" "$baseline" "$exp_id" "$seed" \
    "$status" "$wall_time" "$container_exit" "$concurrency" "$checkpoint_mode"
  [ "$status" = "success" ]
}

echo "=== P0 GPU pilot matrix ==="
echo "GPU slot: $PILOT_GPUS"
echo "Model: Qwen2.5-1.5B-Instruct; steps/run: $PILOT_STEPS"

for seed in "${SEEDS[@]}"; do
  run_pilot E3 b0 b0 "$seed" 64 on
  run_pilot E3 b6 b6 "$seed" 64 on
done

for seed in "${SEEDS[@]}"; do
  run_pilot E6 group-rm-only group_rm "$seed" 32 off
  run_pilot E6 rewardtxn b6 "$seed" 32 off
done

for seed in "${SEEDS[@]}"; do
  run_pilot E7 clean-oracle group_rm "$seed" 64 on
  run_pilot E7 clean-rewardtxn b6 "$seed" 64 on
done

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run only; no run directories or result files were created."
else
  python3 - "$PILOT_LOG" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
runs = payload["runs"]
print("Completed: {}/{} runs".format(sum(r["status"] == "success" for r in runs), len(runs)))
print("Wall time: {:.2f} hours".format(sum(r["wall_time_seconds"] for r in runs) / 3600.0))
PY
  echo "Results: $PILOT_LOG"
  echo "E7 power/precision analysis requires final_eval_accuracy for each paired run."
fi
