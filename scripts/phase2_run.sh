#!/usr/bin/env bash
# Phase 2 宿主入口: 协议层实验 (Seal/CAS/StepToken/Replay 包装层)
# 用法: bash scripts/phase2_run.sh <fault> <start_gidx> <end_gidx> [num_rollout] [exp_id]
# 环境: RTX_CUSTOM_RM 选择 RM 模块 (默认 day2_custom_rm), RTX_SEAL=1 启用 Seal+CAS
#   fault: skew | crm_crash | dup | none (none=仅 instrumentation 对照)
#   窗口:  group_index 范围 (全局递增, step = group_index // 4; 例: 注入 step 5-7 => 20 27)
# 产物: runs/p1-slime-{FAULT}-K8-s42-{date}/{meta.json, logs/, manifests/, rewards.jsonl, metrics.json}
set -euo pipefail

BASE=/public/home/caiyiwen/rewardtxn
FAULT=${1:?usage: day2_run.sh <fault> <start_gidx> <end_gidx> [num_rollout] [exp_id]}
START=${2:?start group_index}
END=${3:?end group_index}
NUM_ROLLOUT=${4:-20}
DATE=$(date +%Y%m%d)
EXP_ID=${5:-p1-slime-${FAULT}-K8-s42-${DATE}}
RUN_DIR="$BASE/runs/$EXP_ID"
NAME="rtx-p2-$EXP_ID"
GPUS=${RTX_GPUS:-"device=0,1,2,5"}

mkdir -p "$RUN_DIR"/{logs,manifests,checkpoints}

python3 - "$EXP_ID" "$RUN_DIR" "$FAULT" "$START" "$END" "$NUM_ROLLOUT" <<'EOF'
import json, sys, datetime
exp_id, run_dir, fault, start, end, nroll = sys.argv[1:7]
meta = {
    "exp_id": exp_id,
    "phase": "p2",
    "stack": "slime",
    "baseline": "B0",
    "commit_sha": "a6272da0d4f3d0a08520c99a2f3b4f6c887960dc",
    "seed": 42,
    "group_size_K": 8,
    "batch_groups_U": 4,
    "fault_injection": {
        "crash_point": {"skew": "R3", "crm_crash": "R1", "dup": "R2", "none": None}[fault],
        "mechanism": "custom_rm_path (${RTX_CUSTOM_RM:-day2_custom_rm.rm_function})",
        "target_step": f"group_index [{start},{end}] (step {int(start)//4}-{int(end)//4})",
        "window_group_start": int(start), "window_group_end": int(end),
    },
    "tolerances": {"max_l2_diff": 0.001, "min_cosine_similarity": 0.995},
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "params": {
        "num_rollout": int(nroll), "rollout_batch_size": 4, "n_samples_per_prompt": 8,
        "global_batch_size": 32, "lr": 1e-4, "kl_loss_coef": 0.01, "save_interval": 10,
        "model": "Qwen2.5-0.5B-Instruct", "dataset": "dapo-math-17k", "rm": "custom_rm_path",
        "gpus": [0, 1, 2, 5], "actor_gpus": 1, "rollout_gpus": 3,
    },
}
with open(f"{run_dir}/meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"meta.json: {run_dir}/meta.json (fault={fault}, window=[{start},{end}])")
EOF

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "$GPUS" -e NUM_GPUS=${RTX_NUM_GPUS:-4} \
  -e SAVE_DIR="${RTX_SAVE_DIR:-/workspace/runs/$EXP_ID/checkpoints}" \
  -e RTX_FAULT="$FAULT" -e RTX_FAULT_START="$START" -e RTX_FAULT_END="$END" \
  -e RTX_RUN_DIR="/workspace/runs/$EXP_ID" \
  -e RTX_NUM_ROLLOUT="$NUM_ROLLOUT" \
  -e RTX_CUSTOM_RM="${RTX_CUSTOM_RM:-day2_custom_rm.rm_function}" \
  -e DATA_PATH="${RTX_DATA_PATH:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}" \
  -e MODEL_DIR="${RTX_MODEL_DIR:-/root/models/Qwen2.5-0.5B-Instruct}" \
  -e RTX_MODEL_CONFIG="${RTX_MODEL_CONFIG:-qwen2.5-0.5B.sh}" \
  -e RTX_EXTRA_MODEL_ARGS="${RTX_EXTRA_MODEL_ARGS:-}" \
  -e RTX_SAVE_INTERVAL="${RTX_SAVE_INTERVAL:-10}" \
  -e RTX_V2_MODE="${RTX_V2_MODE:-strict}" \
  -e RTX_SEAL="${RTX_SEAL:-0}" \
  -e RTX_SEAL_AUTO_FIX="${RTX_SEAL_AUTO_FIX:-0}" \
  -e RTX_GROUP_RM="${RTX_GROUP_RM:-0}" \
  -e RTX_GROUP_SIZE="${RTX_GROUP_SIZE:-8}" \
  -e RTX_FAULT_WINDOWS="${RTX_FAULT_WINDOWS:-}" \
  -e RTX_LOG_RESPONSE="${RTX_LOG_RESPONSE:-0}" \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BASE":/workspace \
  -v "$BASE/third_party/slime":/root/slime \
  -v "$BASE/models":/root/models \
  -v "$BASE/models/datasets":/root/datasets \
  slimerl/slime:v0.3.1 \
  bash -c "
    set -e
    git config --global --add safe.directory /root/slime
    cd /root/slime
    pip install -e . --no-deps -q
    bash /workspace/scripts/day2_slime_train.sh
  "

echo "container $NAME started (fault=$FAULT, group_index [$START,$END])"
nohup docker logs -f "$NAME" > "$RUN_DIR/logs/train.log" 2>&1 &
echo "日志: $RUN_DIR/logs/train.log"
