#!/usr/bin/env bash
# Day 2 宿主入口: Reward/Verifier 层故障注入 (文档 7 节 Day 2)
# 用法: bash scripts/day2_run.sh <fault> <start_gidx> <end_gidx> [num_rollout] [exp_id]
#   fault: skew | crm_crash | dup | none (none=仅 instrumentation 对照)
#   窗口:  group_index 范围 (全局递增, step = group_index // 4; 例: 注入 step 5-7 => 20 27)
# 产物: runs/p1-slime-{FAULT}-K8-s42-{date}/{meta.json, logs/, manifests/, rewards.jsonl, metrics.json}
set -euo pipefail

BASE=/public/home/caiyiwen/rewardtxn
if [ -n "${RTX_PROFILE:-}" ]; then
  source "$BASE/scripts/experiment_profiles.sh"
fi
FAULT=${1:?usage: day2_run.sh <fault> <start_gidx> <end_gidx> [num_rollout] [exp_id]}
START=${2:?start group_index}
END=${3:?end group_index}
NUM_ROLLOUT=${RTX_NUM_ROLLOUT:-${4:-20}}
DATE=${RTX_DATE:-$(date +%Y%m%d-%H%M%S)}
EXP_ID=${RTX_EXP_ID:-${5:-p1-slime-${FAULT}-K8-s42-${DATE}}}
RUN_DIR="$BASE/runs/$EXP_ID"
NAME="rtx-day2-$EXP_ID"
GPUS=${RTX_GPUS:-device=0,1,2,5}
case "$GPUS" in
  '"'*) : ;;                      # 已是 JSON 带引号形式
  *) GPUS="\"$GPUS\"" ;;       # docker --gpus 需要 "device=..." 形式
esac
# Keep Ray spill and the small CAS index on the local/root NVMe by default;
# high-volume experiment artifacts remain under runs/ for audit/replay.
LOCAL_SCRATCH=${RTX_LOCAL_SCRATCH:-/tmp/rewardtxn}
if [ "${RTX_ALLOW_REUSE:-0}" != "1" ] && [ -d "$RUN_DIR" ] \
   && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "run directory already contains data: $RUN_DIR (choose a new exp_id or set RTX_ALLOW_REUSE=1)" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"/{logs,manifests,checkpoints} "$LOCAL_SCRATCH/$EXP_ID"/{ray,cas}

COMMIT_SHA=${RTX_COMMIT_SHA:-$(git -C "$BASE" rev-parse HEAD 2>/dev/null || echo unknown)}
python3 - "$EXP_ID" "$RUN_DIR" "$FAULT" "$START" "$END" "$NUM_ROLLOUT" "$COMMIT_SHA" <<'EOF'
import json, os, sys, datetime
exp_id, run_dir, fault, start, end, nroll, commit_sha = sys.argv[1:8]
meta = {
    "exp_id": exp_id,
    "phase": "p1",
    "stack": "slime",
    "baseline": "B0",
    "commit_sha": commit_sha,
    "seed": 42,
    "group_size_K": 8,
    "batch_groups_U": 4,
    "fault_injection": {
        "crash_point": {"skew": "R3", "crm_crash": "R1", "dup": "R2", "none": None}[fault],
        "mechanism": "custom_rm_path (day2_custom_rm.py)",
        "target_step": f"group_index [{start},{end}] (step {int(start)//4}-{int(end)//4})",
    },
    "tolerances": {"max_l2_diff": 0.001, "min_cosine_similarity": 0.995},
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "params": {
        "num_rollout": int(nroll), "rollout_batch_size": 4, "n_samples_per_prompt": 8,
        "global_batch_size": 32, "lr": 1e-4, "kl_loss_coef": 0.01,
        "save_interval": int(os.environ.get("RTX_SAVE_INTERVAL", "10")),
        "sglang_server_concurrency": int(os.environ.get("RTX_SGLANG_CONCURRENCY", "64")),
        "ray_object_store_memory": int(os.environ.get("RTX_RAY_OBJECT_STORE_MEMORY", "17179869184")),
        "checkpoint_keep": int(os.environ.get("RTX_CKPT_KEEP", "2")),
        "no_save_optim": os.environ.get("RTX_NO_SAVE_OPTIM", "0") == "1",
        "model": "Qwen2.5-0.5B-Instruct", "dataset": "dapo-math-17k", "rm": "custom_rm_path",
        "gpus": [0, 1, 2, 5], "actor_gpus": 1, "rollout_gpus": 3,
    },
}
with open(f"{run_dir}/meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"meta.json: {run_dir}/meta.json (fault={fault}, window=[{start},{end}])")
EOF

if [ "${RTX_META_ONLY:-0}" = "1" ]; then
  echo "meta-only mode: skipping docker launch (RTX_META_ONLY=1)"
  exit 0
fi

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
  -e RTX_SGLANG_CONCURRENCY="${RTX_SGLANG_CONCURRENCY:-64}" \
  -e RTX_RAY_OBJECT_STORE_MEMORY="${RTX_RAY_OBJECT_STORE_MEMORY:-17179869184}" \
  -e RTX_RAY_PLASMA_DIR="${RTX_RAY_PLASMA_DIR:-/dev/shm}" \
  -e RTX_RAY_TMP_DIR="${RTX_RAY_TMP_DIR:-/rtx-scratch/ray}" \
  -e RTX_RAY_SPILL_DIR="${RTX_RAY_SPILL_DIR:-/rtx-scratch/$EXP_ID/ray/spill}" \
  -e RTX_CAS_INDEX_DIR="${RTX_CAS_INDEX_DIR:-/rtx-scratch/$EXP_ID/cas}" \
  -e RTX_NO_SAVE_OPTIM="${RTX_NO_SAVE_OPTIM:-0}" \
  -e RTX_CKPT_KEEP="${RTX_CKPT_KEEP:-2}" \
  -e RTX_CKPT_RETENTION_INTERVAL="${RTX_CKPT_RETENTION_INTERVAL:-15}" \
  -e RTX_CKPT_RETENTION_MIN_AGE="${RTX_CKPT_RETENTION_MIN_AGE:-30}" \
  -e RTX_FULLY_ASYNC="${RTX_FULLY_ASYNC:-1}" \
  -e RTX_MAX_TOKENS_PER_GPU="${RTX_MAX_TOKENS_PER_GPU:-4096}" \
  -e RTX_SGLANG_MEM_FRACTION_STATIC="${RTX_SGLANG_MEM_FRACTION_STATIC:-0.55}" \
  -e RTX_V2_MODE="${RTX_V2_MODE:-strict}" \
  -e RTX_LOG_RESPONSE="${RTX_LOG_RESPONSE:-0}" \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BASE":/workspace \
  -v "$LOCAL_SCRATCH":/rtx-scratch \
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
