#!/usr/bin/env bash
# Day 1 宿主入口：4 卡 (0,1,2,5) 无故障 20 步基线 (B0, K=8, U=4)
# 用法: bash scripts/day1_run.sh [exp_id]
# 产物: runs/{exp_id}/{meta.json, logs/train.log, manifests/, checkpoints/, metrics.json}
set -euo pipefail

BASE=/public/home/caiyiwen/rewardtxn
if [ -n "${RTX_PROFILE:-}" ]; then
  source "$BASE/scripts/experiment_profiles.sh"
fi
DATE=${RTX_DATE:-$(date +%Y%m%d-%H%M%S)}
EXP_ID=${RTX_EXP_ID:-${1:-p1-slime-B0-K8-s42-$DATE}}
RUN_DIR="$BASE/runs/$EXP_ID"
NAME="rtx-day1-$EXP_ID"
GPUS=${RTX_GPUS:-'"device=0,1,2,5"'}
NUM_GPUS=${RTX_NUM_GPUS:-4}
NUM_ROLLOUT=${RTX_NUM_ROLLOUT:-20}
LOCAL_SCRATCH=${RTX_LOCAL_SCRATCH:-/tmp/rewardtxn}

if [ "${RTX_ALLOW_REUSE:-0}" != "1" ] && [ -d "$RUN_DIR" ] \
   && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "run directory already contains data: $RUN_DIR (choose a new exp_id or set RTX_ALLOW_REUSE=1)" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"/{logs,manifests,checkpoints} "$LOCAL_SCRATCH/$EXP_ID"/{ray,cas}

# ---------- meta.json (文档 10.2 规范) ----------
python3 - "$EXP_ID" "$RUN_DIR" "$NUM_ROLLOUT" "$COMMIT_SHA" <<'EOF'
import json, os, sys, datetime
exp_id, run_dir, nroll, commit_sha = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
meta = {
    "exp_id": exp_id,
    "phase": "p1",
    "stack": "slime",
    "baseline": "B0",
    "commit_sha": commit_sha,
    "seed": 42,
    "group_size_K": 8,
    "batch_groups_U": 4,
    "fault_injection": {"crash_point": None, "mechanism": None, "target_step": None},
    "tolerances": {"max_l2_diff": 0.001, "min_cosine_similarity": 0.995},
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "params": {
        "num_rollout": int(nroll),
        "rollout_batch_size": 4,
        "n_samples_per_prompt": 8,
        "global_batch_size": 32,
        "lr": 1e-4,
        "kl_loss_coef": 0.01,
        "save_interval": int(os.environ.get("RTX_SAVE_INTERVAL", "10")),
        "sglang_server_concurrency": int(os.environ.get("RTX_SGLANG_CONCURRENCY", "64")),
        "ray_object_store_memory": int(os.environ.get("RTX_RAY_OBJECT_STORE_MEMORY", "17179869184")),
        "checkpoint_keep": int(os.environ.get("RTX_CKPT_KEEP", "2")),
        "no_save_optim": os.environ.get("RTX_NO_SAVE_OPTIM", "0") == "1",
        "model": "Qwen2.5-0.5B-Instruct",
        "dataset": "dapo-math-17k",
        "rm": "deepscaler",
        "gpus": [0, 1, 2, 5],
        "actor_gpus": 1,
        "rollout_gpus": 3,
    },
}
with open(f"{run_dir}/meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
print(f"meta.json written: {run_dir}/meta.json")
EOF

if [ "${RTX_META_ONLY:-0}" = "1" ]; then
  echo "meta-only mode: skipping docker launch (RTX_META_ONLY=1)"
  exit 0
fi

# ---------- docker run (锁定源码 + 模型 + 数据挂载) ----------
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "$GPUS" -e NUM_GPUS=$NUM_GPUS \
  -e SAVE_DIR="${RTX_SAVE_DIR:-/workspace/runs/$EXP_ID/checkpoints}" \
  -e RTX_SAVE_INTERVAL="${RTX_SAVE_INTERVAL:-10}" \
  -e RTX_SGLANG_CONCURRENCY="${RTX_SGLANG_CONCURRENCY:-64}" \
  -e RTX_RAY_OBJECT_STORE_MEMORY="${RTX_RAY_OBJECT_STORE_MEMORY:-17179869184}" \
  -e RTX_RAY_PLASMA_DIR="${RTX_RAY_PLASMA_DIR:-/dev/shm}" \
  -e RTX_RAY_TMP_DIR="${RTX_RAY_TMP_DIR:-/rtx-scratch/$EXP_ID/ray}" \
  -e RTX_RAY_SPILL_DIR="${RTX_RAY_SPILL_DIR:-/rtx-scratch/$EXP_ID/ray/spill}" \
  -e RTX_NO_SAVE_OPTIM="${RTX_NO_SAVE_OPTIM:-0}" \
  -e RTX_CKPT_KEEP="${RTX_CKPT_KEEP:-2}" \
  -e RTX_CKPT_RETENTION_INTERVAL="${RTX_CKPT_RETENTION_INTERVAL:-15}" \
  -e RTX_CKPT_RETENTION_MIN_AGE="${RTX_CKPT_RETENTION_MIN_AGE:-30}" \
  -e RTX_FULLY_ASYNC="${RTX_FULLY_ASYNC:-1}" \
  -e RTX_MAX_TOKENS_PER_GPU="${RTX_MAX_TOKENS_PER_GPU:-4096}" \
  -e RTX_SGLANG_MEM_FRACTION_STATIC="${RTX_SGLANG_MEM_FRACTION_STATIC:-0.55}" \
  --ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
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
    bash /workspace/scripts/day1_slime_train.sh
  "

echo "container $NAME started on GPUs ${GPUS}"
echo "tailing logs -> $RUN_DIR/logs/train.log"
nohup docker logs -f "$NAME" > "$RUN_DIR/logs/train.log" 2>&1 &
echo "log follow PID $!"
echo "监控: docker logs -f $NAME"
