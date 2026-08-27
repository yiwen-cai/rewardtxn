#!/usr/bin/env bash
# Day 1 宿主入口：4 卡 (0,1,2,5) 无故障 20 步基线 (B0, K=8, U=4)
# 用法: bash scripts/day1_run.sh [exp_id]
# 产物: runs/{exp_id}/{meta.json, logs/train.log, manifests/, checkpoints/, metrics.json}
set -euo pipefail

BASE=/public/home/caiyiwen/rewardtxn
EXP_ID=${1:-p1-slime-B0-K8-s42-20260824}
RUN_DIR="$BASE/runs/$EXP_ID"
NAME="rtx-day1-$EXP_ID"
GPUS='"device=0,1,2,5"'
NUM_GPUS=4

mkdir -p "$RUN_DIR"/{logs,manifests,checkpoints}

# ---------- meta.json (文档 10.2 规范) ----------
python3 - "$EXP_ID" "$RUN_DIR" <<'EOF'
import json, sys, datetime
exp_id, run_dir = sys.argv[1], sys.argv[2]
meta = {
    "exp_id": exp_id,
    "phase": "p1",
    "stack": "slime",
    "baseline": "B0",
    "commit_sha": "a6272da0d4f3d0a08520c99a2f3b4f6c887960dc",
    "seed": 42,
    "group_size_K": 8,
    "batch_groups_U": 4,
    "fault_injection": {"crash_point": None, "mechanism": None, "target_step": None},
    "tolerances": {"max_l2_diff": 0.001, "min_cosine_similarity": 0.995},
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "params": {
        "num_rollout": 20,
        "rollout_batch_size": 4,
        "n_samples_per_prompt": 8,
        "global_batch_size": 32,
        "lr": 1e-4,
        "kl_loss_coef": 0.01,
        "save_interval": 10,
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

# ---------- docker run (锁定源码 + 模型 + 数据挂载) ----------
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "$GPUS" -e NUM_GPUS=$NUM_GPUS \
  -e SAVE_DIR="/workspace/runs/$EXP_ID/checkpoints" \
  --ipc=host --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
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
    bash /workspace/scripts/day1_slime_train.sh
  "

echo "container $NAME started on GPUs 0,1,2,5"
echo "tailing logs -> $RUN_DIR/logs/train.log"
nohup docker logs -f "$NAME" > "$RUN_DIR/logs/train.log" 2>&1 &
echo "log follow PID $!"
echo "监控: docker logs -f $NAME"
