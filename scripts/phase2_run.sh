#!/usr/bin/env bash
# Phase 2 宿主入口: 协议层实验 (Seal/CAS/StepToken/Replay 包装层)
# 用法: bash scripts/phase2_run.sh <fault> <start_gidx> <end_gidx> [num_rollout] [exp_id]
# 环境: RTX_GPUS / RTX_SEED / RTX_BASELINE_MODE / RTX_SCHEDULE 显式选择实验配置
# 正式论文模式: RTX_PAPER_MODE=1 且 RTX_FREEZE_MANIFEST 指向已验证冻结清单
#   fault: skew | crm_crash | dup | none (none=仅 instrumentation 对照)
#   窗口:  group_index 范围 (全局递增, step = group_index // 4; 例: 注入 step 5-7 => 20 27)
# 产物: runs/<exp_id>/{meta.json,config.json,logs/,manifests/,checkpoints/}
set -euo pipefail

BASE=${RTX_BASE:-/public/home/caiyiwen/rewardtxn}
if [ -n "${RTX_PROFILE:-}" ]; then
  source "$BASE/scripts/experiment_profiles.sh"
fi
FAULT=${1:?usage: phase2_run.sh <fault> <start_gidx> <end_gidx> [num_rollout] [exp_id]}
START=${2:?start group_index}
END=${3:?end group_index}
NUM_ROLLOUT=${RTX_NUM_ROLLOUT:-${4:-20}}
DATE=${RTX_DATE:-$(date +%Y%m%d-%H%M%S)}
PAPER_MODE=${RTX_PAPER_MODE:-0}
SEED=${RTX_SEED:-42}
BASELINE_MODE=$(printf '%s' "${RTX_BASELINE_MODE:-b0}" | tr '[:upper:]' '[:lower:]')
IMAGE=${RTX_IMAGE:-slimerl/slime:v0.3.1}
RAW_GPUS=${RTX_GPUS:-device=0,1,2,5}
MODEL_DIR=${RTX_MODEL_DIR:-/root/models/Qwen2.5-0.5B-Instruct}
DATA_PATH=${RTX_DATA_PATH:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}
MODEL_CONFIG=${RTX_MODEL_CONFIG:-qwen2.5-0.5B.sh}
SLIME_SOURCE_DIR=${RTX_SLIME_SOURCE_DIR:-$BASE/third_party/slime}
SLIME_SOURCE_DIR=$(realpath -e "${SLIME_SOURCE_DIR}") \
  || { echo "RTX_SLIME_SOURCE_DIR does not exist: ${RTX_SLIME_SOURCE_DIR}" >&2; exit 2; }
EXP_ID=${RTX_EXP_ID:-${5:-p1-slime-${FAULT}-K8-s${SEED}-${DATE}}}
RUN_DIR="$BASE/runs/$EXP_ID"
NAME="rtx-p2-$EXP_ID"
# Keep Ray spill and the small CAS index on the local/root NVMe by default;
# high-volume experiment artifacts remain under runs/ for audit/replay.
LOCAL_SCRATCH=${RTX_LOCAL_SCRATCH:-/tmp/rewardtxn}

case "${PAPER_MODE}" in 0|1) ;; *) echo "RTX_PAPER_MODE must be 0 or 1" >&2; exit 2 ;; esac
case "${SEED}" in ''|*[!0-9]*) echo "RTX_SEED must be a non-negative integer" >&2; exit 2 ;; esac
case "${FAULT}" in skew|crm_crash|dup|none) ;; *) echo "unknown fault: ${FAULT}" >&2; exit 2 ;; esac
case "${BASELINE_MODE}" in
  b0)
    SEAL=${RTX_SEAL:-0}; GROUP_RM=${RTX_GROUP_RM:-0}; SEAL_AUTO_FIX=${RTX_SEAL_AUTO_FIX:-0}
    [ "${SEAL}/${GROUP_RM}/${SEAL_AUTO_FIX}" = "0/0/0" ] \
      || { echo "b0 requires RTX_SEAL=0 RTX_GROUP_RM=0 RTX_SEAL_AUTO_FIX=0" >&2; exit 2; }
    CUSTOM_RM=${RTX_CUSTOM_RM:-day2_custom_rm.rm_function}
    if [ "${PAPER_MODE}" = "1" ] && [ "${CUSTOM_RM}" != "day2_custom_rm.rm_function" ]; then
      echo "formal b0 requires RTX_CUSTOM_RM=day2_custom_rm.rm_function" >&2
      exit 2
    fi
    ;;
  b6)
    SEAL=${RTX_SEAL:-1}; GROUP_RM=${RTX_GROUP_RM:-1}; SEAL_AUTO_FIX=${RTX_SEAL_AUTO_FIX:-1}
    [ "${SEAL}/${GROUP_RM}/${SEAL_AUTO_FIX}" = "1/1/1" ] \
      || { echo "b6 requires RTX_SEAL=1 RTX_GROUP_RM=1 RTX_SEAL_AUTO_FIX=1" >&2; exit 2; }
    CUSTOM_RM=${RTX_CUSTOM_RM:-phase2_seal_rm.rm_function}
    if [ "${PAPER_MODE}" = "1" ] && [ "${CUSTOM_RM}" != "phase2_seal_rm.rm_function" ]; then
      echo "formal b6 requires RTX_CUSTOM_RM=phase2_seal_rm.rm_function" >&2
      exit 2
    fi
    ;;
  group_rm)
    SEAL=${RTX_SEAL:-0}; GROUP_RM=${RTX_GROUP_RM:-1}; SEAL_AUTO_FIX=${RTX_SEAL_AUTO_FIX:-0}
    [ "${SEAL}/${GROUP_RM}/${SEAL_AUTO_FIX}" = "0/1/0" ] \
      || { echo "group_rm requires RTX_SEAL=0 RTX_GROUP_RM=1 RTX_SEAL_AUTO_FIX=0" >&2; exit 2; }
    CUSTOM_RM=${RTX_CUSTOM_RM:-day2_custom_rm.rm_function}
    [ "${CUSTOM_RM}" = "day2_custom_rm.rm_function" ] \
      || { echo "group_rm requires RTX_CUSTOM_RM=day2_custom_rm.rm_function" >&2; exit 2; }
    ;;
  b1|b2|b3|b4|b5)
    if [ "${PAPER_MODE}" = "1" ]; then
      echo "baseline ${BASELINE_MODE}: mechanism implementation missing; formal run refused" >&2
      exit 2
    fi
    SEAL=${RTX_SEAL:-0}; GROUP_RM=${RTX_GROUP_RM:-0}; SEAL_AUTO_FIX=${RTX_SEAL_AUTO_FIX:-0}
    CUSTOM_RM=${RTX_CUSTOM_RM:-day2_custom_rm.rm_function}
    ;;
  *) echo "RTX_BASELINE_MODE must be b0..b6 or group_rm (got ${BASELINE_MODE})" >&2; exit 2 ;;
esac

GPU_CSV=$(python3 - "${RAW_GPUS}" <<'EOF'
import sys
raw = sys.argv[1].strip().strip('"').strip("'")
if raw.startswith("device="):
    raw = raw[len("device="):]
parts = [part.strip() for part in raw.split(",") if part.strip()]
if not parts or any(not part.isdigit() for part in parts):
    raise SystemExit("RTX_GPUS must be an explicit device list, for example device=0,1,2,5")
values = [int(part) for part in parts]
if len(values) != len(set(values)):
    raise SystemExit("RTX_GPUS contains duplicate device indices")
print(",".join(str(value) for value in values))
EOF
)
GPU_COUNT=$(python3 - "${GPU_CSV}" <<'EOF'
import sys
print(len(sys.argv[1].split(",")))
EOF
)
NUM_GPUS=${RTX_NUM_GPUS:-${GPU_COUNT}}
[ "${NUM_GPUS}" = "${GPU_COUNT}" ] \
  || { echo "RTX_NUM_GPUS=${NUM_GPUS} does not match RTX_GPUS count ${GPU_COUNT}" >&2; exit 2; }
DOCKER_GPUS="\"device=${GPU_CSV}\""

host_path_for_container() {
  case "$1" in
    /workspace/*) printf '%s/%s\n' "$BASE" "${1#/workspace/}" ;;
    /root/models/*) printf '%s/models/%s\n' "$BASE" "${1#/root/models/}" ;;
    /root/datasets/*) printf '%s/models/datasets/%s\n' "$BASE" "${1#/root/datasets/}" ;;
    *) return 1 ;;
  esac
}

MODEL_HOST=$(host_path_for_container "${MODEL_DIR}") \
  || { echo "RTX_MODEL_DIR must be under /root/models or /workspace" >&2; exit 2; }
TORCH_DIST_HOST="${MODEL_HOST}_torch_dist"
DATA_HOST=$(host_path_for_container "${DATA_PATH}") \
  || { echo "RTX_DATA_PATH must be under /root/datasets or /workspace" >&2; exit 2; }

SCHEDULE_HOST=""
SCHEDULE_CONTAINER=""
if [ -n "${RTX_SCHEDULE:-}" ]; then
  case "${RTX_SCHEDULE}" in /*) SCHEDULE_HOST=${RTX_SCHEDULE} ;; *) SCHEDULE_HOST="$BASE/${RTX_SCHEDULE}" ;; esac
  SCHEDULE_HOST=$(realpath -e "${SCHEDULE_HOST}") \
    || { echo "RTX_SCHEDULE does not exist: ${RTX_SCHEDULE}" >&2; exit 2; }
  case "${SCHEDULE_HOST}" in
    "$BASE"/*) SCHEDULE_CONTAINER="/workspace/${SCHEDULE_HOST#${BASE}/}" ;;
    *) echo "RTX_SCHEDULE must be inside ${BASE} so it is mounted in the container" >&2; exit 2 ;;
  esac
fi

FREEZE_MANIFEST=""
FREEZE_VERIFY_OUTPUT=""
if [ "${PAPER_MODE}" = "1" ]; then
  [ -n "${RTX_GPUS+x}" ] || { echo "formal run requires explicit RTX_GPUS" >&2; exit 2; }
  [ -n "${RTX_SEED+x}" ] || { echo "formal run requires explicit RTX_SEED" >&2; exit 2; }
  [ -n "${RTX_BASELINE_MODE+x}" ] || { echo "formal run requires explicit RTX_BASELINE_MODE" >&2; exit 2; }
  [ -n "${SCHEDULE_HOST}" ] || { echo "formal run requires RTX_SCHEDULE" >&2; exit 2; }
  [ "${RTX_SKIP_GATE:-0}" != "1" ] || { echo "formal run cannot set RTX_SKIP_GATE=1" >&2; exit 2; }
  [ "${RTX_ALLOW_REUSE:-0}" != "1" ] || { echo "formal run cannot reuse an existing run directory" >&2; exit 2; }
  FREEZE_MANIFEST=${RTX_FREEZE_MANIFEST:?formal run requires RTX_FREEZE_MANIFEST}
  case "${FREEZE_MANIFEST}" in /*) ;; *) FREEZE_MANIFEST="$BASE/${FREEZE_MANIFEST}" ;; esac
  FREEZE_MANIFEST=$(realpath -e "${FREEZE_MANIFEST}") \
    || { echo "freeze manifest does not exist: ${RTX_FREEZE_MANIFEST}" >&2; exit 2; }
  if ! FREEZE_VERIFY_OUTPUT=$(python3 "$BASE/scripts/freeze_paper_inputs.py" verify \
      --manifest "${FREEZE_MANIFEST}" --require-image "${IMAGE}" \
      --require-repo-path "slime=${SLIME_SOURCE_DIR}" \
      --require-asset-path "${MODEL_HOST}" --require-asset-path "${TORCH_DIST_HOST}" \
      --require-asset-path "${DATA_HOST}" --require-asset-path "${SCHEDULE_HOST}"); then
    echo "formal input freeze verification FAILED" >&2
    printf '%s\n' "${FREEZE_VERIFY_OUTPUT}" >&2
    exit 2
  fi
fi

if [ "${RTX_ALLOW_REUSE:-0}" != "1" ] && [ -d "$RUN_DIR" ] \
   && [ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "run directory already contains data: $RUN_DIR (choose a new exp_id or set RTX_ALLOW_REUSE=1)" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"/{logs,manifests,checkpoints} "$LOCAL_SCRATCH/$EXP_ID"/{ray,cas}
PROVENANCE_ARGS=(--repo "slime=$SLIME_SOURCE_DIR" --ignore-path "$RUN_DIR")
if [ -n "${FREEZE_MANIFEST}" ]; then
  PROVENANCE_ARGS+=(--ignore-path "$FREEZE_MANIFEST")
fi
python3 "$BASE/scripts/freeze_paper_inputs.py" provenance "${PROVENANCE_ARGS[@]}" \
  --out "$RUN_DIR/logs/source_provenance.json" >/dev/null
if [ -n "${FREEZE_VERIFY_OUTPUT}" ]; then
  printf '%s\n' "${FREEZE_VERIFY_OUTPUT}" > "$RUN_DIR/logs/freeze_verify.json"
fi

# ---- 资源健康门禁: 存储/显存不健康拒绝启动 (RTX_SKIP_GATE=1 可跳过) ----
if [ "${RTX_SKIP_GATE:-0}" != "1" ]; then
  python3 "$BASE/scripts/resource_gate.py" check --gpus "$DOCKER_GPUS" \
    --scratch-path "$LOCAL_SCRATCH" --out "$RUN_DIR/logs/resource_gate.json" \
    || { echo "resource gate FAILED: storage/GPU memory/utilization/process conflict, refusing launch" >&2; exit 2; }
fi

export RTX_EFFECTIVE_EXP_ID="$EXP_ID" RTX_EFFECTIVE_RUN_DIR="$RUN_DIR"
export RTX_EFFECTIVE_FAULT="$FAULT" RTX_EFFECTIVE_START="$START" RTX_EFFECTIVE_END="$END"
export RTX_EFFECTIVE_NUM_ROLLOUT="$NUM_ROLLOUT" RTX_EFFECTIVE_GPU_CSV="$GPU_CSV"
export RTX_EFFECTIVE_NUM_GPUS="$NUM_GPUS" RTX_EFFECTIVE_SEED="$SEED"
export RTX_EFFECTIVE_BASELINE="$BASELINE_MODE" RTX_EFFECTIVE_IMAGE="$IMAGE"
export RTX_EFFECTIVE_MODEL_DIR="$MODEL_DIR" RTX_EFFECTIVE_DATA_PATH="$DATA_PATH"
export RTX_EFFECTIVE_MODEL_CONFIG="$MODEL_CONFIG" RTX_EFFECTIVE_SCHEDULE_HOST="$SCHEDULE_HOST"
export RTX_EFFECTIVE_SLIME_SOURCE_DIR="$SLIME_SOURCE_DIR"
export RTX_EFFECTIVE_SCHEDULE_CONTAINER="$SCHEDULE_CONTAINER" RTX_EFFECTIVE_FREEZE_MANIFEST="$FREEZE_MANIFEST"
export RTX_EFFECTIVE_SEAL="$SEAL" RTX_EFFECTIVE_GROUP_RM="$GROUP_RM"
export RTX_EFFECTIVE_SEAL_AUTO_FIX="$SEAL_AUTO_FIX" RTX_EFFECTIVE_CUSTOM_RM="$CUSTOM_RM"
python3 - <<'EOF'
import datetime, hashlib, json, os

exp_id = os.environ["RTX_EFFECTIVE_EXP_ID"]
run_dir = os.environ["RTX_EFFECTIVE_RUN_DIR"]
fault = os.environ["RTX_EFFECTIVE_FAULT"]
start = os.environ["RTX_EFFECTIVE_START"]
end = os.environ["RTX_EFFECTIVE_END"]
nroll = os.environ["RTX_EFFECTIVE_NUM_ROLLOUT"]
provenance = json.load(open(os.path.join(run_dir, "logs", "source_provenance.json")))
repositories = provenance["repositories"]
commit_sha = repositories.get("rewardtxn", {}).get("commit_sha", "unknown")
schedule_path = os.environ["RTX_EFFECTIVE_SCHEDULE_HOST"]
schedule = None
if schedule_path:
    with open(schedule_path, "rb") as stream:
        schedule_hash = hashlib.sha256(stream.read()).hexdigest()
    try:
        schedule_id = json.load(open(schedule_path)).get("schedule_id")
    except (ValueError, AttributeError):
        schedule_id = None
    schedule = {"path": os.path.relpath(schedule_path, os.environ.get("RTX_BASE", "/public/home/caiyiwen/rewardtxn")),
                "container_path": os.environ["RTX_EFFECTIVE_SCHEDULE_CONTAINER"],
                "schedule_id": schedule_id, "sha256": schedule_hash}
freeze_manifest_path = os.environ["RTX_EFFECTIVE_FREEZE_MANIFEST"]
freeze_manifest = None
if freeze_manifest_path:
    with open(freeze_manifest_path, "rb") as stream:
        freeze_manifest = {"path": freeze_manifest_path,
                           "sha256": hashlib.sha256(stream.read()).hexdigest()}
meta = {
    "exp_id": exp_id,
    "phase": "p2",
    "stack": "slime",
    "baseline": os.environ["RTX_EFFECTIVE_BASELINE"].upper(),
    "baseline_mode": os.environ["RTX_EFFECTIVE_BASELINE"],
    "baseline_effective": {
        "seal": os.environ["RTX_EFFECTIVE_SEAL"] == "1",
        "group_rm": os.environ["RTX_EFFECTIVE_GROUP_RM"] == "1",
        "seal_auto_fix": os.environ["RTX_EFFECTIVE_SEAL_AUTO_FIX"] == "1",
        "custom_rm": os.environ["RTX_EFFECTIVE_CUSTOM_RM"],
    },
    "commit_sha": commit_sha,
    "source_repositories": repositories,
    "seed": int(os.environ["RTX_EFFECTIVE_SEED"]),
    "image": os.environ["RTX_EFFECTIVE_IMAGE"],
    "paper_mode": os.environ.get("RTX_PAPER_MODE", "0") == "1",
    "freeze_manifest": freeze_manifest,
    "fault_schedule": schedule,
    "group_size_K": int(os.environ.get("RTX_GROUP_SIZE", "8")),
    "batch_groups_U": 4,
    "fault_injection": {
        "crash_point": {"skew": "R3", "crm_crash": "R1", "dup": "R2", "none": None}[fault],
        "mechanism": "custom_rm_path ({})".format(os.environ["RTX_EFFECTIVE_CUSTOM_RM"]),
        "target_step": f"group_index [{start},{end}] (step {int(start)//4}-{int(end)//4})",
        "window_group_start": int(start), "window_group_end": int(end),
    },
    "tolerances": {"max_l2_diff": 0.001, "min_cosine_similarity": 0.995},
    "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
    "params": {
        "num_rollout": int(nroll), "rollout_batch_size": 4, "n_samples_per_prompt": 8,
        "global_batch_size": 32, "lr": float(os.environ.get("RTX_LR", "5e-5")), "kl_loss_coef": 0.01,
        "train_entry": os.environ.get("RTX_TRAIN_ENTRY", "train_async.py"),
        "use_rollout_logprobs": "--use-rollout-logprobs" in os.environ.get("RTX_EXTRA_MODEL_ARGS", "").split(),
        "restart_protocol_sha256": os.environ.get("RTX_RESTART_PROTOCOL_SHA256"),
        "fully_async_rollout": os.environ.get("RTX_FULLY_ASYNC", "1") == "1",
        "save_hf": os.environ.get("RTX_SAVE_HF", "0") == "1",
        "save_interval": int(os.environ.get("RTX_SAVE_INTERVAL", "10")),
        "sglang_server_concurrency": int(os.environ.get("RTX_SGLANG_CONCURRENCY", "24")),
        "ray_object_store_memory": int(os.environ.get("RTX_RAY_OBJECT_STORE_MEMORY", "17179869184")),
        "checkpoint_keep": int(os.environ.get("RTX_CKPT_KEEP", "2")),
        "no_save_optim": os.environ.get("RTX_NO_SAVE_OPTIM", "0") == "1",
        "model": os.environ["RTX_EFFECTIVE_MODEL_DIR"].rstrip("/").split("/")[-1],
        "model_path": os.environ["RTX_EFFECTIVE_MODEL_DIR"],
        "dataset_path": os.environ["RTX_EFFECTIVE_DATA_PATH"],
        "rm": os.environ["RTX_EFFECTIVE_CUSTOM_RM"],
        "gpus": [int(value) for value in os.environ["RTX_EFFECTIVE_GPU_CSV"].split(",")],
        "gpu_request": "device=" + os.environ["RTX_EFFECTIVE_GPU_CSV"],
        "num_gpus": int(os.environ["RTX_EFFECTIVE_NUM_GPUS"]),
        "actor_gpus": int(os.environ.get("RTX_ACTOR_GPUS", "1")),
        "rollout_gpus": int(os.environ.get("RTX_ROLLOUT_GPUS", str(int(os.environ["RTX_EFFECTIVE_NUM_GPUS"]) - 1))),
    },
}
with open(f"{run_dir}/meta.json", "w") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
config = {
    "exp_id": exp_id,
    "image": os.environ["RTX_EFFECTIVE_IMAGE"],
    "gpus": meta["params"]["gpus"],
    "seed": meta["seed"],
    "baseline_mode": meta["baseline_mode"],
    "baseline_effective": meta["baseline_effective"],
    "schedule": schedule,
    "model_dir": os.environ["RTX_EFFECTIVE_MODEL_DIR"],
    "model_config": os.environ["RTX_EFFECTIVE_MODEL_CONFIG"],
    "slime_source_dir": os.environ["RTX_EFFECTIVE_SLIME_SOURCE_DIR"],
    "data_path": os.environ["RTX_EFFECTIVE_DATA_PATH"],
    "num_rollout": int(nroll),
    "fault": meta["fault_injection"],
    "paper_mode": meta["paper_mode"],
}
with open(f"{run_dir}/config.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"meta.json: {run_dir}/meta.json (fault={fault}, window=[{start},{end}])")
EOF

if [ "${RTX_META_ONLY:-0}" = "1" ]; then
  echo "meta-only mode: skipping docker launch (RTX_META_ONLY=1)"
  exit 0
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus "$DOCKER_GPUS" -e NUM_GPUS="$NUM_GPUS" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e ACTOR_GPUS="${RTX_ACTOR_GPUS:-1}" \
  -e ROLLOUT_GPUS="${RTX_ROLLOUT_GPUS:-$((NUM_GPUS - 1))}" \
  -e SAVE_DIR="${RTX_SAVE_DIR:-/workspace/runs/$EXP_ID/checkpoints}" \
  -e RTX_FAULT="$FAULT" -e RTX_FAULT_START="$START" -e RTX_FAULT_END="$END" \
  -e RTX_RUN_DIR="/workspace/runs/$EXP_ID" \
  -e RTX_NUM_ROLLOUT="$NUM_ROLLOUT" \
  -e RTX_CUSTOM_RM="$CUSTOM_RM" \
  -e DATA_PATH="$DATA_PATH" \
  -e MODEL_DIR="$MODEL_DIR" \
  -e RTX_MODEL_CONFIG="$MODEL_CONFIG" \
  -e RTX_SEED="$SEED" \
  -e RTX_BASELINE_MODE="$BASELINE_MODE" \
  -e RTX_SCHEDULE="$SCHEDULE_CONTAINER" \
  -e RTX_PAPER_MODE="$PAPER_MODE" \
  -e RTX_EXTRA_MODEL_ARGS="${RTX_EXTRA_MODEL_ARGS:-}" \
  -e RTX_SAVE_INTERVAL="${RTX_SAVE_INTERVAL:-10}" \
  -e RTX_SGLANG_CONCURRENCY="${RTX_SGLANG_CONCURRENCY:-24}" \
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
  -e RTX_TRAIN_ENTRY="${RTX_TRAIN_ENTRY:-train_async.py}" \
  -e RTX_LR="${RTX_LR:-5e-5}" \
  -e RTX_SAVE_HF="${RTX_SAVE_HF:-0}" \
  -e RTX_MAX_TOKENS_PER_GPU="${RTX_MAX_TOKENS_PER_GPU:-4096}" \
  -e RTX_SGLANG_MEM_FRACTION_STATIC="${RTX_SGLANG_MEM_FRACTION_STATIC:-0.55}" \
  -e RTX_V2_MODE="${RTX_V2_MODE:-strict}" \
  -e RTX_SEAL="$SEAL" \
  -e RTX_SEAL_AUTO_FIX="$SEAL_AUTO_FIX" \
  -e RTX_GROUP_RM="$GROUP_RM" \
  -e RTX_GROUP_SIZE="${RTX_GROUP_SIZE:-8}" \
  -e RTX_FAULT_WINDOWS="${RTX_FAULT_WINDOWS:-}" \
  -e RTX_LOG_RESPONSE="${RTX_LOG_RESPONSE:-0}" \
  -e RTX_ENTRY="${RTX_ENTRY:-day2_slime_train.sh}" \
  -e RTX_MAX_RETRIES="${RTX_MAX_RETRIES:-3}" \
  -e RTX_KILL_AFTER_ITER="${RTX_KILL_AFTER_ITER:-}" \
  --shm-size=64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$BASE":/workspace \
  -v "$LOCAL_SCRATCH":/rtx-scratch \
  -v "$SLIME_SOURCE_DIR":/root/slime \
  -v "$BASE/models":/root/models \
  -v "$BASE/models/datasets":/root/datasets \
  "$IMAGE" \
  bash -c "
    set -e
    git config --global --add safe.directory /root/slime
    cd /root/slime
    pip install -e . --no-deps -q
    bash /workspace/scripts/${RTX_ENTRY:-day2_slime_train.sh}
  "

echo "container $NAME started (fault=$FAULT, group_index [$START,$END])"
nohup docker logs -f "$NAME" > "$RUN_DIR/logs/train.log" 2>&1 &
echo "日志: $RUN_DIR/logs/train.log"
