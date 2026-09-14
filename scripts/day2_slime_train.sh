#!/bin/bash
# Day 2 容器内训练脚本: 基于 day1_slime_train.sh + 故障注入 custom RM
# 新增: --custom-rm-path ${RTX_CUSTOM_RM:-} (注入模块)
#       num_rollout 由 RTX_NUM_ROLLOUT 覆盖 (默认 20; smoke 用 8)
# 注入窗口经环境变量 RTX_FAULT/RTX_FAULT_START/RTX_FAULT_END/RTX_RUN_DIR 传递
set -ex
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

SLIME_ROOT=${SLIME_ROOT:-/root/slime}
SCRIPT_DIR="${SLIME_ROOT}"
source "${SLIME_ROOT}/scripts/models/${RTX_MODEL_CONFIG:-qwen2.5-0.5B.sh}"

MODEL_DIR=${MODEL_DIR:-/root/models/Qwen2.5-0.5B-Instruct}
DATA_PATH=${DATA_PATH:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}
SAVE_DIR=${SAVE_DIR:-/workspace/runs/p1-slime-B0-K8-s42-20260824/checkpoints}
NUM_ROLLOUT=${RTX_NUM_ROLLOUT:-20}
CUSTOM_RM=${RTX_CUSTOM_RM:-}
SEED=${RTX_SEED:-42}
BASELINE_MODE=${RTX_BASELINE_MODE:-b0}
SCHEDULE_PATH=${RTX_SCHEDULE:-}
PAPER_MODE=${RTX_PAPER_MODE:-0}

case "${SEED}" in
   ''|*[!0-9]*) echo "RTX_SEED must be a non-negative integer (got ${SEED})" >&2; exit 2 ;;
esac
case "${BASELINE_MODE}" in
   b0)
      [ "${RTX_SEAL:-0}" = "0" ] && [ "${RTX_GROUP_RM:-0}" = "0" ] \
         && [ "${RTX_SEAL_AUTO_FIX:-0}" = "0" ] \
         || { echo "b0 requires RTX_SEAL=0 RTX_GROUP_RM=0 RTX_SEAL_AUTO_FIX=0" >&2; exit 2; }
      if [ "${PAPER_MODE}" = "1" ] && [ "${CUSTOM_RM}" != "day2_custom_rm.rm_function" ]; then
         echo "formal b0 requires RTX_CUSTOM_RM=day2_custom_rm.rm_function" >&2
         exit 2
      fi
      ;;
   b6)
      [ "${RTX_SEAL:-0}" = "1" ] && [ "${RTX_GROUP_RM:-0}" = "1" ] \
         && [ "${RTX_SEAL_AUTO_FIX:-0}" = "1" ] \
         || { echo "b6 requires RTX_SEAL=1 RTX_GROUP_RM=1 RTX_SEAL_AUTO_FIX=1" >&2; exit 2; }
      if [ "${PAPER_MODE}" = "1" ] && [ "${CUSTOM_RM}" != "phase2_seal_rm.rm_function" ]; then
         echo "formal b6 requires RTX_CUSTOM_RM=phase2_seal_rm.rm_function" >&2
         exit 2
      fi
      ;;
   group_rm)
      [ "${RTX_SEAL:-0}" = "0" ] && [ "${RTX_GROUP_RM:-0}" = "1" ] \
         && [ "${RTX_SEAL_AUTO_FIX:-0}" = "0" ] \
         || { echo "group_rm requires RTX_SEAL=0 RTX_GROUP_RM=1 RTX_SEAL_AUTO_FIX=0" >&2; exit 2; }
      [ "${CUSTOM_RM}" = "day2_custom_rm.rm_function" ] \
         || { echo "group_rm requires RTX_CUSTOM_RM=day2_custom_rm.rm_function" >&2; exit 2; }
      ;;
   b1|b2|b3|b4|b5)
      if [ "${PAPER_MODE}" = "1" ]; then
         echo "baseline ${BASELINE_MODE}: mechanism implementation missing; formal run refused" >&2
         exit 2
      fi
      ;;
   *) echo "RTX_BASELINE_MODE must be b0..b6 or group_rm (got ${BASELINE_MODE})" >&2; exit 2 ;;
esac
if [ -n "${SCHEDULE_PATH}" ] && [ ! -f "${SCHEDULE_PATH}" ]; then
   echo "RTX_SCHEDULE does not exist in container: ${SCHEDULE_PATH}" >&2
   exit 2
fi
if [ "${PAPER_MODE}" = "1" ] && [ -z "${SCHEDULE_PATH}" ]; then
   echo "formal run requires RTX_SCHEDULE" >&2
   exit 2
fi
echo "[train] baseline=${BASELINE_MODE} seed=${SEED} schedule=${SCHEDULE_PATH:-none}"

# Memory/I/O guardrails.  The previous slime default was 512 requests per
# engine; with three rollout engines that allowed 1536 in-flight requests and
# let fully-async retain a large Python/Ray backlog.
SGLANG_CONCURRENCY=${RTX_SGLANG_CONCURRENCY:-24}
RAY_OBJECT_STORE_MEMORY=${RTX_RAY_OBJECT_STORE_MEMORY:-17179869184}  # 16 GiB
RAY_TMP_DIR=${RTX_RAY_TMP_DIR:-/tmp/rtx-ray}
RAY_PLASMA_DIR=${RTX_RAY_PLASMA_DIR:-/dev/shm}
RAY_SPILL_DIR=${RTX_RAY_SPILL_DIR:-${RAY_TMP_DIR}/spill}
MAX_TOKENS_PER_GPU=${RTX_MAX_TOKENS_PER_GPU:-3072}
SGLANG_MEM_FRACTION_STATIC=${RTX_SGLANG_MEM_FRACTION_STATIC:-0.45}
NO_SAVE_OPTIM=${RTX_NO_SAVE_OPTIM:-0}
FULLY_ASYNC=${RTX_FULLY_ASYNC:-1}
# Keep the latest two completed full checkpoints; the previous one is the
# recovery fallback if a save is interrupted or the newest directory is bad.
CKPT_KEEP=${RTX_CKPT_KEEP:-2}
case "${CKPT_KEEP}" in
   ''|*[!0-9]*) echo "RTX_CKPT_KEEP must be 0 or a positive integer (got ${CKPT_KEEP})" >&2; exit 2 ;;
esac
if [[ "${CKPT_KEEP}" =~ ^0+$ ]]; then CKPT_KEEP=0; fi
CKPT_RETENTION_INTERVAL=${RTX_CKPT_RETENTION_INTERVAL:-15}
CKPT_RETENTION_MIN_AGE=${RTX_CKPT_RETENTION_MIN_AGE:-30}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}_torch_dist"
   --save "${SAVE_DIR}"
   --save-interval ${RTX_SAVE_INTERVAL:-10}
)

# Save HuggingFace checkpoints alongside Megatron checkpoints for direct
# evaluation without conversion.  HF checkpoints are saved every N iterations
# (same as Megatron save-interval) to ${SAVE_DIR}/iter_{rollout_id:07d}_hf/.
# DEFAULT: Disabled (RTX_SAVE_HF=0) to save disk space. Each HF checkpoint is ~3GB.
if [ "${RTX_SAVE_HF:-0}" = "1" ]; then
   CKPT_ARGS+=(--save-hf "${SAVE_DIR}/iter_{rollout_id:07d}_hf")
fi

if [ "${NO_SAVE_OPTIM}" = "1" ]; then
   # Appropriate for non-resumable long baselines only.  Phase 3B recovery
   # must leave this disabled because optimizer state is needed for --load.
   if [[ " ${RTX_EXTRA_MODEL_ARGS:-} " == *" --load "* ]]; then
      echo "[train] RTX_NO_SAVE_OPTIM=1 cannot be combined with --load resume (optimizer state not saved)" >&2
      exit 2
   fi
   CKPT_ARGS+=(--no-save-optim)
fi

ROLLOUT_ARGS=(
   --prompt-data "${DATA_PATH}"
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   --rm-type deepscaler

   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size 4
   --n-samples-per-prompt 8
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --sglang-server-concurrency "${SGLANG_CONCURRENCY}"

   --global-batch-size 32
   --balance-data
)

if [ -n "$CUSTOM_RM" ]; then
   ROLLOUT_ARGS+=(--custom-rm-path "$CUSTOM_RM")
fi

if [ "${RTX_GROUP_RM:-0}" = "1" ]; then
   ROLLOUT_ARGS+=(--group-rm)
fi
if [ "${FULLY_ASYNC}" = "1" ]; then
   ROLLOUT_ARGS+=(--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async)
fi

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${RTX_LR:-5e-5}"
   --lr-decay-style constant
   --lr-warmup-iters 10
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --seed "${SEED}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --clip-grad 0.5
)

NUM_GPUS=${NUM_GPUS:-4}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
mkdir -p "${RAY_TMP_DIR}" "${RAY_SPILL_DIR}"
CKPT_RETENTION_PID=""
RESOURCE_WATCH_PID=""
cleanup_runtime() {
   if [ -n "${CKPT_RETENTION_PID}" ]; then
      kill "${CKPT_RETENTION_PID}" >/dev/null 2>&1 || true
      wait "${CKPT_RETENTION_PID}" >/dev/null 2>&1 || true
   fi
   if [ -n "${RESOURCE_WATCH_PID}" ]; then
      kill "${RESOURCE_WATCH_PID}" >/dev/null 2>&1 || true
      wait "${RESOURCE_WATCH_PID}" >/dev/null 2>&1 || true
   fi
   ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_runtime EXIT
if [ "${CKPT_KEEP}" != "0" ]; then
   RETENTION_LOG_DIR=${RTX_RUN_DIR:-$(dirname "${SAVE_DIR}")}
   mkdir -p "${RETENTION_LOG_DIR}"
   python3 /workspace/scripts/checkpoint_retention.py watch "${SAVE_DIR}" \
      --keep "${CKPT_KEEP}" \
      --interval-seconds "${CKPT_RETENTION_INTERVAL}" \
      --min-age-seconds "${CKPT_RETENTION_MIN_AGE}" \
      >"${RETENTION_LOG_DIR}/checkpoint_retention.log" 2>&1 &
   CKPT_RETENTION_PID=$!
fi
if [ "${RTX_SKIP_GATE:-0}" != "1" ]; then
   WATCH_DIR=${RTX_RUN_DIR:-$(dirname "${SAVE_DIR}")}
   mkdir -p "${WATCH_DIR}/logs"
   python3 /workspace/scripts/resource_gate.py watch --dir "${WATCH_DIR}/logs" \
      --interval "${RTX_GATE_INTERVAL:-60}" --gpus "" &
   RESOURCE_WATCH_PID=$!
fi
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats \
   --object-store-memory "${RAY_OBJECT_STORE_MEMORY}" \
   --plasma-directory "${RAY_PLASMA_DIR}" \
   --object-spilling-directory "${RAY_SPILL_DIR}" \
   --temp-dir "${RAY_TMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/workspace/scripts\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"RTX_CAS_INDEX_DIR\": \"${RTX_CAS_INDEX_DIR:-${RTX_RUN_DIR:-/workspace/runs}/cas}\",
    \"RTX_SEED\": \"${SEED}\",
    \"RTX_BASELINE_MODE\": \"${BASELINE_MODE}\",
    \"RTX_SCHEDULE\": \"${SCHEDULE_PATH}\",
    \"RTX_PAPER_MODE\": \"${PAPER_MODE}\"
  }
}"

ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${RTX_TRAIN_ENTRY:-train_async.py}" \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_GPUS}" \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   ${MODEL_ARGS[@]} \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   ${RTX_EXTRA_MODEL_ARGS:-}
