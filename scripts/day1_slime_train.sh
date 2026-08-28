#!/bin/bash
# Day 1 容器内训练脚本（基于 examples/fully_async/run-qwen2.5-0.5B-fully_async.sh 修改）
#
# 修改点（与示例/文档对齐讨论结论）:
#   * K=8:            --n-samples-per-prompt 8, --rollout-batch-size 4  (每步仍 32 样本, U=4 组/步)
#   * 20 步:          --num-rollout 20
#   * 梯度可测:       --lr 1e-4 (原 1e-6), --kl-loss-coef 0.01 (原 0.00)
#   * 可恢复历史:     --save-interval 10 (原 9999, 20 步内 2 个 checkpoint)
#   * 可复现:         --seed 42
#   * 输出目录:       $SAVE_DIR (宿主 runs/{exp_id}/checkpoints)
set -ex
export PYTHONUNBUFFERED=1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# 锁定源码内的模型配置 (容器内 /root/slime 为挂载的 third_party/slime @ a6272da0)
SLIME_ROOT=${SLIME_ROOT:-/root/slime}
SCRIPT_DIR="${SLIME_ROOT}"
source "${SLIME_ROOT}/scripts/models/qwen2.5-0.5B.sh"

MODEL_DIR=${MODEL_DIR:-/root/models/Qwen2.5-0.5B-Instruct}
DATA_PATH=${DATA_PATH:-/root/datasets/dapo-math-17k/dapo-math-17k.jsonl}
SAVE_DIR=${SAVE_DIR:-/workspace/runs/p1-slime-B0-K8-s42-20260824/checkpoints}
NUM_ROLLOUT=${RTX_NUM_ROLLOUT:-20}

# Keep legacy Day 1 runs under the same memory guardrails as Phase 2/3.
SGLANG_CONCURRENCY=${RTX_SGLANG_CONCURRENCY:-64}
RAY_OBJECT_STORE_MEMORY=${RTX_RAY_OBJECT_STORE_MEMORY:-17179869184}  # 16 GiB
RAY_TMP_DIR=${RTX_RAY_TMP_DIR:-/tmp/rtx-ray}
RAY_PLASMA_DIR=${RTX_RAY_PLASMA_DIR:-/dev/shm}
RAY_SPILL_DIR=${RTX_RAY_SPILL_DIR:-${RAY_TMP_DIR}/spill}
MAX_TOKENS_PER_GPU=${RTX_MAX_TOKENS_PER_GPU:-4096}
SGLANG_MEM_FRACTION_STATIC=${RTX_SGLANG_MEM_FRACTION_STATIC:-0.55}
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
if [ "${NO_SAVE_OPTIM}" = "1" ]; then
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
   --lr 1e-4
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --seed 42
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# launch the master node of ray in container
NUM_GPUS=${NUM_GPUS:-4}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
mkdir -p "${RAY_TMP_DIR}" "${RAY_SPILL_DIR}"
CKPT_RETENTION_PID=""
cleanup_runtime() {
   if [ -n "${CKPT_RETENTION_PID}" ]; then
      kill "${CKPT_RETENTION_PID}" >/dev/null 2>&1 || true
      wait "${CKPT_RETENTION_PID}" >/dev/null 2>&1 || true
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
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats \
   --object-store-memory "${RAY_OBJECT_STORE_MEMORY}" \
   --plasma-directory "${RAY_PLASMA_DIR}" \
   --object-spilling-directory "${RAY_SPILL_DIR}" \
   --temp-dir "${RAY_TMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

# fully-async splits actor / rollout onto disjoint GPUs (no colocation).
ACTOR_GPUS=${ACTOR_GPUS:-1}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-$((NUM_GPUS - ACTOR_GPUS))}

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
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
   "${MISC_ARGS[@]}"
