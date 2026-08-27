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

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_DIR}"
   --ref-load "${MODEL_DIR}_torch_dist"
   --save "${SAVE_DIR}"
   --save-interval ${RTX_SAVE_INTERVAL:-10}
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async

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

   --global-batch-size 32
   --balance-data
)

if [ -n "$CUSTOM_RM" ]; then
   ROLLOUT_ARGS+=(--custom-rm-path "$CUSTOM_RM")
fi

if [ "${RTX_GROUP_RM:-0}" = "1" ]; then
   ROLLOUT_ARGS+=(--group-rm)
fi

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
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
   --sglang-mem-fraction-static 0.55
)

MISC_ARGS=(
   --seed 42
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

NUM_GPUS=${NUM_GPUS:-4}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/workspace/scripts\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

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
   "${MISC_ARGS[@]}" \
   ${RTX_EXTRA_MODEL_ARGS:-}
