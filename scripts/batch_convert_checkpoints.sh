#!/usr/bin/env bash
# Phase C1: 批量转换 8 个 Megatron checkpoint 到 HF 格式（使用 host GPU 0..PARALLEL-1）
# 用法: bash scripts/batch_convert_checkpoints.sh [并行数，默认 4；映射 host GPU 0..N-1]
set -euo pipefail

BASE=$(cd "$(dirname "$0")/.." && pwd)
PARALLEL=${1:-4}

# 8 个正式 run 的 checkpoint 路径（iter_0000499）
declare -a CHECKPOINTS=(
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s29-20260903-194500/checkpoints/iter_0000499"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s29-20260903-194500/checkpoints/iter_0000499"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s42-20260903-194500/checkpoints/iter_0000499"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s42-20260903-194500/checkpoints/iter_0000499"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s73-20260904-093948/checkpoints/iter_0000499"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s73-20260904-093948/checkpoints/iter_0000499"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s101-20260904-093948/checkpoints/iter_0000499"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s101-20260904-093948/checkpoints/iter_0000499"
)

# 与训练一致的模型结构参数（qwen2.5-0.5B.sh）+ 单卡转换所需一致性参数
MODEL_ARGS="--swiglu --num-layers 24 --hidden-size 896 --ffn-hidden-size 4864 --num-attention-heads 14 \
--use-rotary-position-embeddings --disable-bias-linear --add-qkv-bias --normalization RMSNorm --norm-epsilon 1e-6 \
--rotary-base 1000000 --group-query-attention --num-query-groups 2 --vocab-size 151936"

echo "=== Phase C1: Batch Checkpoint Conversion ==="
echo "Date: $(date)"
echo "Checkpoints: ${#CHECKPOINTS[@]}"
echo "Host GPUs: 0-$(($PARALLEL-1)) (parallel=$PARALLEL)"
echo ""

convert_one() {
  local ckpt_path=$1
  local gpu_id=$2
  local output_path="${ckpt_path}_hf"

  if [ -d "$output_path" ] && [ -f "$output_path/model.safetensors.index.json" ]; then
    echo "[GPU$gpu_id] SKIP $ckpt_path (HF dir exists)"
    return 0
  fi

  echo "[GPU$gpu_id] START $ckpt_path → $output_path"
  rm -rf "$output_path"

  docker run --rm --gpus "device=$gpu_id" --shm-size 16g \
    -v "$BASE":/workspace -w /workspace -e CUDA_VISIBLE_DEVICES=0 \
    -v "$BASE/models":/root/models \
    -e PYTHONPATH=/root/Megatron-LM/:/workspace/scripts \
    slimerl/slime:v0.3.1 \
    bash -c "python3 -m torch.distributed.run --nproc_per_node=1 --master_port=$((29600 + gpu_id)) \
      scripts/convert_checkpoint_to_hf.py --checkpoint $ckpt_path --output $output_path \
      --actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 1 \
      --micro-batch-size 1 --global-batch-size 1 --rollout-batch-size 1 --n-samples-per-prompt 1 \
      --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --num-rollout 1 \
      --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct \
      $MODEL_ARGS --no-save-optim --no-load-optim --no-load-rng" \
    > "${ckpt_path}_convert.log" 2>&1

  if [ -f "$output_path/model.safetensors.index.json" ]; then
    echo "[GPU$gpu_id] DONE $output_path"
  else
    echo "[GPU$gpu_id] FAILED $ckpt_path (see ${ckpt_path}_convert.log)" >&2
    tail -5 "${ckpt_path}_convert.log" >&2
    return 1
  fi
}

export -f convert_one
export BASE MODEL_ARGS

i=0
for ckpt in "${CHECKPOINTS[@]}"; do
  gpu_id=$(( i % PARALLEL ))
  convert_one "$ckpt" "$gpu_id" &
  i=$(( i + 1 ))
  # 每 PARALLEL 个等待一波，控制并行度
  if [ $(( i % PARALLEL )) -eq 0 ]; then
    wait
  fi
done
wait

echo ""
echo "=== Phase C1 Complete ==="
FAILED=0
for ckpt in "${CHECKPOINTS[@]}"; do
  hf="${ckpt}_hf"
  if [ -f "$hf/model.safetensors.index.json" ]; then
    size=$(du -sh "$hf" | cut -f1)
    echo "  ✓ $hf ($size)"
  else
    echo "  ✗ $hf (MISSING)" >&2
    FAILED=1
  fi
done
exit $FAILED
