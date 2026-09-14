#!/usr/bin/env bash
# Phase C2: 批量评价 8 个 HF checkpoint 在 GSM8K eval500（使用 host GPU 0..PARALLEL-1）
# 用法: bash scripts/batch_eval_checkpoints.sh [并行数，默认 4]
set -euo pipefail

BASE=$(cd "$(dirname "$0")/.." && pwd)
PARALLEL=${1:-4}
GPU_BASE=${2:-0}  # host GPU 起始编号（0=GPU0..3, 1=GPU1..4）
EVAL_SPLIT="prereg/eval_splits/gsm8k_eval500_seed42.json"  # 相对路径，容器内 /workspace/ 挂载后可见
VERIFIER="v1"

# 8 个 HF checkpoint 路径（对应 C1 输出）
declare -a HF_CHECKPOINTS=(
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s29-20260903-194500/checkpoints/iter_0000499_hf"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s29-20260903-194500/checkpoints/iter_0000499_hf"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s42-20260903-194500/checkpoints/iter_0000499_hf"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s42-20260903-194500/checkpoints/iter_0000499_hf"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s73-20260904-093948/checkpoints/iter_0000499_hf"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s73-20260904-093948/checkpoints/iter_0000499_hf"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s101-20260904-093948/checkpoints/iter_0000499_hf"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s101-20260904-093948/checkpoints/iter_0000499_hf"
)

# 对应输出 JSON 路径（放在各 run 根目录）
declare -a OUTPUT_JSONS=(
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s29-20260903-194500/eval_result.json"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s29-20260903-194500/eval_result.json"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s42-20260903-194500/eval_result.json"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s42-20260903-194500/eval_result.json"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s73-20260904-093948/eval_result.json"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s73-20260904-093948/eval_result.json"
  "runs/formal-E7-clean-oracle-0.5B-4gpu-s101-20260904-093948/eval_result.json"
  "runs/formal-E7-clean-rewardtxn-0.5B-4gpu-s101-20260904-093948/eval_result.json"
)

echo "=== Phase C2: Batch Checkpoint Evaluation ==="
echo "Date: $(date)"
echo "Checkpoints: ${#HF_CHECKPOINTS[@]}"
echo "Eval split: $EVAL_SPLIT (500 samples)"
echo "Verifier: $VERIFIER"
echo "Host GPUs: ${GPU_BASE}-$((${GPU_BASE} + PARALLEL - 1)) (parallel=$PARALLEL)"
echo ""

[ -f "$BASE/$EVAL_SPLIT" ] || { echo "ERROR: eval split not found: $BASE/$EVAL_SPLIT" >&2; exit 1; }

eval_one() {
  local ckpt_path=$1
  local output_json=$2
  local gpu_id=$3
  local host_gpu=$(( gpu_id + GPU_BASE ))
  
  if [ -f "$output_json" ]; then
    echo "[GPU$gpu_id] SKIP $ckpt_path (eval result exists)"
    return 0
  fi
  
  if [ ! -d "$ckpt_path" ]; then
    echo "[GPU$gpu_id] ERROR: HF checkpoint not found: $ckpt_path" >&2
    return 1
  fi
  
  echo "[GPU${host_gpu}] START eval $ckpt_path → $output_json"
  
  # 容器内评价（HF transformers greedy generation，1 GPU，~45 min）
  timeout 3600 docker run --rm --gpus "device=${host_gpu}" --shm-size 32g \
    -v "$BASE":/workspace -w /workspace -e CUDA_VISIBLE_DEVICES=0 \
    slimerl/slime:v0.3.1 \
    python3 scripts/e7_eval_checkpoint_incontainer.py \
      --checkpoint "$ckpt_path" --eval-split "$EVAL_SPLIT" \
      --verifier-version "$VERIFIER" --output "$output_json" \
      --gpus 1 --batch-size 8 \
    > "${output_json%.json}.log" 2>&1
  
  if [ -f "$output_json" ]; then
    acc=$(python3 -c "import json; d=json.load(open('$output_json')); print(f\"{d['final_eval_accuracy']:.4f}\")")
    echo "[GPU${host_gpu}] DONE $output_json (accuracy=$acc)"
  else
    echo "[GPU${host_gpu}] FAILED $ckpt_path (see ${output_json%.json}.log)" >&2
    tail -8 "${output_json%.json}.log" >&2
    return 1
  fi
}

export -f eval_one
export BASE EVAL_SPLIT VERIFIER

i=0
for idx in "${!HF_CHECKPOINTS[@]}"; do
  gpu_id=$(( i % PARALLEL ))
  eval_one "${HF_CHECKPOINTS[$idx]}" "${OUTPUT_JSONS[$idx]}" "$gpu_id" &
  i=$(( i + 1 ))
  # 每 PARALLEL 个等待一波
  if [ $(( i % PARALLEL )) -eq 0 ]; then
    wait
  fi
done
wait

echo ""
echo "=== Phase C2 Complete ==="
FAILED=0
for json in "${OUTPUT_JSONS[@]}"; do
  if [ -f "$json" ]; then
    acc=$(python3 -c "import json; d=json.load(open('$json')); print(f\"{d['final_eval_accuracy']:.4f}\")" 2>/dev/null || echo "?")
    echo "  ✓ $json (acc=$acc)"
  else
    echo "  ✗ $json (MISSING)" >&2
    FAILED=1
  fi
done
exit $FAILED
