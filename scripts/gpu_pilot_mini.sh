#!/usr/bin/env bash
# GPU Pilot Mini: 快速验证版本，只测试 1-2 个核心配置
# 用法: bash scripts/gpu_pilot_mini.sh

set -euo pipefail

BASE=/public/home/caiyiwen/rewardtxn
PILOT_LOG="$BASE/runs/pilot_mini_$(date +%Y%m%d-%H%M%S).json"

echo "=== GPU Pilot Mini 开始 ==="
echo "目标: 快速验证环境并测量 1-2 个核心配置"
echo "结果: $PILOT_LOG"
echo

# 初始化结果文件
cat > "$PILOT_LOG" <<EOF
{
  "pilot_date": "$(date -Iseconds)",
  "pilot_commit": "$(cd "$BASE" && git rev-parse HEAD)",
  "pilot_tag": "$(cd "$BASE" && git describe --tags --exact-match 2>/dev/null || echo 'none')",
  "runs": []
}
EOF

run_pilot() {
  local test_name=$1
  local model_size=$2
  local num_gpus=$3
  local gpus=$4
  local num_steps=${5:-10}
  local seed=${6:-42}
  
  local exp_id="pilot-mini-${test_name}-${model_size}-${num_gpus}gpu-s${seed}-$(date +%H%M%S)"
  
  echo "--- 运行: $test_name (${model_size}, ${num_gpus} GPUs) ---"
  echo "  EXP_ID: $exp_id"
  echo "  GPUs: $gpus"
  echo "  Steps: ~$num_steps"
  
  local start_time=$(date +%s)
  
  # 根据模型大小设置配置
  local model_dir="/root/models/Qwen2.5-${model_size}-Instruct"
  local model_config="qwen2.5-${model_size}.sh"
  local extra_args=""
  
  if [ "$model_size" = "1.5B" ]; then
    extra_args="--rotary-base 1000000"
  fi
  
  echo "  开始时间: $(date '+%H:%M:%S')"
  
  # 运行短训练（使用 phase2_run.sh，none 表示无故障注入）
  RTX_GPUS="device=$gpus" \
  RTX_SEED=$seed \
  RTX_MODEL_DIR=$model_dir \
  RTX_MODEL_CONFIG=$model_config \
  RTX_EXTRA_MODEL_ARGS="$extra_args" \
  RTX_NUM_ROLLOUT=$num_steps \
  RTX_BASELINE_MODE=b0 \
  RTX_EXP_ID=$exp_id \
    bash "$BASE/scripts/phase2_run.sh" none -1 -1 $num_steps "$exp_id" 2>&1 | tee "$BASE/runs/${exp_id}.log"
  
  local exit_code=${PIPESTATUS[0]}
  local end_time=$(date +%s)
  local wall_time=$((end_time - start_time))
  
  if [ $exit_code -eq 0 ]; then
    echo "  ✓ 完成，用时: ${wall_time}s (~$((wall_time / 60))min)"
    local status="success"
  else
    echo "  ✗ 失败，用时: ${wall_time}s"
    local status="failed"
  fi
  
  # 添加结果到 JSON
  python3 - "$PILOT_LOG" "$test_name" "$exp_id" "$model_size" "$num_gpus" "$gpus" "$num_steps" "$seed" "$wall_time" "$start_time" "$end_time" "$status" <<'EOF'
import json, sys
log_path, test_name, exp_id, model_size, num_gpus, gpus, num_steps, seed, wall_time, start_time, end_time, status = sys.argv[1:12]
with open(log_path) as f:
    data = json.load(f)
data['runs'].append({
    "test_name": test_name,
    "exp_id": exp_id,
    "model_size": model_size,
    "num_gpus": int(num_gpus),
    "gpus": gpus,
    "num_steps": int(num_steps),
    "seed": int(seed),
    "wall_time_seconds": int(wall_time),
    "start_time": int(start_time),
    "end_time": int(end_time),
    "status": status
})
with open(log_path, 'w') as f:
    json.dump(data, f, indent=2)
EOF
  
  echo
  
  return $exit_code
}

# 只测试 2 个核心配置
echo "=== 测试 1: 0.5B × 4 GPUs (baseline) ==="
run_pilot "e3-baseline" "0.5B" 4 "0,1,2,3" 10 42 || {
  echo "⚠️  第一个测试失败，停止 pilot"
  exit 1
}

echo "=== 测试 2: 1.5B × 4 GPUs (论文主配置) ==="
run_pilot "e3-baseline" "1.5B" 4 "4,5,6,7" 10 43 || {
  echo "⚠️  第二个测试失败"
  exit 1
}

echo "=== GPU Pilot Mini 完成 ==="
echo "结果文件: $PILOT_LOG"
echo
echo "解析结果:"
python3 - "$PILOT_LOG" <<'EOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
print(f"\n总计运行: {len(data['runs'])} 个测试")
total_time = sum(r['wall_time_seconds'] for r in data['runs'])
print(f"总用时: {total_time} 秒 (~{total_time / 60:.1f} 分钟)")
print(f"成功: {sum(1 for r in data['runs'] if r['status'] == 'success')} / {len(data['runs'])}")
print("\n详细结果:")
for run in data['runs']:
    status_icon = "✓" if run['status'] == 'success' else "✗"
    print(f"  {status_icon} {run['test_name']:20s} {run['model_size']:5s} {run['num_gpus']}GPU  {run['wall_time_seconds']:4d}s (~{run['wall_time_seconds']/60:.1f}min)")
EOF

echo
echo "如果测试成功，可运行完整版: bash scripts/gpu_pilot.sh"
