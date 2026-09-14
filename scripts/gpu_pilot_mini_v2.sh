#!/usr/bin/env bash
# GPU Pilot Mini v2: 使用 day2_run.sh（类似 smoke_test.sh）
set -euo pipefail

BASE=/public/home/caiyiwen/rewardtxn
PILOT_LOG="$BASE/runs/pilot_mini_v2_$(date +%Y%m%d-%H%M%S).json"

echo "=== GPU Pilot Mini v2 开始 ==="
echo "目标: 快速验证环境（使用 day2_run.sh）"
echo "结果: $PILOT_LOG"
echo

cat > "$PILOT_LOG" <<EOF
{
  "pilot_date": "$(date -Iseconds)",
  "pilot_commit": "$(cd "$BASE" && git rev-parse HEAD)",
  "runs": []
}
EOF

run_pilot() {
  local test_name=$1
  local gpus=$2
  local num_steps=${3:-10}
  local seed=${4:-42}
  
  local exp_id="pilot-v2-${test_name}-s${seed}-$(date +%H%M%S)"
  
  echo "--- 运行: $test_name ---"
  echo "  EXP_ID: $exp_id"
  echo "  GPUs: $gpus"
  echo "  Steps: $num_steps"
  echo "  开始时间: $(date '+%H:%M:%S')"
  
  local start_time=$(date +%s)
  
  # 使用 day2_run.sh（类似 smoke_test.sh 的方式）
  # 注意：不设置 SEAL/GROUP_RM/SEAL_AUTO_FIX，让脚本使用 baseline 默认值
  RTX_GPUS="$gpus" \
  RTX_NUM_ROLLOUT=$num_steps \
  RTX_SEED=$seed \
  RTX_EXP_ID=$exp_id \
    bash "$BASE/scripts/day2_run.sh" none -1 -1 $num_steps "$exp_id" 2>&1 | tee "$BASE/runs/${exp_id}.log"
  
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
  
  python3 - "$PILOT_LOG" "$test_name" "$exp_id" "$gpus" "$num_steps" "$seed" "$wall_time" "$start_time" "$end_time" "$status" <<'EOF'
import json, sys
log_path, test_name, exp_id, gpus, num_steps, seed, wall_time, start_time, end_time, status = sys.argv[1:10]
with open(log_path) as f:
    data = json.load(f)
data['runs'].append({
    "test_name": test_name,
    "exp_id": exp_id,
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

# 测试 1: 0.5B × 4 GPUs
echo "=== 测试 1: 0.5B baseline (默认配置) ==="
run_pilot "0.5B-baseline" "device=0,1,2,3" 10 42 || {
  echo "⚠️  第一个测试失败，停止 pilot"
  exit 1
}

# 测试 2: 使用不同的 4 个 GPU
echo "=== 测试 2: 0.5B baseline (GPU 4-7) ==="
run_pilot "0.5B-baseline-gpu4567" "device=4,5,6,7" 10 43 || {
  echo "⚠️  第二个测试失败"
  exit 1
}

echo "=== GPU Pilot Mini v2 完成 ==="
echo "结果文件: $PILOT_LOG"
echo
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
    print(f"  {status_icon} {run['test_name']:25s} {run['wall_time_seconds']:4d}s (~{run['wall_time_seconds']/60:.1f}min)")
EOF

echo
echo "如果测试成功，说明环境正常，可以规划完整 pilot"
