#!/usr/bin/env bash
# 短冒烟实验入口: 用 4 步验证 Ray/CAS/checkpoint/retention/日志链路。
# 用法: bash scripts/smoke_test.sh [exp_id]
#   默认 exp_id = smoke-K8-s42-<时间戳>; 冒烟完成后人工确认再停容器:
#     docker logs -f rtx-day2-<exp_id>
#     docker rm -f rtx-day2-<exp_id>   (或等训练自然结束)
set -euo pipefail

BASE=/public/home/caiyiwen/rewardtxn
export RTX_PROFILE=smoke
export RTX_NUM_ROLLOUT=${RTX_NUM_ROLLOUT:-4}
EXP_ID=${RTX_EXP_ID:-${1:-smoke-K8-s42-$(date +%Y%m%d-%H%M%S)}}

echo "=== smoke test: exp=$EXP_ID num_rollout=$RTX_NUM_ROLLOUT profile=smoke ==="
bash "$BASE/scripts/day2_run.sh" none -1 -1 "$RTX_NUM_ROLLOUT" "$EXP_ID"
echo
echo "smoke launched: runs/$EXP_ID"
echo "日志: tail -f runs/$EXP_ID/logs/train.log"
echo "监控: docker logs -f rtx-day2-$EXP_ID"
