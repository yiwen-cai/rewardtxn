#!/usr/bin/env bash
# Day 4 注入辅助: 等待指定 step 出现后 kill trainer actor (MegatronTrainRayActor)
# 用法: bash day4_inject.sh <容器名> <目标step> <日志路径>
set -uo pipefail
C=$1; TARGET=$2; LOG=$3
echo "[inject] waiting for step $TARGET in $LOG"
for i in $(seq 1 600); do
  if grep -qE "step ${TARGET}:" "$LOG" 2>/dev/null; then
    sleep 3  # 确保进入该 step 的训练
    PID=$(docker exec "$C" bash -c "ps aux | grep 'ray::MegatronTrainRayActor' | grep -v grep | awk '{print \$2}'" 2>/dev/null | head -1)
    if [ -n "$PID" ]; then
      echo "[inject] step $TARGET reached, killing trainer actor PID=$PID at $(date +%H:%M:%S)"
      docker exec "$C" kill -9 "$PID"
      echo "[inject] done"
      exit 0
    fi
    echo "[inject] step found but actor PID not located"; exit 1
  fi
  sleep 5
done
echo "[inject] TIMEOUT: step $TARGET never appeared"; exit 1
