#!/bin/bash
# Phase 3B 自动恢复入口 (容器内): 训练崩溃 -> 检测 -> Reconciler 恢复计划 -> 自动重启续跑
# 机制 (3B 探针验证): slime --load <ckpt> + --override-opt-param-scheduler (megatron 原生参数)
#   -> scheduler num_steps 断言跳过, 从 latest checkpoint 的下一步续跑到目标步数
# 幂等: 恢复计数上限; 恢复后关闭故障注入; 已提交步由 checkpoint iteration 天然保证不重训
set -euo pipefail
export PYTHONUNBUFFERED=1

TARGET_ROLLOUT=${RTX_NUM_ROLLOUT:-20}
SAVE_DIR=${SAVE_DIR:-/workspace/runs/p3b/checkpoints}
MAX_RETRIES=${RTX_MAX_RETRIES:-3}
RESTART_FILE=/workspace/runs/rtx_restart_count.txt
HISTORY=/workspace/runs/rtx_recovery_history.jsonl

attempt=0

# ---- 崩溃注入 (实验用, 仅 attempt 1): checkpoint 落盘后 kill 训练进程 ----
# 注: crm_crash 异常注入被 slime fully_async 容错捕获 (R1 语义=组静默丢弃),
# 自动恢复针对真实进程崩溃 (kill -9) 设计, 如 Day 4 L2
KILL_AFTER_ITER=${RTX_KILL_AFTER_ITER:-}
if [ -n "$KILL_AFTER_ITER" ]; then
  echo "[3B] crash injection armed: kill -9 training after iter >= $KILL_AFTER_ITER"
  (
    while true; do
      if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
        IT=$(cat "$SAVE_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
        if [ "$IT" -ge "$KILL_AFTER_ITER" ]; then
          echo "[3B] INJECT: kill -9 training (iter=$IT)"
          pkill -9 -f train_async.py >/dev/null 2>&1 || true
          pkill -9 -f raysubmit >/dev/null 2>&1 || true
          pkill -9 -f sglang >/dev/null 2>&1 || true
          ray stop --force >/dev/null 2>&1 || true
          exit 0
        fi
      fi
      sleep 5
    done
  ) &
fi
while true; do
  attempt=$((attempt + 1))
  echo "[3B] ===== attempt $attempt (num_rollout=$TARGET_ROLLOUT) ====="

  if bash /workspace/scripts/day2_slime_train.sh; then
    echo "[3B] training SUCCEEDED after attempt $attempt"
    exit 0
  fi
  rc=$?
  echo "[3B] training FAILED rc=$rc (attempt $attempt)"

  if [ "$attempt" -gt "$MAX_RETRIES" ]; then
    echo "[3B] max retries ($MAX_RETRIES) reached -> ALERT, giving up" >&2
    exit 1
  fi

  # ---- 恢复计划: 识别已提交步 (StepToken/Manifest 审计) ----
  if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LATEST=$(cat "$SAVE_DIR/latest_checkpointed_iteration.txt")
  else
    LATEST="none"
  fi
  echo "[3B] recovery plan: latest committed step = $LATEST"
  python3 - "$LATEST" "$rc" "$attempt" <<'EOF'
import json, sys, time, os
latest, rc, attempt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
rec = {
    "ts": time.time(), "attempt": attempt, "exit_rc": rc,
    "committed_step": None if latest == "none" else int(latest),
    "recovery_action": ("resume" if latest != "none" else "cold_restart"),
    "note": "resume: --load + --override-opt-param-scheduler (megatron native, 3B 探针验证); cold_restart: 无 checkpoint 从头",
}
os.makedirs(os.path.dirname("/workspace/runs/rtx_recovery_history.jsonl"), exist_ok=True)
with open("/workspace/runs/rtx_recovery_history.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print("[3B]", json.dumps(rec, ensure_ascii=False))
EOF

  # ---- 自动重启参数 ----
  # 1) 关闭故障注入 (故障已发生, 恢复后不再注入)
  export RTX_FAULT=none
  export RTX_FAULT_START=-1
  export RTX_FAULT_END=-1
  export RTX_FAULT_WINDOWS=""
  export RTX_KILL_AFTER_ITER=""   # 关闭崩溃注入 (仅 attempt 1)
  # 2) 若存在已提交 checkpoint: --load + override scheduler (续跑)
  if [ "$LATEST" != "none" ] && [ -d "$SAVE_DIR" ]; then
    export RTX_EXTRA_MODEL_ARGS="--load $SAVE_DIR --override-opt-param-scheduler"
    echo "[3B] resume from step $LATEST: RTX_EXTRA_MODEL_ARGS='$RTX_EXTRA_MODEL_ARGS'"
  else
    export RTX_EXTRA_MODEL_ARGS=""
    echo "[3B] no committed checkpoint -> cold restart (rollout data replay via protocol records)"
  fi

  # ---- 清理残留 Ray/训练进程 (防重启冲突) ----
  echo "[3B] cleaning up stale ray/training processes"
  ray stop --force >/dev/null 2>&1 || true
  pkill -f "train_async.py" >/dev/null 2>&1 || true
  pkill -f "sglang" >/dev/null 2>&1 || true
  sleep 10
done
