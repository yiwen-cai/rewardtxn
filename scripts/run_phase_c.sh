#!/usr/bin/env bash
# Phase C 主控：串联 C1（转换）→ C2（评价）→ C3（TOST）
# 用法: bash scripts/run_phase_c.sh [并行数，默认 4] [跳过步骤: skip-c1|skip-c2|skip-c3]
set -euo pipefail

BASE=$(cd "$(dirname "$0")/.." && pwd)
PARALLEL=${1:-4}
SKIP_FLAGS=${2:-}

echo "============================================"
echo "Phase C: Checkpoint 转换 + 评价 + TOST 统计"
echo "============================================"
echo "Date: $(date)"
echo "Parallel GPUs: $PARALLEL"
echo "Skip flags: ${SKIP_FLAGS:-none}"
echo ""

# C1: 转换 Megatron → HF
if [[ "$SKIP_FLAGS" != *skip-c1* ]]; then
  echo "=== C1: Converting Megatron checkpoints to HF ==="
  bash "$BASE/scripts/batch_convert_checkpoints.sh" "$PARALLEL"
  echo ""
else
  echo "=== C1: SKIPPED (skip-c1) ==="
fi

# C2: 评价 HF checkpoint
if [[ "$SKIP_FLAGS" != *skip-c2* ]]; then
  echo "=== C2: Evaluating HF checkpoints on GSM8K eval500 ==="
  bash "$BASE/scripts/batch_eval_checkpoints.sh" "$PARALLEL"
  echo ""
else
  echo "=== C2: SKIPPED (skip-c2) ==="
fi

# C3: TOST 统计（需在容器内运行，有 scipy）
if [[ "$SKIP_FLAGS" != *skip-c3* ]]; then
  echo "=== C3: TOST equivalence analysis ==="
  docker run --rm \
    -v "$BASE:/workspace" \
    -w /workspace \
    slimerl/slime:v0.3.1 \
    python3 scripts/e7_aggregate_tost.py
  echo ""
else
  echo "=== C3: SKIPPED (skip-c3) ==="
fi

echo "============================================"
echo "Phase C Complete"
echo "============================================"
echo "Results:"
echo "  - HF checkpoints: runs/formal-E7-*/checkpoints/iter_0000499_hf/"
echo "  - Eval JSONs:     runs/formal-E7-*/eval_result.json"
echo "  - TOST report:    runs/e7_tost_report.{json,md}"
echo ""
echo "Next: Review TOST report, update HANDOFF.md Phase C ✓"
