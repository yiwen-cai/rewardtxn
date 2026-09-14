#!/usr/bin/env bash
# E7 Pre-flight Check: Verify all prerequisites before formal training
set -euo pipefail

BASE=${RTX_BASE:-/public/home/caiyiwen/rewardtxn}
REQUIRED_SPACE_GB=240
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== E7 Formal Training Pre-flight Check ==="
echo ""

PASS=0
WARN=0
FAIL=0

check_pass() {
  echo -e "${GREEN}✓${NC} $1"
  PASS=$((PASS+1))
}

check_warn() {
  echo -e "${YELLOW}⚠${NC} $1"
  WARN=$((WARN+1))
}

check_fail() {
  echo -e "${RED}✗${NC} $1"
  FAIL=$((FAIL+1))
}

# 1. Storage space
echo "1. Storage Space"
AVAILABLE_GB=$(df -BG /public | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_GB" -ge "$REQUIRED_SPACE_GB" ]; then
  check_pass "Sufficient space: ${AVAILABLE_GB} GB available (need ${REQUIRED_SPACE_GB} GB)"
else
  check_fail "Insufficient space: ${AVAILABLE_GB} GB available (need ${REQUIRED_SPACE_GB} GB)"
fi
echo ""

# 2. Frozen margin
echo "2. Frozen Prereg Parameters"
if [ -f "$BASE/prereg/tost_margin.json" ]; then
  FROZEN_MARGIN=$(python3 -c "import json; p=json.load(open('$BASE/prereg/tost_margin.json')); print(p.get('frozen_margin_pp') or '')")
  FROZEN_COMMIT=$(python3 -c "import json; p=json.load(open('$BASE/prereg/tost_margin.json')); print(p.get('frozen_commit') or '')")
  if [ -n "$FROZEN_MARGIN" ]; then
    check_pass "Margin frozen: ${FROZEN_MARGIN} pp"
  else
    check_fail "Margin not frozen (frozen_margin_pp is null)"
  fi
  if [ -n "$FROZEN_COMMIT" ]; then
    check_pass "Commit frozen: ${FROZEN_COMMIT:0:8}"
  else
    check_warn "Commit not recorded (frozen_commit is null)"
  fi
else
  check_fail "prereg/tost_margin.json not found"
fi
echo ""

# 3. Model and data files
echo "3. Required Files"
if [ -d "$BASE/models/Qwen2.5-1.5B-Instruct" ]; then
  check_pass "Base model: models/Qwen2.5-1.5B-Instruct"
else
  check_fail "Base model not found: models/Qwen2.5-1.5B-Instruct"
fi

if [ -d "$BASE/models/Qwen2.5-1.5B-Instruct_torch_dist" ]; then
  check_pass "Ref model: models/Qwen2.5-1.5B-Instruct_torch_dist"
else
  check_fail "Ref model not found: models/Qwen2.5-1.5B-Instruct_torch_dist"
fi

if [ -f "$BASE/models/datasets/gsm8k/dapo-gsm8k-train.jsonl" ]; then
  check_pass "Training data: models/datasets/gsm8k/dapo-gsm8k-train.jsonl"
else
  check_fail "Training data not found: models/datasets/gsm8k/dapo-gsm8k-train.jsonl"
fi

if [ -f "$BASE/prereg/eval_splits/gsm8k_eval500_seed42.json" ]; then
  check_pass "Eval split: prereg/eval_splits/gsm8k_eval500_seed42.json"
else
  check_fail "Eval split not found"
fi
echo ""

# 4. Scripts
echo "4. Training Scripts"
for script in day2_slime_train.sh e7_formal_launch.sh phase2_run.sh; do
  if [ -f "$BASE/scripts/$script" ]; then
    if bash -n "$BASE/scripts/$script" 2>/dev/null; then
      check_pass "$script (syntax OK)"
    else
      check_fail "$script (syntax error)"
    fi
  else
    check_fail "$script not found"
  fi
done
echo ""

# 5. Docker image
echo "5. Docker Environment"
if docker images slimerl/slime:v0.3.1 --format "{{.Repository}}:{{.Tag}}" | grep -q "slimerl/slime:v0.3.1"; then
  check_pass "Docker image: slimerl/slime:v0.3.1"
else
  check_fail "Docker image slimerl/slime:v0.3.1 not found"
fi
echo ""

# 6. GPU availability
echo "6. GPU Resources"
if command -v nvidia-smi &> /dev/null; then
  GPU_FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | awk -F', ' '$2 > 70000 {print $1}' | wc -l)
  if [ "$GPU_FREE" -ge 4 ]; then
    check_pass "GPUs available: ${GPU_FREE} with >70GB free (need 4)"
  else
    check_warn "GPUs available: ${GPU_FREE} with >70GB free (need 4)"
  fi
else
  check_warn "nvidia-smi not available (cannot check GPU status)"
fi
echo ""

# 7. HF checkpoint tools
echo "7. HF Checkpoint Tools"
if [ -f "$BASE/scripts/convert_checkpoint_to_hf.py" ]; then
  check_pass "Converter: scripts/convert_checkpoint_to_hf.py"
else
  check_fail "Converter not found"
fi

if [ -f "$BASE/scripts/e7_eval_checkpoint_incontainer.py" ]; then
  check_pass "Eval script: scripts/e7_eval_checkpoint_incontainer.py"
else
  check_fail "Eval script not found"
fi
echo ""

# Summary
echo "========================================="
echo -e "${GREEN}PASS${NC}: $PASS  ${YELLOW}WARN${NC}: $WARN  ${RED}FAIL${NC}: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo -e "${RED}Pre-flight FAILED${NC}: Fix $FAIL critical issues before launching E7"
  exit 1
elif [ "$WARN" -gt 0 ]; then
  echo -e "${YELLOW}Pre-flight PASSED with warnings${NC}: Review $WARN warnings"
  exit 0
else
  echo -e "${GREEN}Pre-flight PASSED${NC}: Ready for E7 formal training"
  exit 0
fi
